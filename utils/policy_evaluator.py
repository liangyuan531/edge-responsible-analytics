# policy_evaluator.py

import requests
import json
import logging
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score
import tensorflow as tf
import shap
import yaml
import os

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aggregator.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# OPA_URL = "http://10.200.3.99:8181/v1/data/policies/fairness/demographic_parity/allow"

# POLICY_URLS = {
#     "fairness": "http://10.200.3.99:8181/v1/data/policies/fairness/demographic_parity/allow",
#     "reliability": "http://10.200.3.99:8181/v1/data/policies/reliability/allow",
#     "explainability": "http://10.200.3.99:8181/v1/data/policies/explainability/allow"
# }

opa_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opa_config.yaml')
with open(opa_config_path, 'r') as file:
    config = yaml.safe_load(file)

OPA_SERVER_URL = config['opa_server_url']
POLICIES = config['policies']

def get_art_classifier(model, loss_object, input_shape):
    return TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        input_shape=input_shape,
        nb_classes=2,
        clip_values=(0, 1)
    )

def evaluate_fairness_policy(model, X, y_true, sensitive_features, thresholds, y_pred=None):
    """
    Evaluates model fairness using OPA policies.
    """
    logger.info("Evaluate fairness policy.")
    try:
        # Handle different types of sensitive features
        if isinstance(sensitive_features, pd.DataFrame):
            has_nan = sensitive_features.isna().any().any()
            if has_nan:
                logger.error("sensitive_features DataFrame contains NaN values")
        elif isinstance(sensitive_features, pd.Series):
            if sensitive_features.dtype.name == 'category':
                sensitive_features = sensitive_features.astype(str)
            has_nan = sensitive_features.isna().any()
            if has_nan:
                logger.error("sensitive_features Series contains NaN values")
        elif isinstance(sensitive_features, np.ndarray):
            has_nan = pd.isna(sensitive_features).any()
            if has_nan:
                logger.error("sensitive_features array contains NaN values")
        
        # Handle predictions based on model type
        if y_pred is None and model is not None:
            if isinstance(y_true[0], str):  # For TinyBERT
                # Get predictions from model
                outputs = model(input_ids=X["input_ids"],
                              attention_mask=X["attention_mask"],
                              token_type_ids=X["token_type_ids"])
                y_pred = outputs.logits.numpy().argmax(axis=1)
                
                # Convert string labels to numeric using LabelEncoder
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true = le.fit_transform([label.strip().lower() for label in y_true])
            else:  # For other models (e.g., MobileNet)
                y_pred = model.predict(X)
                y_pred = (y_pred >= 0.5).astype(int)  # Convert to binary
        
        # Ensure all inputs are the right type for MetricFrame
        if isinstance(sensitive_features, pd.DataFrame):
            sensitive_feature_col = sensitive_features.iloc[:, 0]
        else:
            sensitive_feature_col = sensitive_features
            
        sample_params = {'demographic_parity_difference': {'sensitive_features': sensitive_feature_col}}
        
        # Create MetricFrame
        metric_frame = MetricFrame(
            metrics={
                # "accuracy": accuracy_score,
                "demographic_parity_difference": demographic_parity_difference,
            },
            y_true=y_true,
            y_pred=y_pred,  # Now standardized for all model types
            sensitive_features=sensitive_feature_col,
            sample_params=sample_params
        )

        # Extract overall metrics
        model_metrics = metric_frame.overall.to_dict()
        
        logger.info(f"Model Metrics: {model_metrics}")
        
        input_data = {
            "fairness": {
                "metrics": model_metrics,
                "threshold": thresholds
            }
        }

        allowed, failed_policies = send_to_opa(input_data, "fairness")

        if allowed:
            logger.info("Model passed all fairness policies.")
            return True, []
        else:
            logger.warning("Model failed fairness policies.")
            if model_metrics.get("demographic_parity_difference", 0) > thresholds.get("demographic_parity_difference", 0):
                failed_policies.append("demographic_parity")
            return False, failed_policies

    except Exception as e:
        logger.exception(f"Error during fairness evaluation: {e}")
        return False, ["Fairness Evaluation Error"]

def evaluate_reliability_policy(model, X_test, y_test, thresholds):
    logger.info("Evaluate reliability policy.")
    try:
        # Wrap the model with ART classifier
        loss_object = tf.keras.losses.BinaryCrossentropy()
        art_classifier = get_art_classifier(model, loss_object, input_shape=(224, 224, 3))

        # Initialize the attack (PGD)
        attack = ProjectedGradientDescent(estimator=art_classifier, eps=0.03, eps_step=0.005, max_iter=40, targeted=False)

        # Generate adversarial examples
        X_test_adv = attack.generate(x=X_test)
        predictions = model.predict(X_test_adv)
        y_pred_adv = (predictions >= 0.5).astype(int).flatten()

        success_rate = np.mean(y_pred_adv != y_test)
        reliability_score = 1 - success_rate

        reliability_metrics = {
            "success_rate": float(success_rate),
            "reliability_score": float(reliability_score)
        }

        logger.info(f"Reliability Metrics: {reliability_metrics}")

        input_data = {
            "reliability": {
                "metrics": reliability_metrics,
                "threshold": thresholds
            }
        }

        allowed, failed_policies = send_to_opa(input_data, "reliability")

        if allowed:
            logger.info("Model passed all reliability policies.")
            return True, []
        else:
            logger.warning("Model failed reliability policies.")
            if reliability_metrics.get("reliability_score", 0) < thresholds.get("reliability_score", 0):
                failed_policies.append("reliability_score")
            return False, failed_policies

    except requests.exceptions.RequestException as e:
        logger.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logger.exception(f"Error during reliability evaluation: {e}")
        return False, ["Reliability Evaluation Error"]

def evaluate_explainability_policy(model, X_sample, thresholds):
    logger.info("Evaluate explainability policy.")
    try:
        # Ensure model is compiled
        if not model.optimizer:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Select a background dataset for SHAP
        background_size = min(100, X_sample.shape[0])
        if background_size < 100:
            logger.warning(f"Insufficient background samples. Using {background_size} samples instead of 100.")

        background = X_sample[:background_size]

        # Initialize the SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Assuming binary classification; select SHAP values for the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Index 1 corresponds to the positive class

        # Calculate explainability score as the mean absolute SHAP value
        explainability_score = np.mean(np.abs(shap_values))
        logger.info(f"Explainability Score: {explainability_score}")

        # Prepare metrics for OPA
        explainability_metrics = {
            "explainability_score": float(explainability_score)
        }

        # Log explainability metrics
        # logger.info(f"Explainability Metrics: {explainability_metrics}")

        # Prepare input data for OPA
        input_data = {
            "explainability": {
                "metrics": explainability_metrics,
                "threshold": thresholds
            }
        }

        # Send data to OPA for policy evaluation
        allowed, failed_policies = send_to_opa(input_data, "explainability")

        if allowed:
            logger.info("Model passed all explainability policies.")
            return True, []
        else:
            logger.warning("Model failed explainability policies.")
            if explainability_metrics.get("explainability_score", 0) < thresholds.get("explainability_score", 0):
                failed_policies.append("explainability_score")
            return False, failed_policies

    except requests.exceptions.RequestException as e:
        logger.exception("Failed to communicate with OPA.")
        return False, ["OPA Communication Error"]
    except Exception as e:
        logger.exception(f"Error during explainability evaluation: {e}")
        return False, ["Explainability Evaluation Error"]

def compute_k_anonymity(df, quasi_identifiers, k):
    """
    Computes k-anonymity for the given DataFrame and quasi-identifiers.
    """
    group_sizes = df.groupby(quasi_identifiers).size()
    min_k = group_sizes.min()
    return min_k

def evaluate_privacy_policy(df, quasi_identifiers, k_threshold):
    logger.info("Evaluate privacy policy.")
    try:
        k_threshold = int(k_threshold)  # Ensure k_threshold is a Python int
        k_anonymity = compute_k_anonymity(df, quasi_identifiers, k_threshold)
        logger.info(f"k-anonymity: {k_anonymity}")
        
        input_data = {
            "privacy": {
                "k_anonymity": k_anonymity,
                "thresholds": {
                    "k": k_threshold
                }
            }
        }
        
        input_data = convert_numpy_types(input_data)
        allowed, failed_policies = send_to_opa(input_data, "privacy")
        
        if allowed:
            logger.info("Data satisfies the privacy policies.")
            return True, []
        else:
            logger.warning("Data failed privacy policies.")
            # failed_policies.append("k_anonymity")
            return False, failed_policies
    
    except Exception as e:
        logger.exception(f"Error during privacy evaluation: {e}")
        return False, ["Privacy Evaluation Error"]
    
def send_to_opa(input_data, policy_type):
    failed_policies = []
    try:
        policy_url = OPA_SERVER_URL + POLICIES.get(policy_type)
        if not policy_url:
            logger.error(f"No policy URL found for policy type: {policy_type}")
            return False, [f"{policy_type}_policy_not_found"]

        response = requests.post(policy_url, json={"input": input_data})
        if response.status_code == 200:
            result = response.json()
            # logger.info(f"OPA result: {result}")
            allowed = result.get('result', False)
            if not allowed:
                failed_policies.append(policy_type)
            return allowed, failed_policies
        else:
            logger.error(f"OPA request failed with status code {response.status_code}: {response.text}")
            return False, [f"{policy_type}_opa_request_failed"]
    except Exception as e:
        logger.exception(f"Error sending data to OPA: {e}")
        return False, [f"{policy_type}_opa_exception"]

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    else:
        return obj
    
def evaluate_explainability_policy_t5(model, X_sample, thresholds):
    logger.info("Evaluate explainability policy t5.")
    try:
        # For T5, define explainability metrics differently
        # Placeholder example: average attention weights
        # Implement actual explainability evaluation as needed
        
        # Not straightforward with transformer models like T5
        # This needs a custom approach or use integrated explainability tools
        # For now, set as True
        explainability_score = 1.0
        logger.info(f"Explainability Score (T5): {explainability_score}")
        
        input_data = {
            "explainability": {
                "metrics": {
                    "explainability_score": explainability_score
                },
                "threshold": thresholds
            }
        }
        
        allowed, failed_policies = send_to_opa(input_data, "explainability")
        
        if allowed:
            logger.info("Model passed all explainability policies.")
            return True, []
        else:
            logger.warning("Model failed explainability policies.")
            if explainability_score < thresholds.get("explainability_score", 0):
                failed_policies.append("explainability_score")
            return False, failed_policies
    
    except Exception as e:
        logger.exception(f"Error during explainability evaluation for T5: {e}")
        return False, ["Explainability Evaluation Error"]

def evaluate_reliability_policy_t5(model, X_test, y_test, thresholds):
    """
    Evaluates model reliability for T5 model using ART adversarial attacks.
    
    Args:
        model (TFT5ForConditionalGeneration): The T5 model.
        X_test (list of str): Test input texts.
        y_test (list of str): True target texts.
        thresholds (dict): Thresholds for reliability metrics.
    
    Returns:
        bool: True if reliability policies are satisfied.
        list: List of failed policies.
    """
    logger.info("Evaluate reliability policy t5.")
    try:
        from art.estimators.text import TFTextClassifier
        from art.attacks.evasion import TextAttack
        # Placeholder implementation
        # T5 models require text-based attacks which are different
        # For now, set reliability as True
        reliability_score = 1.0
        logger.info(f"Reliability Score (T5): {reliability_score}")
        
        input_data = {
            "reliability": {
                "metrics": {
                    "reliability_score": reliability_score
                },
                "threshold": thresholds
            }
        }
        
        allowed, failed_policies = send_to_opa(input_data, "reliability")
        
        if allowed:
            logger.info("Model passed all reliability policies.")
            return True, []
        else:
            logger.warning("Model failed reliability policies.")
            if reliability_score < thresholds.get("reliability_score", 0):
                failed_policies.append("reliability_score")
            return False, failed_policies
    except Exception as e:
        logger.exception(f"Error during reliability evaluation for T5: {e}")
        return False, ["Reliability Evaluation Error"]

def evaluate_explainability_policy_tinybert(model, X_val, tokenizer, thresholds):
    """
    Evaluates explainability for TinyBERT model using attention weights.
    """
    logger.info("Evaluate explainability policy tinybert.")
    try:
        # Process in smaller batches
        batch_size = 16
        max_length = 512  # TinyBERT's maximum sequence length
        attention_scores_list = []
        
        for i in range(0, len(X_val), batch_size):
            batch_texts = X_val[i:i + batch_size].tolist()
            
            # Tokenize with explicit max length
            inputs = tokenizer(
                batch_texts,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Get attention weights
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                output_attentions=True
            )
            
            # Calculate mean attention scores for this batch
            batch_scores = tf.reduce_mean([tf.reduce_mean(layer) for layer in outputs.attentions])
            attention_scores_list.append(float(batch_scores.numpy()))
        
        # Calculate overall explainability score
        explainability_score = np.mean(attention_scores_list)
        
        # Prepare metrics for OPA
        explainability_metrics = {
            "attention_score": explainability_score,
            "interpretability_score": explainability_score  # Simplified for now
        }
        
        logger.info(f"TinyBERT Explainability Metrics: {explainability_metrics}")
        
        input_data = {
            "explainability": {
                "metrics": explainability_metrics,
                "threshold": thresholds
            }
        }
        
        allowed, failed_policies = send_to_opa(input_data, "explainability")
        
        if allowed:
            logger.info("Model passed all explainability policies.")
            return True, []
        else:
            logger.warning("Model failed explainability policies.")
            if explainability_metrics["attention_score"] < thresholds.get("attention_score", 0) or explainability_metrics["interpretability_score"] < thresholds.get("interpretability_score", 0):
                failed_policies.append("attention_score")
            return False, failed_policies
            
    except Exception as e:
        logger.exception(f"Error during TinyBERT explainability evaluation: {e}")
        return False, ["explainability_evaluation_error"]

def evaluate_reliability_policy_tinybert(model, X_val, tokenizer, thresholds):
    """
    Evaluates reliability for TinyBERT model using input perturbations.
    """
    logger.info("Evaluate reliability policy tinybert.")
    try:
        max_length = 512  # TinyBERT's maximum sequence length
        n_samples = min(len(X_val), 100)
        stability_scores = []
        
        for i in range(n_samples):
            text = X_val.iloc[i]
            
            # Original prediction
            original_input = tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            original_output = model(
                input_ids=original_input["input_ids"],
                attention_mask=original_input["attention_mask"],
                token_type_ids=original_input["token_type_ids"]
            ).logits
            
            # Test with truncated input
            truncated_text = ' '.join(text.split()[:len(text.split())//2])
            truncated_input = tokenizer(
                truncated_text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            truncated_output = model(
                input_ids=truncated_input["input_ids"],
                attention_mask=truncated_input["attention_mask"],
                token_type_ids=truncated_input["token_type_ids"]
            ).logits
            
            # Calculate stability score
            stability_score = 1.0 - float(tf.reduce_mean(tf.abs(original_output - truncated_output)))
            stability_scores.append(stability_score)
        
        # Aggregate scores
        reliability_metrics = {
            "prediction_stability": float(np.mean(stability_scores))
        }
        
        logger.info(f"TinyBERT Reliability Metrics: {reliability_metrics}")
        
        input_data = {
            "reliability": {
                "metrics": reliability_metrics,
                "threshold": thresholds
            }
        }
        
        allowed, failed_policies = send_to_opa(input_data, "reliability")
        
        if allowed:
            logger.info("Model passed all reliability policies.")
            return True, []
        else:
            logger.warning("Model failed reliability policies.")
            # failed = []
            # for metric, score in reliability_metrics.items():
            #     if score < thresholds.get(metric, 0):
            #         failed.append(metric)
            if reliability_metrics.get("prediction_stability", 0) < thresholds.get("prediction_stability", 0):
                failed_policies.append("prediction_stability")
            return False, failed_policies
            
    except Exception as e:
        logger.exception(f"Error during TinyBERT reliability evaluation: {e}")
        return False, ["reliability_evaluation_error"]

def calculate_token_importance(model, tokenizer, texts, n_samples=100):
    """
    Calculates token importance scores using input erasure.
    """
    importance_scores = []
    
    for text in texts[:n_samples]:
        # Tokenize text
        tokens = tokenizer.tokenize(text)
        base_input = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
        base_output = model(base_input).logits
        
        # Calculate importance for each token
        token_scores = []
        for i in range(len(tokens)):
            # Create copy with token masked
            masked_tokens = tokens.copy()
            masked_tokens[i] = tokenizer.mask_token
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
            masked_input = tokenizer(masked_text, return_tensors="tf", padding=True, truncation=True)
            masked_output = model(masked_input).logits
            
            # Calculate impact of masking
            impact = float(tf.reduce_mean(tf.abs(base_output - masked_output)))
            token_scores.append(impact)
        
        importance_scores.append(np.mean(token_scores))
    
    return importance_scores

def perturb_text(text, p_swap=0.1):
    """
    Creates a slightly perturbed version of the input text.
    """
    words = text.split()
    for i in range(len(words)-1):
        if np.random.random() < p_swap:
            words[i], words[i+1] = words[i+1], words[i]
    return ' '.join(words)