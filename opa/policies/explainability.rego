package policies.explainability

default allow = false

allow {
    input.explainability.metrics.explainability_score >= input.explainability.threshold.explainability
}   

allow {
    input.explainability.metrics.attention_score >= input.explainability.threshold.attention_score
    input.explainability.metrics.interpretability_score >= input.explainability.threshold.interpretability_score
}
