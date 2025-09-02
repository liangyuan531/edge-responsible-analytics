package policies.reliability

default allow = false

# For vision models
allow {
    input.reliability.metrics.reliability_score >= input.reliability.threshold.reliability_score
}

# For text models
allow {
    input.reliability.metrics.prediction_stability >= input.reliability.threshold.prediction_stability
}