package policies.privacy

default allow = false

allow {
    input.privacy.k_anonymity >= input.privacy.thresholds.k
}