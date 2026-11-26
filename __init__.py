"""Caption Reliability Inspector - a simple, single-process live demo built on the repository's own modules.

It reuses src/ (BLIP predictor, teacher-forced replay, cross-attention extraction, Grounding DINO detector, frozen OpenCLIP
region encoder, three-gate eligibility filter, hard-negative mining, InfoNCE loss, token projection f_s, detector-evidence
reliability rule) and adds only what the demo needs: token-to-region attention mass, a zero-shot OpenCLIP inspection
signal, a fixed-region counterfactual, a controlled single-token optimisation of a fresh f_s, cached replay and a web page.
"""
