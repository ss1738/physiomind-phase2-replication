# PhysioMind Phase 2 — Cross-Modal Grounding, and Why It Didn't Replicate

**TL;DR — this is a negative result, reported honestly.** A biomedical
text model's internal features *appeared* to track real autonomic
physiology on one dataset (WESAD, AUC ≈ 0.73, leave‑one‑subject‑out
robust, p = 0.001). On **independent replication** with a different
cohort, sensor, and protocol (Stress‑Predict), the effect **collapsed
to chance** (AUC ≈ 0.47, p ≈ 0.91). The apparent signal was
cohort‑specific, not a general phenomenon. Two secondary hypotheses
were also falsified. Posted because a finding that dies under
independent replication — caught before anything is built on it — is
worth more than an impressive unreplicated number.

## The question

Does a model trained on biomedical *text* develop internal features
that correspond to **real measured autonomic physiology** (rest vs.
acute stress), as opposed to plausible-sounding correlations?

## Method (controls were the point)

- Features = sparse‑autoencoder (SAE) activations over a frozen
  GPT‑2's biomedical‑text representation.
- Physiology = real ECG → HRV metrics (NeuroKit2).
- **Anti‑circularity:** the text probe contains *only numeric HRV
  values* (RMSSD, LF/HF, …) — never the words "rest"/"stress". The
  model is never shown the label.
- **No leakage:** subject‑grouped cross‑validation; nucleus/window
  from one subject can never appear in both train and test.
- **Permutation null:** 1,000× label shuffles.
- **Baselines:** raw GPT‑2 activations, PCA, random projection — the
  SAE must beat these to mean anything.

## Results (`results/*.json`, fully reproducible)

| Test | Cohort | SAE AUC | best baseline | perm p |
|---|---|---|---|---|
| H1 GroupKFold | WESAD (15 subj) | 0.734 | rawGPT2 0.781 | 0.001 |
| H1 Leave‑one‑subject‑out | WESAD | 0.720 | rawGPT2 0.779 | 0.001 |
| **H1 INDEPENDENT REPLICATION** | **Stress‑Predict (34 subj)** | **0.467** | PCA 0.550 | **0.911** |

- **H1 (grounding):** looked real on WESAD (above the permuted null),
  but **did not replicate** on an independent cohort/sensor/protocol —
  chance level, p ≈ 0.91. **Cohort‑specific artifact, not a general
  effect.** Also: even on WESAD the SAE never beat raw activations/PCA.
- **H2 ("unnamed SAE features = undiscovered concepts"):** *falsified.*
  Unnamed features add ≈ 0 over named‑concept features (increment p ≈ 1.0).
- **H3 (physiological specificity):** features are generic‑arousal,
  not vagal‑vs‑sympathetic specific (Jaccard ≈ 0.67).

## Honest conclusion

There is **no validated cross‑modal grounding**. The one apparent
positive was a single‑cohort artifact that failed independent
replication; the SAE adds nothing over trivial baselines; the
"undiscovered concept" hypothesis is falsified. This repo exists so
the claim and the evidence cannot diverge.

## Reproduce

```
code/pm_phase2.py        # WESAD H1/H2/H3, GroupKFold
code/pm_phase2_loso.py   # WESAD, leave-one-subject-out
code/pm_replicate.py     # independent replication on Stress-Predict
```
Datasets: WESAD (open) and Stress‑Predict (MIT‑licensed, open). Results
in `results/`; figures in `figures/`.

## Why publish a negative

Generating a plausible result is easy; one that survives independent
replication is the actual problem. This didn't survive — and finding
that out *before* building on it is the result that matters.
