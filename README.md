# PhysioMind Phase 2

This is a negative result. I'm putting it up with the code and the
numbers so the claim and the evidence stay in the same place.

## The question

A model trained on biomedical text builds internal features. Some line
up with known concepts, some don't. The question for Phase 2 was
whether any of those features actually track real physiology (rest vs
acute stress), or just produce plausible-sounding correlations.

## What I did

Features are sparse-autoencoder activations over a frozen GPT-2's
text representation. Physiology is real ECG turned into HRV metrics
with NeuroKit2.

The controls were the point:

* The text probe only ever contains numeric HRV values (RMSSD, LF/HF,
  and so on). It never sees the words "rest" or "stress". So the model
  can't just read the label.
* Subject-grouped splits. A subject in train is never in test.
* 1,000x permutation null.
* Baselines: raw GPT-2 activations, PCA, random projection. The SAE
  has to beat these to mean anything.

## Results

All numbers are in `results/`.

| Test | Cohort | SAE AUC | best baseline | perm p |
|------|--------|---------|---------------|--------|
| H1 GroupKFold | WESAD (15 subj) | 0.734 | rawGPT2 0.781 | 0.001 |
| H1 leave-one-subject-out | WESAD | 0.720 | rawGPT2 0.779 | 0.001 |
| H1 independent replication | Stress-Predict (34 subj) | 0.467 | PCA 0.550 | 0.911 |

On WESAD it looked real. It was above the permutation null and held
under leave-one-subject-out. Even there it never beat raw activations
or PCA, but it looked like something.

Then I ran the same pipeline on a different dataset. Different people,
different sensor, different stress protocol. It dropped to chance,
p around 0.9. The WESAD effect was specific to that cohort, not a
general thing.

Two more results:

* H2: the idea that the unnamed features are undiscovered concepts.
  False. Unnamed features add about zero over named ones (p around 1.0).
* H3: the features are generic arousal, not vagal vs sympathetic
  specific.

## Conclusion

There is no validated cross-modal grounding here. The one positive was
a single-cohort artifact that failed independent replication. The SAE
adds nothing over trivial baselines. The undiscovered-concept idea is
false. The repo exists so the claim can't drift away from the data.

## Reproduce

```
code/pm_phase2.py        WESAD H1/H2/H3, GroupKFold
code/pm_phase2_loso.py   WESAD, leave-one-subject-out
code/pm_replicate.py     independent replication on Stress-Predict
```

WESAD is open. Stress-Predict is MIT-licensed and open. Results are
in `results/`, reproducible from the scripts in `code/`.

A result that doesn't survive an independent dataset, caught before
anything gets built on it, is the one worth keeping.
