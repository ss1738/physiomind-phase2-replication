#!/usr/bin/env python3
"""
PhysioMind Phase 2, leave-one-subject-out variant: do text-SAE
features track real autonomic physiology (WESAD chest ECG), or just
look like they do?

H1  Text-SAE features predict real rest vs stress (from ECG HRV)
    above a permutation null and above raw-activation, PCA and
    random-projection baselines.
H2  Unnamed SAE features add predictive value beyond named-concept
    features (nested, permutation-controlled). This is the
    "undiscovered concept" idea, made falsifiable.
H3  Features that predict vagal HRV (RMSSD/HF) are distinct from
    those that predict sympatho-vagal balance (LF/HF), i.e.
    physiological specificity rather than one generic arousal blob.

Controls:
  * Leave-one-subject-out CV. The strictest subject-grouped split:
    every subject is held out as the test set in turn.
  * Anti-circularity: the text probe contains only numeric HRV
    values. The words stress, rest and baseline never appear.
  * Permutation null, raw/PCA/random baselines, nested comparison.
  * Smoke-gated. A clean negative is a real result.
"""
from __future__ import annotations
import glob, json, os, pickle, sys, time
import numpy as np

WESAD = "/data/experiments/physiomind/phase2_data/wesad"
SAE_PATH = "/data/experiments/physiomind/data/sae_hrv_gpt2_layer6_v2.safetensors"
OUT = "/data/experiments/physiomind/phase2_loso_result.json"
FS = 700                      # WESAD chest ECG sampling rate
WIN_S, STEP_S = 120, 60       # 120s windows, 60s hop
LBL = {1: 0, 2: 1}            # WESAD: 1=baseline(rest)->0, 2=stress->1
N_PERM = 1000


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def extract_hrv():
    import neurokit2 as nk
    rows = []
    pkls = sorted(glob.glob(f"{WESAD}/**/S*.pkl", recursive=True))
    if os.environ.get("SMOKE") == "1":
        pkls = pkls[:3]
    log(f"WESAD subjects: {len(pkls)}")
    for p in pkls:
        d = pickle.load(open(p, "rb"), encoding="latin1")
        subj = str(d.get("subject"))
        ecg = np.asarray(d["signal"]["chest"]["ECG"]).reshape(-1)
        lab = np.asarray(d["label"]).reshape(-1)
        for raw_lbl, y in LBL.items():
            m = lab == raw_lbl
            if m.sum() < WIN_S * FS:
                continue
            # contiguous run(s) of this label
            idx = np.where(m)[0]
            seg = ecg[idx[0]:idx[-1] + 1]
            w, s = WIN_S * FS, STEP_S * FS
            for st in range(0, len(seg) - w + 1, s):
                chunk = seg[st:st + w]
                try:
                    sig, _ = nk.ecg_process(chunk, sampling_rate=FS)
                    h = nk.hrv(sig, sampling_rate=FS, show=False)
                    feat = {k: float(h[k].iloc[0]) for k in
                            ("HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
                             "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF",
                             "HRV_SampEn") if k in h.columns}
                    if len(feat) < 6 or any(not np.isfinite(v)
                                            for v in feat.values()):
                        continue
                    feat.update(subject=subj, y=y)
                    rows.append(feat)
                except Exception:
                    continue
        log(f"  {subj}: cumulative windows={len(rows)}")
    return rows


def probe_text(f):
    # numeric values only; no condition word ever appears (anti-circularity)
    return (f"Cardiac interbeat recording. Mean RR interval "
            f"{f.get('HRV_MeanNN',0):.0f} milliseconds. RMSSD "
            f"{f.get('HRV_RMSSD',0):.0f} milliseconds. SDNN "
            f"{f.get('HRV_SDNN',0):.0f} milliseconds. pNN50 "
            f"{f.get('HRV_pNN50',0):.1f} percent. Low frequency power "
            f"{f.get('HRV_LF',0):.3f}. High frequency power "
            f"{f.get('HRV_HF',0):.3f}. LF to HF ratio "
            f"{f.get('HRV_LFHF',0):.2f}. Sample entropy "
            f"{f.get('HRV_SampEn',0):.2f}.")


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import GPT2Model, GPT2Tokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut as _LOGO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    rows = extract_hrv()
    if len(rows) < 40:
        sys.exit(f"ABORT: only {len(rows)} HRV windows, too few.")
    y = np.array([r["y"] for r in rows])
    grp = np.array([r["subject"] for r in rows])
    log(f"windows={len(y)} rest={int((y==0).sum())} stress={int((y==1).sum())}"
        f" subjects={len(set(grp))}")

    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    gpt2 = GPT2Model.from_pretrained("gpt2").to(dev).eval()
    sd = load_file(SAE_PATH)
    Wenc = sd["encoder.weight"].float().to(dev)
    benc = sd["encoder.bias"].float().to(dev)

    def encode(texts):
        raw, sae = [], []
        with torch.no_grad():
            for s in range(0, len(texts), 32):
                b = tok(texts[s:s+32], return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(dev)
                h = gpt2(**b, output_hidden_states=True).hidden_states[6]
                mk = b["attention_mask"].unsqueeze(-1).float()
                act = (h*mk).sum(1)/mk.sum(1).clamp(min=1)
                raw.append(act.cpu().numpy())
                sae.append(torch.relu(act@Wenc.t()+benc).cpu().numpy())
        return np.concatenate(raw), np.concatenate(sae)

    RAW, SAE = encode([probe_text(r) for r in rows])
    log(f"RAW={RAW.shape} SAE={SAE.shape} active={(SAE>0).any(0).sum()}")

    # named vs unnamed features: concept phrases vs neutral controls
    concept = ["vagal tone", "parasympathetic nervous activity",
               "sympathetic activation", "baroreflex sensitivity",
               "cardiac autonomic regulation", "heart rate variability"]
    control = ["the weather today", "a wooden table", "quarterly revenue",
               "a blue bicycle", "grammar of a sentence", "a mountain trail"]
    _, Sc = encode(concept)
    _, Su = encode(control)
    named_mask = (Sc.mean(0) - Su.mean(0)) > (Su.std(0) + 1e-6)
    log(f"named(concept-aligned) feats={int(named_mask.sum())} / {SAE.shape[1]}"
        f"  unnamed={int((~named_mask).sum())}")

    gkf = _LOGO()

    def cv_auc(F, yy=None):
        yy = y if yy is None else yy
        oof = np.zeros(len(yy))
        for tr, te in gkf.split(F, yy, grp):
            mu, sg = F[tr].mean(0), F[tr].std(0)+1e-8
            c = LogisticRegression(max_iter=2000, C=0.5)
            c.fit((F[tr]-mu)/sg, yy[tr])
            oof[te] = c.predict_proba((F[te]-mu)/sg)[:, 1]
        return roc_auc_score(yy, oof), oof

    res = {}
    pca = PCA(n_components=min(64, RAW.shape[1]),
              random_state=0).fit_transform((RAW-RAW.mean(0))/(RAW.std(0)+1e-8))
    randF = RAW @ rng.normal(0, 1, (RAW.shape[1], 64)).astype(np.float32)

    # H1
    a_sae, oof_sae = cv_auc(SAE)
    res["H1_SAE_auc"] = float(a_sae)
    res["H1_rawGPT2_auc"] = float(cv_auc(RAW)[0])
    res["H1_PCA_auc"] = float(cv_auc(pca)[0])
    res["H1_rand_auc"] = float(cv_auc(randF)[0])
    null = np.array([roc_auc_score(rng.permutation(y), oof_sae)
                     for _ in range(N_PERM)])
    res["H1_perm_p"] = float((1+np.sum(null >= a_sae))/(1+N_PERM))
    res["H1_perm_null95"] = float(np.percentile(null, 95))

    # H2 nested: named-only vs named+unnamed
    if named_mask.sum() >= 3 and (~named_mask).sum() >= 3:
        a_named = cv_auc(SAE[:, named_mask])[0]
        a_full, oof_full = cv_auc(SAE)            # named+unnamed = all
        res["H2_named_only_auc"] = float(a_named)
        res["H2_named_plus_unnamed_auc"] = float(a_full)
        # permutation null on the INCREMENT (shuffle which feats are 'unnamed')
        incr = a_full - a_named
        nulli = []
        allf = np.arange(SAE.shape[1])
        for _ in range(200):
            fake_named = rng.choice(allf, int(named_mask.sum()), False)
            nulli.append(cv_auc(SAE[:, fake_named])[0])
        res["H2_increment"] = float(incr)
        res["H2_increment_p"] = float(
            (1+np.sum((a_full-np.array(nulli)) >= incr))/(1+len(nulli)))
    else:
        res["H2"] = "skipped (insufficient named/unnamed split)"

    # H3 discriminant: vagal (RMSSD/HF) vs sympatho-vagal (LFHF) feature sets
    rm = np.array([r.get("HRV_RMSSD", np.nan) for r in rows])
    lh = np.array([r.get("HRV_LFHF", np.nan) for r in rows])
    def topcorr(target, k=50):
        t = (target-np.nanmean(target))/(np.nanstd(target)+1e-9)
        c = np.array([abs(np.corrcoef(SAE[:, j], t)[0, 1])
                      if SAE[:, j].std() > 0 else 0
                      for j in range(SAE.shape[1])])
        return set(np.argsort(-c)[:k].tolist())
    vag, sym = topcorr(rm), topcorr(lh)
    jac = len(vag & sym)/max(1, len(vag | sym))
    res["H3_vagal_vs_lfhf_jaccard"] = float(jac)  # low => specific/distinct

    log("="*62)
    for k, v in res.items():
        log(f"  {k:28s} {v}")
    sae, base = res["H1_SAE_auc"], max(res["H1_rawGPT2_auc"],
                                       res["H1_PCA_auc"], res["H1_rand_auc"])
    p = res["H1_perm_p"]
    if p >= 0.05:
        v1 = (f"H1 NEGATIVE: SAE not above permuted null "
              f"(AUC {sae:.3f}, null95 {res['H1_perm_null95']:.3f}, p={p:.3f})"
              f". Text-concept features do NOT track real autonomic state.")
    elif sae <= base + 0.02:
        v1 = (f"H1 PARTIAL: SAE above null (p={p:.3f}) but not beating "
              f"baselines ({sae:.3f} vs {base:.3f}). Grounding exists but "
              f"no SAE-specific value.")
    else:
        v1 = (f"H1 POSITIVE: SAE {sae:.3f} beats baselines {base:.3f} and "
              f"null (p={p:.3f}). Text-concept features genuinely track "
              f"real autonomic physiology.")
    log("VERDICT H1: " + v1)
    if "H2_increment" in res:
        i, ip = res["H2_increment"], res["H2_increment_p"]
        log("VERDICT H2: " + (
            f"POSITIVE: unnamed features add real predictive value "
            f"(+{i:.3f}, p={ip:.3f}); the 'undiscovered concept' thesis is "
            f"SUPPORTED on real physiology." if (i > 0.02 and ip < 0.05) else
            f"NEGATIVE: unnamed features add no value beyond named "
            f"concepts (+{i:.3f}, p={ip:.3f}); thesis NOT supported. Valid "
            f"result."))
    log(f"VERDICT H3: vagal/LFHF feature-set Jaccard="
        f"{res['H3_vagal_vs_lfhf_jaccard']:.3f} "
        f"({'distinct (specific)' if res['H3_vagal_vs_lfhf_jaccard']<0.3 else 'overlapping (generic arousal)'})")
    json.dump(res, open(OUT, "w"), indent=2, default=str)
    log(f"written {OUT}")


if __name__ == "__main__":
    sys.exit(main())
