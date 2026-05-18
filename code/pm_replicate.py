#!/usr/bin/env python3
"""
PhysioMind Phase-2 — INDEPENDENT REPLICATION of H1 on Stress-Predict.

Independent cohort/sensor/protocol vs WESAD: Empatica E4 wrist PPG-IBI
(not chest ECG), Stroop/interview/hyperventilation (not TSST), 35 new
subjects. Real binary labels (no fabrication). Same honesty harness:
subject-grouped CV, 1000-perm null, raw/PCA/random baselines,
anti-circularity (numeric-only probe, never the label word).

H1 only (the one real positive worth replicating). A non-replication
is a valid, important result (WESAD-specific) — reported straight.
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np

ROOT = "/data/experiments/physiomind/replication_data/Stress-Predict-Dataset"
RAW = f"{ROOT}/Raw_data"
LBLCSV = f"{ROOT}/Processed_data/Improved_All_Combined_hr_rsp_binary.csv"
SAE_PATH = "/data/experiments/physiomind/data/sae_hrv_gpt2_layer6_v2.safetensors"
OUT = "/data/experiments/physiomind/replication_result.json"
WIN_S, STEP_S = 120, 60
N_PERM = 1000


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_labels():
    import pandas as pd
    df = pd.read_csv(LBLCSV)
    df.columns = [c.strip() for c in df.columns]
    # per participant: array of (unix_sec, label)
    out = {}
    for p, g in df.groupby("Participant"):
        t = g["Time(sec)"].to_numpy(float)
        y = g["Label"].to_numpy(int)
        o = np.argsort(t)
        out[int(p)] = (t[o], y[o])
    return out


def label_for(window_t0, window_t1, lt, ly):
    """Return 0/1 if the whole window is cleanly inside ONE label, else None."""
    m = (lt >= window_t0) & (lt <= window_t1)
    if m.sum() < max(5, 0.5 * (window_t1 - window_t0)):
        return None
    vals = ly[m]
    if vals.mean() == 0:
        return 0
    if vals.mean() == 1:
        return 1
    return None  # mixed -> ambiguous, drop (no fabrication)


def extract():
    import neurokit2 as nk
    labels = load_labels()
    smoke = os.environ.get("SMOKE") == "1"
    subs = sorted(glob.glob(f"{RAW}/S*"))
    if smoke:
        subs = subs[:4]
    rows = []
    matched_subj = 0
    for sd in subs:
        sid = os.path.basename(sd)              # S01..S35
        pnum = int(sid[1:])
        if pnum not in labels:
            continue
        ibi_f = os.path.join(sd, "IBI.csv")
        if not os.path.isfile(ibi_f):
            continue
        with open(ibi_f) as f:
            first = f.readline().split(",")
        try:
            start = float(first[0])
        except Exception:
            continue
        arr = np.genfromtxt(ibi_f, delimiter=",", skip_header=1)
        if arr.ndim != 2 or len(arr) < 50:
            continue
        beat_t = start + arr[:, 0]              # absolute unix sec
        rr_ms = arr[:, 1] * 1000.0              # IBI sec -> ms
        lt, ly = labels[pnum]
        t0, t1 = beat_t[0], beat_t[-1]
        nseg = 0
        s = t0
        while s + WIN_S <= t1:
            e = s + WIN_S
            yl = label_for(s, e, lt, ly)
            if yl is not None:
                msk = (beat_t >= s) & (beat_t < e)
                rr = rr_ms[msk]
                rr = rr[(rr > 300) & (rr < 2000)]  # physiologic RR
                if len(rr) >= 40:
                    try:
                        peaks = np.cumsum(rr) / 1000.0
                        pk_idx = (peaks * 1000).astype(int)
                        sig = np.zeros(pk_idx[-1] + 1)
                        sig[pk_idx] = 1
                        h = nk.hrv(sig, sampling_rate=1000, show=False)
                        feat = {k: float(h[k].iloc[0]) for k in
                                ("HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
                                 "HRV_pNN50", "HRV_LF", "HRV_HF",
                                 "HRV_LFHF", "HRV_SampEn")
                                if k in h.columns}
                        if len(feat) >= 6 and all(
                                np.isfinite(v) for v in feat.values()):
                            feat.update(subject=sid, y=int(yl))
                            rows.append(feat)
                            nseg += 1
                    except Exception:
                        pass
            s += STEP_S
        if nseg:
            matched_subj += 1
        log(f"  {sid}: usable windows={nseg} (cum {len(rows)})")
    log(f"matched subjects={matched_subj}  total windows={len(rows)}")
    return rows


def probe_text(f):
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
    from sklearn.model_selection import GroupKFold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    rows = extract()
    if len(rows) < 40:
        sys.exit(f"ABORT: only {len(rows)} windows — too few to test.")
    y = np.array([r["y"] for r in rows])
    grp = np.array([r["subject"] for r in rows])
    log(f"windows={len(y)} baseline={int((y==0).sum())} "
        f"stress={int((y==1).sum())} subjects={len(set(grp))}")
    if y.sum() < 20 or (y == 0).sum() < 20:
        sys.exit("ABORT: class imbalance too severe for honest test.")

    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    gpt2 = GPT2Model.from_pretrained("gpt2").to(dev).eval()
    sd = load_file(SAE_PATH)
    Wenc = sd["encoder.weight"].float().to(dev)
    benc = sd["encoder.bias"].float().to(dev)
    txts = [probe_text(r) for r in rows]
    RAWf, SAEf = [], []
    with torch.no_grad():
        for s in range(0, len(txts), 32):
            b = tok(txts[s:s+32], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(dev)
            h = gpt2(**b, output_hidden_states=True).hidden_states[6]
            mk = b["attention_mask"].unsqueeze(-1).float()
            act = (h*mk).sum(1)/mk.sum(1).clamp(min=1)
            RAWf.append(act.cpu().numpy())
            SAEf.append(torch.relu(act@Wenc.t()+benc).cpu().numpy())
    RAWf = np.concatenate(RAWf); SAEf = np.concatenate(SAEf)

    gkf = GroupKFold(n_splits=min(5, len(set(grp))))
    def cv(F, yy):
        oof = np.zeros(len(yy))
        for tr, te in gkf.split(F, yy, grp):
            mu, sg = F[tr].mean(0), F[tr].std(0)+1e-8
            c = LogisticRegression(max_iter=2000, C=0.5)
            c.fit((F[tr]-mu)/sg, yy[tr])
            oof[te] = c.predict_proba((F[te]-mu)/sg)[:, 1]
        return roc_auc_score(yy, oof), oof
    pca = PCA(n_components=min(64, RAWf.shape[1]),
              random_state=0).fit_transform(
              (RAWf-RAWf.mean(0))/(RAWf.std(0)+1e-8))
    rnd = RAWf @ rng.normal(0, 1, (RAWf.shape[1], 64)).astype(np.float32)
    res = {}
    a, oof = cv(SAEf, y); res["SAE_auc"] = float(a)
    res["rawGPT2_auc"] = float(cv(RAWf, y)[0])
    res["PCA_auc"] = float(cv(pca, y)[0])
    res["rand_auc"] = float(cv(rnd, y)[0])
    null = np.array([roc_auc_score(rng.permutation(y), oof)
                     for _ in range(N_PERM)])
    res["perm_p"] = float((1+np.sum(null >= a))/(1+N_PERM))
    res["perm_null95"] = float(np.percentile(null, 95))
    res["n_windows"] = len(y); res["n_subjects"] = len(set(grp))

    log("="*60)
    for k, v in res.items():
        log(f"  {k:16s} {v}")
    sae = res["SAE_auc"]
    base = max(res["rawGPT2_auc"], res["PCA_auc"], res["rand_auc"])
    p = res["perm_p"]
    if p >= 0.05:
        v = (f"NON-REPLICATION: grounding NOT found on independent cohort "
             f"(SAE {sae:.3f}, null95 {res['perm_null95']:.3f}, p={p:.3f}). "
             f"WESAD H1 may be cohort-specific. Honest, important.")
    elif sae <= base + 0.02:
        v = (f"REPLICATES (partial, as in WESAD): grounding holds "
             f"(p={p:.3f}) but SAE {sae:.3f} not beating baselines "
             f"{base:.3f} — same pattern, now cross-cohort/cross-sensor.")
    else:
        v = (f"REPLICATES + SAE-positive: SAE {sae:.3f} beats baselines "
             f"{base:.3f} & null (p={p:.3f}) on independent cohort. "
             f"Strong cross-cohort result.")
    log("VERDICT: " + v)
    json.dump(res, open(OUT, "w"), indent=2, default=str)
    log(f"written {OUT}")


if __name__ == "__main__":
    sys.exit(main())
