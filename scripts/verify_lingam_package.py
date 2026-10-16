# scripts/verify_lingam_package.py
import numpy as np
from lingam import DirectLiNGAM
from data.sachs import load_sachs, SACHS_COLUMNS
from evaluation import ancestral_accuracy

X, true_adj = load_sachs()

X_log = np.log1p(X - X.min(axis=0) + 1)

for name, model, data in [
    ("DirectLiNGAM (raw, default)", DirectLiNGAM(), X),
    ("DirectLiNGAM (log, default)", DirectLiNGAM(), X_log),
    ("DirectLiNGAM (log, pwling)", DirectLiNGAM(measure='pwling'), X_log),
    ("DirectLiNGAM (raw, pwling)", DirectLiNGAM(measure='pwling'), X),
]:
    model.fit(data)
    aa = ancestral_accuracy(true_adj, model.causal_order_)
    order_names = [SACHS_COLUMNS[i] for i in model.causal_order_]
    print(f"{name:<35} | AA={aa:.3f} | {order_names}")