"""Evaluation and explanation helpers shared by the Bitcoin and ERC-20 notebooks."""

from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42


def harmonise(frames: dict[str, pd.DataFrame], fraud_sources: list[str], key: str) -> pd.DataFrame:
    """Concatenate labelled sources applying the conservative fraud rule.

    ``frames`` maps a source name to a table that contains ``key`` (the entity
    identifier). Rows from fraud sources get ``y = 1``. A licit row is dropped
    when its identifier also appears in any fraud source; repeated identifiers
    are kept once.
    """
    parts = []
    for name, df in frames.items():
        part = df.copy()
        part["source"] = name
        part["y"] = int(name in fraud_sources)
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    fraud_ids = set(data.loc[data["y"] == 1, key])
    data = data[(data["y"] == 1) | ~data[key].isin(fraud_ids)]
    return data.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def scores(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Macro-averaged metrics plus the class-1 (fraud) operating figures."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": p, "recall": r, "f1": f,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_fraud": f1_score(y_true, y_pred, zero_division=0),
        "f2_fraud": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["auc_pr"] = average_precision_score(y_true, y_score)
    return out


# --------------------------------------------------------------------------
# CIU
# --------------------------------------------------------------------------

def ciu_means(model, X_train: pd.DataFrame, X_explain: pd.DataFrame, nsamples: int = 100) -> pd.DataFrame:
    """Mean contextual importance and utility of every feature for class 1."""
    from ciu import CIU

    X_train, X_explain = X_train.astype(float), X_explain.astype(float)  # py-ciu writes floats in place
    explainer = CIU(predictor=lambda a: model.predict_proba(np.atleast_2d(a)),
                    out_names=["licit", "fraud"], data=X_train, input_names=list(X_train.columns),
                    nsamples=nsamples, output_inds=[1])
    rows = []
    for i in range(len(X_explain)):
        with warnings.catch_warnings():  # py-ciu warns once per perturbed instance
            warnings.simplefilter("ignore")
            res = explainer.explain(instance=X_explain.iloc[[i]])
        for _, r in res.iterrows():
            rows.append({"feature": r.get("feature", r.get("input")), "CI": float(r["CI"]), "CU": float(r["CU"])})
    per_instance = pd.DataFrame(rows)
    return (per_instance.groupby("feature")[["CI", "CU"]].mean()
            .sort_values("CI", ascending=False))


# --------------------------------------------------------------------------
# DEXiRE through a distilled MLP
# --------------------------------------------------------------------------

def distil_and_extract(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list[str],
                       hidden=(32, 16), epochs: int = 80, batch_size: int = 64, rule_sample: float = 0.5):
    """Distil ``model`` into a small MLP and extract DEXiRE rules from it.

    The MLP is trained on the ensemble's labels for the training partition
    (last 15% kept for early stopping). Rules are extracted on a stratified half
    of the training partition, on standardised inputs.
    Returns the rule set, the fitted scaler and a dict of fidelity figures.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from dexire.core.dexire_abstract import Mode, RuleExtractorEnum
    from dexire.dexire import DEXiRE
    from sklearn.model_selection import train_test_split

    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train).astype(np.float32), scaler.transform(X_test).astype(np.float32)
    teacher_tr = model.predict(X_train).astype(np.int64)
    teacher_te = model.predict(X_test).astype(np.int64)

    cut = int(0.85 * len(Xtr))
    pos, neg = teacher_tr[:cut].sum(), (1 - teacher_tr[:cut]).sum()
    weights = {0: (pos + neg) / (2 * neg), 1: (pos + neg) / (2 * pos)}

    mlp = tf.keras.Sequential([tf.keras.layers.Input(shape=(Xtr.shape[1],))]
                              + [layer for h in hidden for layer in (tf.keras.layers.Dense(h, activation="relu"),
                                                                    tf.keras.layers.Dropout(0.2))]
                              + [tf.keras.layers.Dense(1, activation="sigmoid")])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc")])
    mlp.fit(Xtr[:cut], teacher_tr[:cut], validation_data=(Xtr[cut:], teacher_tr[cut:]),
            epochs=epochs, batch_size=batch_size, class_weight=weights, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10,
                                                        restore_best_weights=True)])
    mlp_te = (mlp.predict(Xte, verbose=0).ravel() >= 0.5).astype(np.int64)

    _, Xs, _, ys = train_test_split(Xtr, teacher_tr, test_size=rule_sample, random_state=SEED, stratify=teacher_tr)
    dex = DEXiRE(model=mlp, feature_names=feature_names, class_names=["licit", "fraud"],
                 mode=Mode.CLASSIFICATION, rule_extraction_method=RuleExtractorEnum.TREERULE)
    rules = dex.extract_rules(Xs, ys)

    rule_pred = np.array([int(v) if isinstance(v, (int, float, np.integer, np.floating)) else -1
                          for v in rules.predict_numpy_rules(Xte)])
    covered = rule_pred != -1
    figures = {
        "n_rules": len(rules.get_rules()),
        "mlp_fidelity": float((mlp_te == teacher_te).mean()),
        "coverage": float(covered.mean()),
        "rule_fidelity": float((rule_pred[covered] == teacher_te[covered]).mean()),
    }
    return rules, scaler, figures, rule_pred, covered


def readable_rule(rule_text: str, scaler: StandardScaler, feature_names: list[str]) -> str:
    """Rewrite standardised thresholds of a DEXiRE rule in original units."""
    index = {name: i for i, name in enumerate(feature_names)}

    def back(match: re.Match) -> str:
        name, op, value = match.group(1), match.group(2), float(match.group(3))
        i = index[name]
        return f"{name} {op} {value * scaler.scale_[i] + scaler.mean_[i]:.4g}"

    return re.sub(r"(\w+)\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)", back, rule_text)
