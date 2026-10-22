"""Read-only frozen-feature diagnostic for a single source-only HUST checkpoint.

No target labels are used to fit or select a classifier. Target labels are read
only after the source-trained heads have been fixed, to report Macro-F1.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from collate import multimodal_collate_fn
from models.learnable_prototypes import LearnablePrototypes
from utils.evaluation_utils import _collect_z_metric_embeddings


def _disable_augmentation(dataset, changed):
    if hasattr(dataset, "spec_augment"):
        changed.append((dataset, dataset.spec_augment))
        dataset.spec_augment = False
    if hasattr(dataset, "dataset"):
        _disable_augmentation(dataset.dataset, changed)
    if hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            _disable_augmentation(child, changed)


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _score(y, predicted):
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(
            y, predicted, labels=np.arange(4), average=None, zero_division=0
        )],
    }


def _relation_calibrated_centers(centers, relation, weight):
    """Fit only a 4x4 source-center mixing matrix; no target samples enter."""
    if weight == 0.0:
        return centers.copy()
    c = torch.tensor(centers, dtype=torch.float32)
    r = torch.tensor(relation, dtype=torch.float32)
    offdiag = ~torch.eye(len(c), dtype=torch.bool)
    b = torch.nn.Parameter(torch.zeros(len(c), len(c)))
    optimizer = torch.optim.Adam([b], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        p = F.normalize(c + (b * offdiag) @ c, dim=-1)
        geometry = F.smooth_l1_loss((p @ p.T)[offdiag], r[offdiag])
        center_drift = (1.0 - (p * c).sum(dim=-1)).mean()
        loss = center_drift + weight * geometry + 0.01 * b.square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return F.normalize(c + (b * offdiag) @ c, dim=-1).numpy()


def _condition_ids(dataset):
    if isinstance(dataset, Subset):
        return _condition_ids(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_condition_ids(part) for part in dataset.datasets])
    if hasattr(dataset, "condition_ids"):
        return np.asarray(dataset.condition_ids, dtype=np.int32)
    raise ValueError("Training dataset does not expose source condition ids")


def _fit_proto_calibration(x, y, prototypes, mode, penalty):
    """Fit only per-class bias, optionally with four positive class scales."""
    base = 10.0 * np.asarray(x @ prototypes.T, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    n_class = prototypes.shape[0]
    n_param = n_class if mode == "bias" else 2 * n_class

    def objective(theta):
        if mode == "bias":
            scores = base + theta
            grad_scale = None
        else:
            scales = np.exp(theta[:n_class])
            scores = base * scales + theta[n_class:]
            grad_scale = base * scales
        probs = softmax(scores, axis=1)
        delta = probs
        delta[np.arange(len(labels)), labels] -= 1.0
        delta /= len(labels)
        regularization = penalty * np.sum(theta * theta)
        value = (np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels])
                 + regularization)
        if mode == "bias":
            gradient = delta.sum(axis=0) + 2.0 * penalty * theta
        else:
            gradient = np.concatenate([
                np.sum(delta * grad_scale, axis=0), delta.sum(axis=0)
            ]) + 2.0 * penalty * theta
        return value, gradient

    bounds = [(-2.0, 2.0)] * n_class + [(-20.0, 20.0)] * (n_param - n_class)
    if mode == "bias":
        bounds = [(-20.0, 20.0)] * n_class
    fit = minimize(objective, np.zeros(n_param), method="L-BFGS-B", jac=True,
                   bounds=bounds, options={"maxiter": 200})
    if not np.all(np.isfinite(fit.x)) or not np.isfinite(fit.fun):
        raise RuntimeError(f"Prototype calibration failed: {fit.message}")
    return fit.x


def _predict_proto_calibration(x, prototypes, mode, theta):
    base = 10.0 * np.asarray(x @ prototypes.T, dtype=np.float64)
    if mode == "bias":
        scores = base + theta
    else:
        n_class = prototypes.shape[0]
        scores = base * np.exp(theta[:n_class]) + theta[n_class:]
    return np.argmax(scores, axis=1)


def run_frozen_proto_probe(detector, train_dataset, val_loader, target_loader,
                           checkpoint_path, output_path=""):
    if target_loader is None:
        raise ValueError("Frozen prototype probe requires a target loader")
    device = next(detector.fusion_decoder.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("learnable_proto_module")
    if not state:
        raise ValueError("Checkpoint lacks learnable_proto_module")
    text_anchor = state["text_anchor"]
    module = LearnablePrototypes(text_anchor.to(device)).to(device)
    relation = state.get("relation_target")
    if relation is not None and relation.numel():
        module.set_relation_target(relation.to(device))
    module.load_state_dict(state, strict=True)
    module.eval()
    detector.fusion_decoder._learnable_proto_module = module
    del checkpoint

    changed = []
    _disable_augmentation(train_dataset, changed)
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False, num_workers=0,
            collate_fn=multimodal_collate_fn,
        )
        x_train, y_train, _ = _collect_z_metric_embeddings(detector, train_loader, "HUST", match_direct_eval=True)
        x_val, y_val, _ = _collect_z_metric_embeddings(detector, val_loader, "HUST", match_direct_eval=True)
        x_target, y_target, _ = _collect_z_metric_embeddings(detector, target_loader, "HUST", match_direct_eval=True)
    finally:
        for dataset, old_value in changed:
            dataset.spec_augment = old_value
    if any(x is None for x in (x_train, x_val, x_target)):
        raise RuntimeError("Failed to collect one or more embedding splits")

    classes = np.arange(text_anchor.shape[0])
    if not np.array_equal(np.unique(y_train), classes):
        raise ValueError(f"Source train classes {np.unique(y_train)} != {classes}")
    centers = _unit(np.stack([x_train[y_train == k].mean(axis=0) for k in classes]))
    learned = module.forward().detach().cpu().numpy()
    learned = _unit(learned)

    result = {
        "checkpoint": checkpoint_path,
        "splits": {"source_train": len(y_train), "source_val": len(y_val),
                   "target_test": len(y_target)},
        "source_centroid_cosine": (centers @ centers.T).round(5).tolist(),
        "checkpoint_proto_to_fresh_center_cosine":
            np.sum(learned * centers, axis=1).round(5).tolist(),
        "heads": {},
    }
    for name, prototype in (("checkpoint_proto", learned), ("fresh_source_center", centers)):
        result["heads"][name] = {
            "source_val": _score(y_val, classes[np.argmax(x_val @ prototype.T, axis=1)]),
            "target_test": _score(y_target, classes[np.argmax(x_target @ prototype.T, axis=1)]),
        }

    # Linear readout is fitted on the identical frozen z_metric features.
    # C selection uses only the source validation split.
    candidates = []
    for c in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=1000, random_state=0)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        candidates.append((f1_score(y_val, val_pred, average="macro", zero_division=0), c, model))
    _, best_c, linear = max(candidates, key=lambda t: (t[0], -t[1]))
    result["heads"]["source_fitted_linear"] = {
        "chosen_c": best_c,
        "source_val": _score(y_val, linear.predict(x_val)),
        "target_test": _score(y_target, linear.predict(x_target)),
        "candidate_source_val_macro_f1": {str(c): float(score) for score, c, _ in candidates},
    }

    if os.environ.get("FROZEN_HEAD_CV", "0") == "1":
        groups = _condition_ids(train_dataset)
        if len(groups) != len(y_train):
            raise RuntimeError("Source condition ids do not match frozen features")
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise RuntimeError("Need at least two source conditions for head CV")
        result["head_calibration"] = {
            "scope": "fixed encoder/prototypes; held-out source condition never used to fit its calibration head",
            "source_condition_ids": unique_groups.tolist(),
            "heads": {},
        }
        for mode in ("bias", "scale_bias"):
            trials = []
            for penalty in (0.001, 0.01, 0.1):
                group_scores = []
                for group in unique_groups:
                    held_out = groups == group
                    theta = _fit_proto_calibration(
                        x_train[~held_out], y_train[~held_out], learned, mode, penalty
                    )
                    predicted = _predict_proto_calibration(
                        x_train[held_out], learned, mode, theta
                    )
                    group_scores.append(float(f1_score(
                        y_train[held_out], predicted, average="macro", zero_division=0
                    )))
                trials.append((float(np.mean(group_scores)), penalty, group_scores))
            best_score, chosen_penalty, per_group = max(trials, key=lambda row: (row[0], -row[1]))
            theta = _fit_proto_calibration(x_train, y_train, learned, mode, chosen_penalty)
            result["head_calibration"]["heads"][mode] = {
                "selected_penalty": chosen_penalty,
                "source_condition_cv_macro_f1": best_score,
                "source_condition_cv_per_group": per_group,
                "source_val": _score(y_val, _predict_proto_calibration(x_val, learned, mode, theta)),
                "target_test": _score(y_target, _predict_proto_calibration(x_target, learned, mode, theta)),
                "fitted_parameter_count": len(theta),
            }
            print(f"[FrozenHeadCV] mode={mode} source_condition_cv={best_score:.4f} "
                  f"target={result['head_calibration']['heads'][mode]['target_test']['macro_f1']:.4f}",
                  flush=True)

    if os.environ.get("FROZEN_FORWARD_AUDIT", "0") == "1":
        result["forward_audit"] = {}
        changed_again = []
        _disable_augmentation(train_dataset, changed_again)
        try:
            for mode in ("conditioned", "conditioned_ln"):
                xa, ya, _ = _collect_z_metric_embeddings(
                    detector, train_loader, "HUST", diagnostic_forward_mode=mode
                )
                xv, yv, _ = _collect_z_metric_embeddings(
                    detector, val_loader, "HUST", diagnostic_forward_mode=mode
                )
                xt, yt, _ = _collect_z_metric_embeddings(
                    detector, target_loader, "HUST", diagnostic_forward_mode=mode
                )
                if not (np.array_equal(ya, y_train) and np.array_equal(yv, y_val)
                        and np.array_equal(yt, y_target)):
                    raise RuntimeError(f"Split labels changed in forward mode {mode}")
                ca = _unit(np.stack([xa[ya == k].mean(axis=0) for k in classes]))
                audit_heads = {}
                for head_name, p in (("checkpoint_proto", learned), ("fresh_source_center", ca)):
                    audit_heads[head_name] = {
                        "source_val": _score(yv, classes[np.argmax(xv @ p.T, axis=1)]),
                        "target_test": _score(yt, classes[np.argmax(xt @ p.T, axis=1)]),
                    }
                candidates_a = []
                for c in (0.1, 1.0, 10.0):
                    head = LogisticRegression(C=c, max_iter=1000, random_state=0)
                    head.fit(xa, ya)
                    score_a = f1_score(yv, head.predict(xv), average="macro", zero_division=0)
                    candidates_a.append((score_a, c, head))
                _, chosen_c, chosen_head = max(candidates_a, key=lambda row: (row[0], -row[1]))
                audit_heads["source_fitted_linear"] = {
                    "chosen_c": chosen_c,
                    "source_val": _score(yv, chosen_head.predict(xv)),
                    "target_test": _score(yt, chosen_head.predict(xt)),
                }
                result["forward_audit"][mode] = {
                    "heads": audit_heads,
                    "checkpoint_proto_to_fresh_center_cosine":
                        np.sum(learned * ca, axis=1).round(5).tolist(),
                }
                print(f"[FrozenForwardAudit] mode={mode} "
                      f"proto_val={audit_heads['checkpoint_proto']['source_val']['macro_f1']:.4f} "
                      f"proto_target={audit_heads['checkpoint_proto']['target_test']['macro_f1']:.4f}",
                      flush=True)
        finally:
            for dataset, old_value in changed_again:
                dataset.spec_augment = old_value

    # Paired semantic control on the *same* frozen embeddings and source centers.
    # Choose the strength only on source validation; do not tune with target data.
    if relation is not None and relation.numel():
        correct_r = relation.detach().cpu().numpy()
        permutation = np.roll(classes, -1)
        relation_variants = {
            "correct": correct_r,
            "shuffled": correct_r[np.ix_(permutation, permutation)],
        }
        strengths = (0.0, 0.05, 0.2, 1.0)
        result["semantic_relation"] = {
            "correct_relation_cosine": correct_r.round(5).tolist(),
            "shuffled_permutation": permutation.tolist(),
            "variants": {},
        }
        for variant, r in relation_variants.items():
            trial = []
            for weight in strengths:
                p = _relation_calibrated_centers(centers, r, weight)
                val = _score(y_val, classes[np.argmax(x_val @ p.T, axis=1)])
                target = _score(y_target, classes[np.argmax(x_target @ p.T, axis=1)])
                trial.append({"weight": weight, "source_val": val, "target_test": target})
            best = max(trial, key=lambda row: (row["source_val"]["macro_f1"], -row["weight"]))
            result["semantic_relation"]["variants"][variant] = {
                "source_selected_weight": best["weight"],
                "source_selected_target_test": best["target_test"],
                "candidates": trial,
            }
        correct_weight = result["semantic_relation"]["variants"]["correct"]["source_selected_weight"]
        shuffled_trials = result["semantic_relation"]["variants"]["shuffled"]["candidates"]
        result["semantic_relation"]["shuffled_at_correct_weight"] = next(
            row for row in shuffled_trials if row["weight"] == correct_weight
        )

    print("[FrozenProtoProbe] " + json.dumps(result, ensure_ascii=False), flush=True)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    return result
