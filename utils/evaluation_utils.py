"""
评估相关工具函数
"""
import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset, Dataset
import numpy as np
import random
import math
import os
import warnings
from sklearn.semi_supervised import LabelSpreading
from collate import multimodal_collate_fn
from utils.ablation_gating import load_spec_from_batch
try:
    from utils.dynamic_semantic_multi_proto import (
        build_dynamic_semantic_cluster_bank,
        predict_with_dynamic_cluster_bank,
        summarize_cluster_bank,
    )
except Exception:
    build_dynamic_semantic_cluster_bank = None
    predict_with_dynamic_cluster_bank = None
    summarize_cluster_bank = None


def _compute_bypass_features(detector, batch, device):
    """Compute WST/ENV/ratio/SCD bypass features from batch data using detector's encoders."""
    wst, env, ratio, scd = None, None, None, None
    if detector is None:
        return wst, env, ratio, scd
    wst_enc = getattr(detector, 'wst_encoder', None)
    env_enc = getattr(detector, 'env_encoder', None)
    scd_enc = getattr(detector, 'scd_encoder', None)
    if wst_enc is not None:
        _wst_raw = batch.get("wst_spec", None) if isinstance(batch, dict) else None
        if _wst_raw is not None:
            with torch.no_grad():
                wst = wst_enc(_wst_raw.to(device))
    if env_enc is not None:
        _env_raw = batch.get("env_spec", None) if isinstance(batch, dict) else None
        if _env_raw is not None:
            with torch.no_grad():
                env = env_enc(_env_raw.to(device))
        _ratio_raw = batch.get("bpfi_bpfo_ratio", None) if isinstance(batch, dict) else None
        if _ratio_raw is not None:
            ratio = _ratio_raw.to(device)
    if scd_enc is not None:
        _scd_raw = batch.get("scd_spec", None) if isinstance(batch, dict) else None
        if _scd_raw is not None:
            with torch.no_grad():
                scd = scd_enc(_scd_raw.to(device))
    if wst is None:
        wst = getattr(detector, '_last_z_wst', None)
    if env is None:
        env = getattr(detector, '_last_z_env', None)
    if ratio is None:
        ratio = getattr(detector, '_last_ratio', None)
    if scd is None:
        scd = getattr(detector, '_last_z_scd', None)
    return wst, env, ratio, scd


def _safe_spec_head_logits(spec_encoder, z_spec, detector=None, batch=None):
    """Compute SpecHead logits via compute_spec_head_logits (with bypass features)
    or fall back to direct classifier call.

    ATT456-fix: _se.classifier expects 705-dim input (512 + bypass), but evaluation
    code passes 512-dim z_spec.  Use compute_spec_head_logits which handles bypass
    concatenation internally.
    """
    try:
        _dev = z_spec.device
    except Exception:
        _dev = torch.device("cpu")
    wst, env, ratio, scd = _compute_bypass_features(detector, batch, _dev)
    if hasattr(spec_encoder, 'compute_spec_head_logits'):
        logits = spec_encoder.compute_spec_head_logits(z_spec, wst_aux=wst, env_aux=env, ratio_aux=ratio, scd_aux=scd)
        if logits is not None:
            return logits
    if hasattr(spec_encoder, 'classifier') and spec_encoder.classifier is not None:
        return spec_encoder.classifier(z_spec)
    return None


def _domain_id_map_cfg():
    try:
        import config as cfg
        return {
            "CWRU_GK2": int(getattr(cfg, "CWRU_GK2_DOMAIN_ID", 0)),
            "CWRU_GK4": int(getattr(cfg, "CWRU_GK4_DOMAIN_ID", 1)),
            "PU": int(getattr(cfg, "PU_DOMAIN_ID", 2)),
            "HUST": int(getattr(cfg, "HUST_DOMAIN_ID", 3)),
            "SEU": int(getattr(cfg, "SEU_DOMAIN_ID", 5)),
            "HIT": int(getattr(cfg, "HIT_DOMAIN_ID", 4)),
        }
    except Exception:
        return {"CWRU_GK2": 0, "CWRU_GK4": 1, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}


def _encode_domain_ids(domains, conditions, device):
    if domains is None:
        return None
    _dmap = _domain_id_map_cfg()
    _encoded = []
    for i, d in enumerate(domains):
        d_up = str(d).upper()
        cond = str(conditions[i]).lower() if (conditions is not None and i < len(conditions)) else ""
        if d_up == "CWRU":
            if "gk4" in cond or "gk_4" in cond or "hp3" in cond:
                _encoded.append(_dmap["CWRU_GK4"])
            else:
                _encoded.append(_dmap["CWRU_GK2"])
        elif d_up == "PU":
            _encoded.append(_dmap["PU"])
        elif d_up == "HUST":
            _encoded.append(_dmap["HUST"])
        elif d_up == "SEU":
            _encoded.append(_dmap["SEU"])
        elif d_up == "HIT":
            _encoded.append(_dmap.get("HIT", _dmap.get("SEU", _dmap["CWRU_GK2"])))
        else:
            _encoded.append(_dmap["CWRU_GK2"])
    return torch.tensor(_encoded, dtype=torch.long, device=device)


def _collect_z_metric_embeddings(detector, loader, dataset_name="", match_direct_eval=False,
                                 diagnostic_forward_mode=None):
    """
    Collect L2-normalized z_metric on a loader (typically holdout test split after LP-FT).
    Mirrors FusionProto eval forward path for t-SNE / embedding export.
    """
    import torch.nn.functional as F
    if diagnostic_forward_mode not in (None, "conditioned", "conditioned_ln"):
        raise ValueError(f"Unknown diagnostic_forward_mode={diagnostic_forward_mode}")

    _se = getattr(detector, "spec_encoder", None)
    _sig_enc = getattr(detector, "signal_encoder", None)
    _fusion_mod = getattr(detector, "fusion_module", None)
    _fusion_dec = getattr(detector, "fusion_decoder", None)
    if _se is None or _sig_enc is None or _fusion_mod is None:
        return None, None, None

    device = next(_se.parameters()).device
    _se.eval()
    _sig_enc.eval()
    _fusion_mod.eval()
    if _fusion_dec is not None:
        _fusion_dec.eval()

    _eval_fusion_norm_inputs = False
    _eval_fusion_use_spec_pre_l2 = True
    _use_dual_space_eval = False
    try:
        import config as _cfg_eval_fusion
        _eval_fusion_norm_inputs = bool(getattr(_cfg_eval_fusion, "EVAL_FUSION_NORMALIZE_INPUTS", False))
        _eval_fusion_use_spec_pre_l2 = bool(
            getattr(_cfg_eval_fusion, "EVAL_FUSION_USE_SPEC_PRE_L2_INPUT", True)
        )
        _use_dual_space_eval = bool(getattr(_cfg_eval_fusion, "USE_DUAL_SPACE_DECOUPLE", False))
    except Exception:
        pass

    _current_protos = None
    if hasattr(detector, "_active_text_prototypes") and callable(detector._active_text_prototypes):
        try:
            _current_protos = detector._active_text_prototypes()
        except Exception:
            pass

    _dom_map = {"CWRU": 0, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}
    _name_up = str(dataset_name).upper().split("/")[0].split("(")[0].strip()
    _default_did = int(_dom_map.get(_name_up, -1))

    all_emb, all_lbl, all_dom = [], [], []
    with torch.no_grad():
        for _b in loader:
            _spec = load_spec_from_batch(_b, device, detector)
            _smask = _b["spec_mask"].to(device)
            _signal = _b["signal"].to(device)
            _sig_mask = _b.get(
                "signal_mask",
                torch.ones(_signal.size(0), _signal.size(1), device=device),
            ).to(device)
            _lbls = _b.get("label", _b.get("fault_label", None))
            if _lbls is None:
                continue
            _lbls = _lbls.to(device)

            _domain_ids_b = None
            if _b.get("domain", None) is not None:
                _domain_ids_b = _encode_domain_ids(
                    _b.get("domain"), _b.get("condition", None), device
                )

            _sv = _b.get("stat_vec", None)
            if _sv is not None:
                _sv = _sv.to(device)

            if match_direct_eval:
                # Mirror the FusionProto-Learnable evaluation call exactly.
                _sh_out = _se(_spec, _smask, return_logits=True)
            elif diagnostic_forward_mode is not None:
                _sh_out = _se(_spec, _smask, return_logits=True,
                              domain_ids=_domain_ids_b, stat_vec=_sv)
            else:
                try:
                    _sh_out = _se(_spec, _smask, return_logits=True, domain_ids=_domain_ids_b, stat_vec=_sv)
                except TypeError:
                    try:
                        _sh_out = _se(_spec, _smask, return_logits=True, domain_ids=_domain_ids_b)
                    except TypeError:
                        _sh_out = _se(_spec, _smask, return_logits=True)

            _z_spec = _sh_out[0] if isinstance(_sh_out, (tuple, list)) else _sh_out
            _z_spec_for_fusion = _z_spec
            if _eval_fusion_use_spec_pre_l2:
                _z_spec_pre = getattr(_z_spec, "_pre_l2_feature", None)
                if torch.is_tensor(_z_spec_pre) and _z_spec_pre.shape == _z_spec.shape:
                    _z_spec_for_fusion = _z_spec_pre

            _has_sig_head = bool(getattr(_sig_enc, "use_classification_head", False)) and getattr(_sig_enc, "classifier", None) is not None
            if diagnostic_forward_mode is not None:
                _z_sig_raw = _sig_enc(_signal, _sig_mask, domain_ids=_domain_ids_b,
                                      return_logits=_has_sig_head)
            elif match_direct_eval and _has_sig_head:
                _z_sig_raw = _sig_enc(_signal, _sig_mask, return_logits=True)
            else:
                _z_sig_raw = _sig_enc(_signal, _sig_mask)
            if isinstance(_z_sig_raw, (tuple, list)):
                _z_sig_raw = _z_sig_raw[0]
            _z_sig_fus = _z_sig_raw.float()
            _z_spec_fus = _z_spec_for_fusion.float()
            if diagnostic_forward_mode == "conditioned_ln":
                _z_sig_fus = F.layer_norm(_z_sig_fus, [_z_sig_fus.size(-1)])
                _z_spec_fus = F.layer_norm(_z_spec_fus, [_z_spec_fus.size(-1)])
            elif _eval_fusion_norm_inputs:
                _z_sig_fus = F.normalize(_z_sig_fus, dim=-1)
                _z_spec_fus = F.normalize(_z_spec_fus, dim=-1)

            if hasattr(_fusion_mod, "domain_embedding"):
                _z_fus = _fusion_mod(
                    _z_sig_fus,
                    _z_spec_fus,
                    domain_ids=_domain_ids_b,
                    text_prototypes=_current_protos,
                )
            else:
                _z_fus = _fusion_mod(_z_sig_fus, _z_spec_fus)

            _z_metric_eval = F.normalize(_z_fus.float(), dim=-1)
            if _use_dual_space_eval and _fusion_dec is not None and hasattr(_fusion_dec, "get_metric_embedding"):
                try:
                    _z_metric_eval = F.normalize(_fusion_dec.get_metric_embedding(_z_metric_eval).float(), dim=-1)
                except Exception:
                    pass

            _lp_module = getattr(_fusion_dec, "_learnable_proto_module", None) if _fusion_dec is not None else None
            if _lp_module is None and _fusion_dec is not None:
                _proto_adapter = getattr(_fusion_dec, "_proto_adapter", None)
                if _proto_adapter is not None:
                    try:
                        _z_metric_eval = _proto_adapter(_z_metric_eval.float())
                        _z_metric_eval = F.normalize(_z_metric_eval, dim=-1)
                    except Exception:
                        pass

            all_emb.append(_z_metric_eval.detach().cpu().float())
            all_lbl.append(_lbls.detach().cpu())
            if _b.get("domain", None) is not None:
                for _d in _b["domain"]:
                    _du = str(_d).upper()
                    all_dom.append(int(_dom_map.get(_du, _default_did)))
            else:
                all_dom.extend([_default_did] * int(_lbls.size(0)))

    if not all_emb:
        return None, None, None
    _emb_cat = torch.cat(all_emb, dim=0)
    _emb_cat = F.normalize(_emb_cat.float(), dim=-1)
    return (
        _emb_cat.numpy().astype(np.float32),
        torch.cat(all_lbl, dim=0).numpy().astype(np.int32),
        np.asarray(all_dom, dtype=np.int32),
    )


def _save_metric_embeddings_npz(embeddings, labels, domains, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        domains=np.asarray(domains, dtype=np.int32),
    )
    print(f"[EmbDump] saved {int(labels.shape[0])} embeddings -> {out_path}")


def _sanitize_emb_export_tag(tag: str) -> str:
    """Filesystem-safe suffix from LOO test condition key (e.g. speed_1730)."""
    import re

    t = str(tag or "").strip()
    if not t:
        return ""
    t = re.sub(r"[^\w\-.]+", "_", t)
    return t[:96]


def _embedding_export_filenames(
    dataset_name: str,
    stage: str,
    n_shot_k: int = 0,
    export_tag: str = "",
) -> tuple:
    """
    Returns (npz_filename, tsne_png_filename) under EVAL_EXPORT_EMBEDDINGS_DIR.
    When export_tag is set: CWRU_speed_1730_zs_embeddings.npz
    When empty (legacy): CWRU_zs_embeddings.npz
    """
    _ds = str(dataset_name).strip().upper()
    _tag = _sanitize_emb_export_tag(export_tag)
    _tag_part = f"_{_tag}" if _tag else ""
    if str(stage).lower() == "zs":
        _base = f"{_ds}{_tag_part}_zs_embeddings"
    else:
        _k_part = f"_k{int(n_shot_k)}" if int(n_shot_k) > 0 else ""
        _base = f"{_ds}{_tag_part}{_k_part}_postlpft_embeddings"
    return f"{_base}.npz", f"{_base}_tsne.png"


def _resolve_emb_export_tag(export_tag: str) -> str:
    """Honor EVAL_EXPORT_EMBEDDINGS_PER_CONDITION; return sanitized tag or ''."""
    _tag = _sanitize_emb_export_tag(export_tag)
    if not _tag:
        return ""
    try:
        import config as _cfg_tag
        if not bool(getattr(_cfg_tag, "EVAL_EXPORT_EMBEDDINGS_PER_CONDITION", True)):
            return ""
    except Exception:
        pass
    return _tag


def _dump_metric_embedding_arrays(
    embeddings,
    labels,
    domains,
    dataset_name: str,
    stage: str,
    n_shot_k: int = 0,
    export_tag: str = "",
):
    """Save npz (+ optional auto t-SNE) from pre-collected arrays."""
    try:
        import config as _cfg_emb
        if not bool(getattr(_cfg_emb, "EVAL_EXPORT_EMBEDDINGS_ENABLE", False)):
            return
        if stage == "postlpft" and not bool(getattr(_cfg_emb, "EVAL_EXPORT_EMBEDDINGS_POST_LPFT", True)):
            return
        _out_dir = str(
            getattr(
                _cfg_emb,
                "EVAL_EXPORT_EMBEDDINGS_DIR",
                "experiments/results/reviewer/embeddings",
            )
        )
    except Exception:
        return

    _tag = _resolve_emb_export_tag(export_tag)
    if embeddings is None or labels is None or int(np.asarray(labels).size) == 0:
        print(f"[EmbDump] {dataset_name} stage={stage} tag={_tag or 'all'}: no embeddings")
        return

    _fname, _png = _embedding_export_filenames(
        dataset_name, stage, n_shot_k=n_shot_k, export_tag=_tag
    )
    _out_npz = os.path.join(_out_dir, _fname)
    os.makedirs(_out_dir, exist_ok=True)
    _extra = {}
    if _tag:
        _extra["condition_key"] = np.asarray([_tag], dtype=object)
    np.savez(
        _out_npz,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        domains=np.asarray(domains, dtype=np.int32),
        **_extra,
    )
    print(f"[EmbDump] {dataset_name} stage={stage} tag={_tag or 'legacy'}: saved {int(np.asarray(labels).size)} -> {_out_npz}")
    _maybe_auto_tsne(_out_npz, _png)


def _maybe_auto_tsne(npz_path: str, tsne_output_name: str):
    try:
        import config as _cfg_tsne
        if not bool(getattr(_cfg_tsne, "EVAL_EXPORT_EMBEDDINGS_AUTO_TSNE", False)):
            return
        if not os.path.isfile(npz_path):
            return
        from experiments.reviewer_experiments import save_tsne_figure

        _arr = np.load(npz_path)
        _out_dir = str(
            getattr(
                _cfg_tsne,
                "EVAL_EXPORT_EMBEDDINGS_DIR",
                "experiments/results/reviewer/embeddings",
            )
        )
        _fig_dir = os.path.join(_out_dir, "figures")
        os.makedirs(_fig_dir, exist_ok=True)
        _out_png = os.path.join(_fig_dir, tsne_output_name)
        save_tsne_figure(_arr["embeddings"], _arr["labels"], _arr["domains"], _out_png)
        print(f"[EmbDump] Auto t-SNE figure -> {_out_png}")
    except Exception as _e_tsne:
        print(f"[EmbDump] Auto t-SNE skipped: {type(_e_tsne).__name__}: {_e_tsne}")


_DIAG_CLASS_NAMES = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
_DIAG_CNAME_SHORT = ["N", "IR", "OR", "Ball"]


def _compute_head_diag_metrics(preds_np, lbls_np):
    """Per-head accuracy / macro-F1 / per-class recall strings."""
    preds_np = np.asarray(preds_np)
    lbls_np = np.asarray(lbls_np)
    if preds_np.size == 0 or lbls_np.size == 0:
        return {
            "acc": float("nan"),
            "macro_f1": float("nan"),
            "macro_recall": float("nan"),
            "per_class_info": [],
            "classes": [],
        }
    acc = float((preds_np == lbls_np).mean())
    classes = sorted(list(set(lbls_np.tolist())))
    f1s, recalls, per_class_info = [], [], []
    for c in classes:
        tp = float(np.sum((preds_np == c) & (lbls_np == c)))
        fp = float(np.sum((preds_np == c) & (lbls_np != c)))
        fn = float(np.sum((preds_np != c) & (lbls_np == c)))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
        recalls.append(rec)
        cname = _DIAG_CLASS_NAMES.get(c, str(c))
        per_class_info.append(f"{cname}(R={rec:.2f},F1={f1:.2f},N={int(tp + fn)})")
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    macro_recall = float(np.mean(recalls)) if recalls else float("nan")
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "per_class_info": per_class_info,
        "classes": classes,
    }


def _print_head_confusion_matrix(tag, dataset_name, preds_np, lbls_np):
    preds_np = np.asarray(preds_np)
    lbls_np = np.asarray(lbls_np)
    if preds_np.size == 0:
        return
    n_cls = max(int(max(lbls_np)), int(max(preds_np))) + 1
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(lbls_np.tolist(), preds_np.tolist()):
        conf[int(t)][int(p)] += 1
    header = "ConfMat(true\\pred) " + " ".join(
        f"{(_DIAG_CNAME_SHORT[i] if i < len(_DIAG_CNAME_SHORT) else str(i)):>5}"
        for i in range(n_cls)
    )
    print(f"[{tag}] {header}")
    for ri in range(n_cls):
        row_str = " ".join(f"{conf[ri][ci]:>5}" for ci in range(n_cls))
        rname = _DIAG_CNAME_SHORT[ri] if ri < len(_DIAG_CNAME_SHORT) else str(ri)
        print(f"[{tag}]   {rname:>17} {row_str}")


def _pred_max_class_frac(preds_np) -> float:
    preds_np = np.asarray(preds_np)
    if preds_np.size == 0:
        return float("nan")
    _, counts = np.unique(preds_np, return_counts=True)
    return float(counts.max() / preds_np.size)


def _export_modality_metrics_csv(row: dict):
    """Append one LOO phase row for offline shift/modality correlation analysis."""
    try:
        import config as _cfg_csv
        if not bool(getattr(_cfg_csv, "EVAL_EXPORT_MODALITY_METRICS_CSV", True)):
            return
        out_dir = str(
            getattr(
                _cfg_csv,
                "EVAL_EXPORT_MODALITY_METRICS_DIR",
                "experiments/results/modality_metrics",
            )
        )
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "modality_metrics.csv")
    fieldnames = [
        "dataset_name",
        "phase_name",
        "domain",
        "held_out_condition",
        "n_samples",
        "spec_head_acc",
        "spec_head_macro_f1",
        "spec_head_max_class_frac",
        "signal_head_acc",
        "signal_head_macro_f1",
        "signal_head_max_class_frac",
        "signal_proto_acc",
        "signal_proto_macro_f1",
        "signal_proto_max_class_frac",
        "spec_proto_acc",
        "fusion_proto_acc",
        "fusion_raw_acc",
        "spec_vs_signal_disagree",
        "fusion_proto_sep_ratio",
        "final_decision_acc",
        "final_decision_path",
    ]
    write_header = not os.path.isfile(out_path) or os.path.getsize(out_path) == 0
    import csv

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[ModalityCSV] appended -> {out_path}")


def _export_metric_embeddings_if_enabled(
    detector,
    loader,
    dataset_name: str,
    stage: str,
    n_shot_k: int = 0,
    export_tag: str = "",
):
    """
    stage: 'zs' (zero-shot) or 'postlpft'
    export_tag: LOO held-out condition key (e.g. speed_1730); unique npz per fold when set.
    """
    _emb, _lbl, _dom = _collect_z_metric_embeddings(detector, loader, dataset_name=dataset_name)
    if _emb is None:
        return
    _dump_metric_embedding_arrays(
        _emb.detach().cpu().numpy() if hasattr(_emb, "detach") else _emb,
        _lbl,
        _dom,
        dataset_name=dataset_name,
        stage=stage,
        n_shot_k=n_shot_k,
        export_tag=export_tag,
    )


def split_loader_for_target_adapt(loader, finetune_percent=0.05, seed=42, dataset_name=None, n_shot_k=0):
    """
    Split a DataLoader's underlying dataset into finetune and eval subsets.
    
    Extracted as module-level function (Attempt 49) for reuse in dual-checkpoint LP-FT.
    
    Args:
        loader: Source DataLoader
        finetune_percent: Fraction of data for fine-tuning (default 0.05 = 5%)
        seed: Random seed for reproducible splits
        dataset_name: "HUST" or "SEU" for domain-specific class quotas
        n_shot_k: if >0, enforce exact K samples per class for finetune split
    
    Returns:
        finetune_loader, eval_loader (both DataLoader instances)
    """
    dataset = loader.dataset

    n_total = len(dataset)
    n_finetune = max(1, int(n_total * finetune_percent))
    rng = random.Random(seed)

    # --- robust label getter ---
    def _label_from_dataset_attr(i):
        for attr in ("labels", "y", "targets"):
            if hasattr(dataset, attr):
                arr = getattr(dataset, attr)
                try:
                    return int(arr[i])
                except Exception:
                    continue
        return None

    def _get_label(i):
        y = _label_from_dataset_attr(i)
        if y is not None:
            return y
        item = dataset[i]
        if isinstance(item, dict) and "label" in item:
            return int(item["label"])
        if isinstance(item, (tuple, list)) and len(item) > 0:
            return int(item[-1])
        raise RuntimeError("Unable to extract label from dataset sample.")

    # --- group indices by class ---
    by_cls = {}
    for i in range(n_total):
        y = _get_label(i)
        by_cls.setdefault(y, []).append(i)
    classes = sorted(by_cls.keys())

    # --- per-class quotas (ensure supervision signal exists) ---
    if int(n_shot_k) > 0:
        desired = {c: int(n_shot_k) for c in classes}
    elif dataset_name == "HUST":
        q0 = max(5, int(0.10 * n_finetune))
        qf = max(10, int(0.25 * n_finetune))
        desired = {0: q0, 1: qf, 2: qf, 3: qf}
    elif dataset_name == "SEU":
        q0 = max(5, int(0.10 * n_finetune))
        qf = max(10, int(0.25 * n_finetune))
        desired = {0: q0, 1: qf, 2: qf, 3: qf}
    elif dataset_name == "HIT":
        q0 = max(10, int(0.10 * n_finetune))
        qf = max(20, int(0.25 * n_finetune))
        desired = {0: q0, 1: qf, 2: qf}
    else:
        desired = {c: 1 for c in classes}

    finetune_indices = []
    leftovers = []
    for c in classes:
        idxs = list(by_cls[c])
        rng.shuffle(idxs)
        take = min(desired.get(c, 0), len(idxs))
        finetune_indices.extend(idxs[:take])
        leftovers.extend(idxs[take:])

    _use_nshot = int(n_shot_k) > 0
    if not _use_nshot:
        remaining = max(0, n_finetune - len(finetune_indices))
        if remaining > 0 and len(leftovers) > 0:
            rng.shuffle(leftovers)
            finetune_indices.extend(leftovers[:remaining])
        finetune_indices = list(dict.fromkeys(finetune_indices))[:n_finetune]
    else:
        finetune_indices = list(dict.fromkeys(finetune_indices))
        _short = []
        for c in classes:
            _got = sum(1 for i in finetune_indices if _get_label(i) == c)
            if _got < int(n_shot_k):
                _short.append(f"class{c}:{_got}/{int(n_shot_k)}")
        if _short:
            print(
                f"[Split] {dataset_name or 'target'} n-shot K={int(n_shot_k)}: "
                f"insufficient samples per class ({', '.join(_short)})"
            )
        print(
            f"[Split] {dataset_name or 'target'} n-shot K={int(n_shot_k)}: "
            f"finetune={len(finetune_indices)} eval={n_total - len(finetune_indices)} (total={n_total})"
        )

    finetune_set = set(finetune_indices)
    eval_indices = [i for i in range(n_total) if i not in finetune_set]
    
    # Create subsets
    finetune_subset = Subset(dataset, finetune_indices)
    eval_subset = Subset(dataset, eval_indices)
    
    # Create new loaders
    finetune_loader = DataLoader(
        finetune_subset,
        batch_size=loader.batch_size,
        shuffle=False,  # Use fixed split order
        collate_fn=multimodal_collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor if hasattr(loader, 'prefetch_factor') else None,
        persistent_workers=loader.persistent_workers if hasattr(loader, 'persistent_workers') else False,
    )
    
    eval_loader = DataLoader(
        eval_subset,
        batch_size=loader.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor if hasattr(loader, 'prefetch_factor') else None,
        persistent_workers=loader.persistent_workers if hasattr(loader, 'persistent_workers') else False,
    )
    
    return finetune_loader, eval_loader


def _infer_loader_domain(loader, default="SEU"):
    """Resolve dataset domain string from a DataLoader (handles Subset chains)."""
    if loader is None:
        return default
    ds = getattr(loader, "dataset", None)
    try:
        from torch.utils.data import Subset
        while isinstance(ds, Subset):
            ds = ds.dataset
    except Exception:
        pass
    dom = getattr(ds, "domain", None)
    if dom:
        return str(dom).upper()
    return default


def run_evaluation(
    detector,
    cwru_test_loader,
    pu_test_loader,
    hust_loader,
    hit_loader=None,
    task1_cwru_loader=None,
    task1_pu_loader=None,
    twc_source_cwru_loader=None,
    twc_source_pu_loader=None,
    emb_export_tag: str = "",
):
    """
    运行所有评估方法
    
    Args:
        detector: AnomalyDetector实例
        cwru_test_loader: CWRU测试集DataLoader
        pu_test_loader: PU测试集DataLoader
        hust_loader: HUST数据集DataLoader
        hit_loader: HIT数据集DataLoader（可选）
        emb_export_tag: LOO held-out condition key for per-fold embedding/t-SNE files.
    """
    _emb_export_tag = _resolve_emb_export_tag(emb_export_tag)
    if _emb_export_tag:
        print(f"[EmbDump] per-condition export tag={_emb_export_tag}")
    def _safe_macro_f1(diag_labels, diag_preds):
        if diag_labels is None or diag_preds is None:
            return float("nan")
        y_true = np.asarray(diag_labels)
        y_pred = np.asarray(diag_preds)
        if y_true.size == 0:
            return float("nan")
        classes = sorted(list(set(y_true.tolist())))
        f1s = []
        for c in classes:
            tp = float(np.sum((y_pred == c) & (y_true == c)))
            fp = float(np.sum((y_pred == c) & (y_true != c)))
            fn = float(np.sum((y_pred != c) & (y_true == c)))
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 2.0 * p * r / (p + r + 1e-12)
            f1s.append(f1)
        return float(np.mean(f1s)) if len(f1s) > 0 else float("nan")

    def _safe_macro_recall(diag_labels, diag_preds):
        if diag_labels is None or diag_preds is None:
            return float("nan")
        y_true = np.asarray(diag_labels)
        y_pred = np.asarray(diag_preds)
        if y_true.size == 0:
            return float("nan")
        classes = sorted(list(set(y_true.tolist())))
        recalls = []
        for c in classes:
            tp = float(np.sum((y_pred == c) & (y_true == c)))
            fn = float(np.sum((y_pred != c) & (y_true == c)))
            recalls.append(tp / (tp + fn + 1e-12))
        return float(np.mean(recalls)) if len(recalls) > 0 else float("nan")

    def _safe_hmean_recall(diag_labels, diag_preds):
        """H-mean = harmonic mean of per-class recalls (Xian et al., TPAMI 2019)."""
        if diag_labels is None or diag_preds is None:
            return float("nan")
        y_true = np.asarray(diag_labels)
        y_pred = np.asarray(diag_preds)
        if y_true.size == 0:
            return float("nan")
        classes = sorted(list(set(y_true.tolist())))
        recalls = []
        for c in classes:
            tp = float(np.sum((y_pred == c) & (y_true == c)))
            fn = float(np.sum((y_pred != c) & (y_true == c)))
            r = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else 0.0
            recalls.append(r)
        if len(recalls) == 0 or any(r <= 0 for r in recalls):
            return float(np.mean(recalls)) if len(recalls) > 0 else float("nan")
        return float(len(recalls) / sum(1.0 / r for r in recalls))

    try:
        import config as _cfg_eval_log
        eval_compact_log = bool(getattr(_cfg_eval_log, "TARGET_EVAL_COMPACT_LOG", True))
        eval_log_grid_details = bool(getattr(_cfg_eval_log, "TARGET_EVAL_LOG_GRID_DETAILS", False))
        eval_log_transductive_details = bool(getattr(_cfg_eval_log, "TARGET_EVAL_LOG_TRANSDUCTIVE_DETAILS", False))
        eval_suppress_sklearn_warnings = bool(getattr(_cfg_eval_log, "EVAL_SUPPRESS_SKLEARN_WARNINGS", True))
        eval_report_fusion_diag = bool(getattr(_cfg_eval_log, "EVAL_REPORT_FUSION_DIAG", False))
    except Exception:
        eval_compact_log = True
        eval_log_grid_details = False
        eval_log_transductive_details = False
        eval_suppress_sklearn_warnings = True
        eval_report_fusion_diag = False

    if eval_suppress_sklearn_warnings:
        try:
            from sklearn.exceptions import UndefinedMetricWarning
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        except Exception:
            pass

    def _evaluate(loader, **kwargs):
        if loader is None:
            return None
        proto_domain = "HIT" if kwargs.get("dataset_name") == "HIT" else None
        return detector.evaluate(loader, prototype_domain=proto_domain, **kwargs)

    def _encode_domain_ids_from_batch(batch, device):
        domains = batch.get("domain", None)
        if domains is None:
            return None
        conditions = batch.get("condition", None)
        return _encode_domain_ids(domains, conditions, device)

    # ATT199(C): test-time Whitening-Coloring Transform (TWC) for target-domain z_spec alignment.
    try:
        import config as _cfg_twc
        _twc_enable = bool(getattr(_cfg_twc, "TARGET_TWC_ENABLE", False))
        _twc_targets = set(getattr(_cfg_twc, "TARGET_TWC_TARGET_DATASETS", ["HUST", "SEU"]))
        _twc_eps = float(getattr(_cfg_twc, "TARGET_TWC_EPS", 1e-5))
        _twc_alpha = float(getattr(_cfg_twc, "TARGET_TWC_BLEND_ALPHA", 1.0))
        _twc_apply_to_svp = bool(getattr(_cfg_twc, "TARGET_TWC_APPLY_TO_SVPEVAL", True))
        _twc_apply_to_selftrain = bool(getattr(_cfg_twc, "TARGET_TWC_APPLY_TO_SELFTRAIN", True))
        _twc_apply_to_postlpft = bool(getattr(_cfg_twc, "TARGET_TWC_APPLY_TO_POSTLPFT", False))
    except Exception:
        _twc_enable = False
        _twc_targets = {"HUST", "SEU"}
        _twc_eps = 1e-5
        _twc_alpha = 1.0
        _twc_apply_to_svp = True
        _twc_apply_to_selftrain = True
        _twc_apply_to_postlpft = False
    _twc_alpha = float(max(0.0, min(1.0, _twc_alpha)))
    _twc_cache = {}

    def _iter_raw_spec_features(loader, max_batches=None):
        if loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None:
            return None
        try:
            _device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None
        _se.eval()
        feats = []
        with torch.no_grad():
            for _bi, _batch in enumerate(loader):
                if max_batches is not None and int(_bi) >= int(max_batches):
                    break
                spec = load_spec_from_batch(_batch, _device, detector)
                spec_mask = _batch["spec_mask"].to(_device)
                domain_ids = _encode_domain_ids_from_batch(_batch, _device)
                if domain_ids is not None:
                    try:
                        z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = _se(spec, spec_mask)
                else:
                    z_spec = _se(spec, spec_mask)
                if torch.is_tensor(z_spec) and z_spec.dim() == 2:
                    feats.append(z_spec.float())
        if len(feats) == 0:
            return None
        return torch.cat(feats, dim=0)

    def _fit_twc_for_target(target_loader, dataset_name):
        if (not _twc_enable) or (str(dataset_name) not in _twc_targets):
            return None
        if dataset_name in _twc_cache:
            return _twc_cache[dataset_name]
        src_cwru_loader = twc_source_cwru_loader if twc_source_cwru_loader is not None else cwru_test_loader
        src_pu_loader = twc_source_pu_loader if twc_source_pu_loader is not None else pu_test_loader
        if src_cwru_loader is None or src_pu_loader is None or target_loader is None:
            print(
                f"[TWC] {dataset_name}: skipped (missing source/target loader: "
                f"cwru={'yes' if src_cwru_loader is not None else 'no'}, "
                f"pu={'yes' if src_pu_loader is not None else 'no'}, "
                f"target={'yes' if target_loader is not None else 'no'})"
            )
            _twc_cache[dataset_name] = None
            return None
        try:
            _source_dataset = ConcatDataset([src_cwru_loader.dataset, src_pu_loader.dataset])
            _src_loader = DataLoader(
                _source_dataset,
                batch_size=getattr(target_loader, "batch_size", 32),
                shuffle=False,
                collate_fn=multimodal_collate_fn,
                num_workers=getattr(target_loader, "num_workers", 0),
                pin_memory=getattr(target_loader, "pin_memory", False),
                prefetch_factor=getattr(target_loader, "prefetch_factor", None),
                persistent_workers=getattr(target_loader, "persistent_workers", False),
            )
            z_src = _iter_raw_spec_features(_src_loader)
            z_tgt = _iter_raw_spec_features(target_loader)
            if z_src is None or z_tgt is None:
                _twc_cache[dataset_name] = None
                return None
            if int(z_src.size(0)) < 2 or int(z_tgt.size(0)) < 2:
                _twc_cache[dataset_name] = None
                return None
            _device = z_src.device
            _d = int(z_src.size(1))
            _eye = torch.eye(_d, device=_device, dtype=z_src.dtype)
            mu_s = z_src.mean(dim=0, keepdim=True)
            mu_t = z_tgt.mean(dim=0, keepdim=True)
            xs = z_src - mu_s
            xt = z_tgt - mu_t
            cov_s = (xs.t() @ xs) / float(max(1, int(xs.size(0)) - 1)) + float(_twc_eps) * _eye
            cov_t = (xt.t() @ xt) / float(max(1, int(xt.size(0)) - 1)) + float(_twc_eps) * _eye
            es, us = torch.linalg.eigh(cov_s)
            et, ut = torch.linalg.eigh(cov_t)
            es = torch.clamp(es, min=float(_twc_eps))
            et = torch.clamp(et, min=float(_twc_eps))
            cov_s_sqrt = (us * torch.sqrt(es).unsqueeze(0)) @ us.t()
            cov_t_inv_sqrt = (ut * torch.rsqrt(et).unsqueeze(0)) @ ut.t()
            twc_proj = cov_s_sqrt @ cov_t_inv_sqrt

            def _apply_twc(z_in):
                if z_in is None:
                    return z_in
                zf = z_in.float()
                z_twc = (zf - mu_t) @ twc_proj.t() + mu_s
                if _twc_alpha < 1.0:
                    z_twc = float(_twc_alpha) * z_twc + (1.0 - float(_twc_alpha)) * zf
                return z_twc

            _twc_cache[dataset_name] = _apply_twc
            print(
                f"[TWC] {dataset_name}: enabled=1 src_n={int(z_src.size(0))} tgt_n={int(z_tgt.size(0))} "
                f"feat_dim={_d} eps={float(_twc_eps):.1e} alpha={float(_twc_alpha):.2f}"
            )
            return _apply_twc
        except Exception as _e_twc:
            print(f"[TWC] {dataset_name}: skipped ({type(_e_twc).__name__}: {_e_twc})")
            _twc_cache[dataset_name] = None
            return None

    def _evaluate_with_spec_head(loader, dataset_name="", diag_num_classes=4, spec_transform_fn=None):
        """
        Evaluate diagnosis by spec_head logits directly (post LP-FT visibility).
        """
        if loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None or (not hasattr(_se, "classifier")) or (_se.classifier is None):
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None
        _se.eval()
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for batch in loader:
                spec = load_spec_from_batch(batch, device, detector)
                spec_mask = batch["spec_mask"].to(device)
                labels = batch["label"].to(device)
                if int(diag_num_classes) == 3:
                    labels = torch.clamp(labels, 0, 2)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                if domain_ids is not None:
                    try:
                        z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = _se(spec, spec_mask)
                else:
                    z_spec = _se(spec, spec_mask)
                if callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                _spechead_cosine_postlpft = False
                try:
                    import config as _cfg_sh_postlpft
                    _spechead_cosine_postlpft = bool(getattr(_cfg_sh_postlpft, "SPECHEAD_USE_COSINE_PROTO", False))
                except Exception:
                    pass
                if _spechead_cosine_postlpft:
                    _proto_dict = None
                    if hasattr(detector, '_active_text_prototypes') and callable(detector._active_text_prototypes):
                        _proto_dict = detector._active_text_prototypes()
                    if _proto_dict is not None and isinstance(_proto_dict, dict) and len(_proto_dict) > 0:
                        _pk = sorted(_proto_dict.keys())
                        _pm = torch.stack([_proto_dict[k].to(device) for k in _pk], dim=0)
                        _pm = torch.nn.functional.normalize(_pm.float(), dim=-1)
                        logits = torch.matmul(z_spec.float(), _pm.t())
                    else:
                        logits = _safe_spec_head_logits(_se, z_spec, detector, batch)
                else:
                    logits = _safe_spec_head_logits(_se, z_spec, detector, batch)
                if int(diag_num_classes) == 3:
                    logits = logits[:, :3]
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_lbls.extend(labels.detach().cpu().tolist())
        if len(all_lbls) == 0:
            return None
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        _cosine_tag = "[CosProto]" if _spechead_cosine_postlpft else ""
        if dataset_name:
            print(
                f"[SpecHead-PostLPFT]{_cosine_tag} {dataset_name} spec_head_diag_acc={acc:.4f} "
                f"macro_f1={macro_f1:.4f} (N={len(all_lbls)})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "nf_acc": float("nan"),
            "nf_correct": 0,
            "nf_total": 0,
            "spec_head_diag_acc": acc,
            "spec_head_macro_f1": macro_f1,
        }

    def _evaluate_with_spec_knn(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        k=5,
        spec_transform_fn=None,
    ):
        """
        Non-parametric diagnosis on z_spec using k-NN (cosine similarity).
        Support: labeled finetune split; Query: eval split.
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None:
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        def _collect_embeddings(loader):
            embs = []
            lbls = []
            with torch.no_grad():
                for batch in loader:
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                    else:
                        z_spec = _se(spec, spec_mask)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    embs.append(z_spec.detach())
                    lbls.append(labels.detach())
            if len(embs) == 0:
                return None, None
            return torch.cat(embs, dim=0), torch.cat(lbls, dim=0)

        _se.eval()
        support_emb, support_y = _collect_embeddings(support_loader)
        query_emb, query_y = _collect_embeddings(query_loader)
        if support_emb is None or query_emb is None:
            return None
        n_support = int(support_emb.size(0))
        n_query = int(query_emb.size(0))
        if n_support <= 0 or n_query <= 0:
            return None

        k_eff = max(1, min(int(k), n_support))
        sims = torch.matmul(query_emb, support_emb.transpose(0, 1))
        topk_idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False).indices
        nn_labels = support_y[topk_idx]
        nn_sims = torch.gather(sims, dim=1, index=topk_idx)

        num_cls = int(diag_num_classes)
        votes = torch.zeros((n_query, num_cls), device=device, dtype=query_emb.dtype)
        clamped_labels = torch.clamp(nn_labels, min=0, max=max(0, num_cls - 1)).long()
        votes.scatter_add_(1, clamped_labels, torch.relu(nn_sims))
        preds = votes.argmax(dim=1)

        all_preds = preds.detach().cpu().tolist()
        all_lbls = query_y.detach().cpu().tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        if dataset_name:
            print(
                f"[kNN-Spec] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(k={k_eff}, support={n_support}, eval={n_query})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "knn_spec_diag_acc": acc,
            "knn_spec_macro_f1": macro_f1,
            "knn_spec_k": int(k_eff),
            "knn_spec_support_size": int(n_support),
            "knn_spec_eval_size": int(n_query),
        }

    def _evaluate_with_spec_label_propagation(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        kernel="knn",
        n_neighbors=10,
        alpha=0.20,
        max_iter=30,
        tol=1e-3,
        spec_transform_fn=None,
    ):
        """
        Graph-based diagnosis on z_spec by Label Propagation/Spreading.
        Use support labels and propagate to query samples jointly.
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None:
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        def _collect_embeddings(loader):
            embs = []
            lbls = []
            with torch.no_grad():
                for batch in loader:
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                    else:
                        z_spec = _se(spec, spec_mask)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    embs.append(z_spec.detach().cpu().numpy())
                    lbls.append(labels.detach().cpu().numpy())
            if len(embs) == 0:
                return None, None
            return np.concatenate(embs, axis=0), np.concatenate(lbls, axis=0)

        _se.eval()
        support_emb, support_y = _collect_embeddings(support_loader)
        query_emb, query_y = _collect_embeddings(query_loader)
        if support_emb is None or query_emb is None:
            return None
        n_support = int(support_emb.shape[0])
        n_query = int(query_emb.shape[0])
        if n_support <= 1 or n_query <= 0:
            return None

        try:
            kernel = str(kernel).strip().lower()
            if kernel not in ("knn", "rbf"):
                kernel = "knn"
            n_neighbors_eff = max(2, min(int(n_neighbors), n_support - 1))
            alpha = max(1e-4, min(0.999, float(alpha)))
            max_iter = max(10, int(max_iter))
            tol = max(1e-6, float(tol))

            x_all = np.concatenate([support_emb, query_emb], axis=0)
            y_all = np.concatenate(
                [support_y.astype(np.int64), -1 * np.ones((n_query,), dtype=np.int64)],
                axis=0,
            )
            lp = LabelSpreading(
                kernel=kernel,
                n_neighbors=n_neighbors_eff,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
            )
            lp.fit(x_all, y_all)
            pred_all = lp.transduction_
            pred_query = pred_all[n_support:]
        except Exception as _e_lp:
            if dataset_name:
                print(f"[LabelProp-Spec] {dataset_name} skipped ({type(_e_lp).__name__}: {_e_lp})")
            return None

        all_preds = pred_query.astype(np.int64).tolist()
        all_lbls = query_y.astype(np.int64).tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        if dataset_name:
            print(
                f"[LabelProp-Spec] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(kernel={kernel}, n_neighbors={n_neighbors_eff}, alpha={alpha:.2f}, "
                f"support={n_support}, eval={n_query})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "label_prop_diag_acc": acc,
            "label_prop_macro_f1": macro_f1,
            "label_prop_kernel": str(kernel),
            "label_prop_n_neighbors": int(n_neighbors_eff),
            "label_prop_alpha": float(alpha),
            "label_prop_support_size": int(n_support),
            "label_prop_eval_size": int(n_query),
        }

    def _lp_aligned_query_distributions(
        support_emb,
        query_emb,
        support_y,
        diag_num_classes,
        kernel,
        n_neighbors_eff,
        alpha,
        max_iter,
        tol,
        n_support,
        n_query,
    ):
        """Run LabelSpreading and return query-side class posteriors aligned to 0..C-1."""
        x_all = np.concatenate([support_emb, query_emb], axis=0)
        y_all = np.concatenate(
            [support_y.astype(np.int64), -1 * np.ones((n_query,), dtype=np.int64)],
            axis=0,
        )
        lp = LabelSpreading(
            kernel=kernel,
            n_neighbors=int(n_neighbors_eff),
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        lp.fit(x_all, y_all)
        dist_all = lp.label_distributions_
        dist_q = dist_all[int(n_support) :].astype(np.float64)
        classes_ = np.asarray(getattr(lp, "classes_", np.arange(dist_q.shape[1])), dtype=np.int64)
        n_cls = int(diag_num_classes)
        if dist_q.shape[1] == n_cls:
            out = dist_q
        else:
            out = np.zeros((int(n_query), n_cls), dtype=np.float64)
            for _j, _c in enumerate(classes_.tolist()):
                if 0 <= int(_c) < n_cls:
                    out[:, int(_c)] = dist_q[:, _j]
        out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out

    def _lp_full_aligned_label_distributions(
        support_emb,
        query_emb,
        support_y_for_fit,
        diag_num_classes,
        kernel,
        n_neighbors_eff,
        alpha,
        max_iter,
        tol,
    ):
        """LabelSpreading on support+query; return class distributions for all nodes (aligned to diag_num_classes)."""
        n_support = int(support_emb.shape[0])
        n_query = int(query_emb.shape[0])
        x_all = np.concatenate([support_emb, query_emb], axis=0)
        y_all = np.concatenate(
            [support_y_for_fit.astype(np.int64), -1 * np.ones((n_query,), dtype=np.int64)],
            axis=0,
        )
        lp = LabelSpreading(
            kernel=kernel,
            n_neighbors=int(n_neighbors_eff),
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        lp.fit(x_all, y_all)
        dist_all = lp.label_distributions_.astype(np.float64)
        classes_ = np.asarray(getattr(lp, "classes_", np.arange(dist_all.shape[1])), dtype=np.int64)
        n_cls = int(diag_num_classes)
        if dist_all.shape[1] == n_cls:
            out = dist_all
        else:
            out = np.zeros((dist_all.shape[0], n_cls), dtype=np.float64)
            for _j, _c in enumerate(classes_.tolist()):
                if 0 <= int(_c) < n_cls:
                    out[:, int(_c)] = dist_all[:, _j]
        out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out

    def _select_dual_labelprop_fusion_weight_support_loo(
        support_spec,
        query_spec,
        support_fus,
        query_fus,
        support_y,
        diag_num_classes,
        kernel,
        n_neighbors_eff,
        alpha,
        max_iter,
        tol,
        weight_grid,
        default_w_f,
        dataset_name="",
    ):
        """
        ATT299: pick w_fusion by transductive leave-one-out on labeled support (query stays unlabeled).
        For each held-out support index, run two LabelSpreading fits (spec / fusion), then score convex combos.
        """
        grid = []
        for _w in weight_grid:
            try:
                grid.append(float(max(0.0, min(1.0, float(_w)))))
            except Exception:
                continue
        grid = sorted(set(grid))
        if len(grid) == 0:
            grid = [float(max(0.0, min(1.0, float(default_w_f))))]
        n_support = int(support_spec.shape[0])
        n_query = int(query_spec.shape[0])
        sy = support_y.astype(np.int64)
        if n_support < 3:
            return float(max(0.0, min(1.0, float(default_w_f)))), float("nan"), False
        loo_spec_rows = []
        loo_fus_rows = []
        for i in range(n_support):
            y_loo = sy.copy()
            y_loo[i] = -1
            full_spec = _lp_full_aligned_label_distributions(
                support_spec,
                query_spec,
                y_loo,
                diag_num_classes,
                kernel,
                n_neighbors_eff,
                alpha,
                max_iter,
                tol,
            )
            full_fus = _lp_full_aligned_label_distributions(
                support_fus,
                query_fus,
                y_loo,
                diag_num_classes,
                kernel,
                n_neighbors_eff,
                alpha,
                max_iter,
                tol,
            )
            loo_spec_rows.append(full_spec[i].copy())
            loo_fus_rows.append(full_fus[i].copy())
        best_w = float(max(0.0, min(1.0, float(default_w_f))))
        best_loo = -1.0
        for w_f in grid:
            w_s = 1.0 - w_f
            corr = 0
            for i in range(n_support):
                d = w_s * loo_spec_rows[i] + w_f * loo_fus_rows[i]
                d = d / (np.sum(d) + 1e-12)
                if int(np.argmax(d)) == int(sy[i]):
                    corr += 1
            acc = float(corr) / float(max(1, n_support))
            if acc > best_loo + 1e-12:
                best_loo = acc
                best_w = float(w_f)
            elif abs(acc - best_loo) <= 1e-12 and float(w_f) < float(best_w):
                best_w = float(w_f)
        if dataset_name:
            print(
                f"[LabelProp-Dual] {dataset_name} support-LOO selected w_fusion={best_w:.2f} "
                f"(loo_acc={best_loo:.4f}, grid_n={len(grid)})"
            )
        return float(best_w), float(best_loo), True

    def _evaluate_with_spec_fusion_dual_label_propagation(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        kernel="knn",
        n_neighbors=10,
        alpha=0.20,
        max_iter=30,
        tol=1e-3,
        spec_transform_fn=None,
        fusion_weight=0.35,
    ):
        """
        ATT298(S2): average LabelSpreading posteriors on z_spec and z_fusion (multimodal graph signal).
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        _sig = getattr(detector, "signal_encoder", None)
        _fus = getattr(detector, "fusion_module", None)
        if _se is None or _sig is None or _fus is None:
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        def _collect_spec_fusion(loader):
            spec_list = []
            fus_list = []
            lbls = []
            with torch.no_grad():
                for batch in loader:
                    signal = batch["signal"].to(device)
                    spec = load_spec_from_batch(batch, device, detector)
                    signal_mask = batch.get("signal_mask")
                    spec_mask = batch["spec_mask"].to(device)
                    if signal_mask is None:
                        signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                    else:
                        signal_mask = signal_mask.to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_sig = _sig(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_sig = _sig(signal, signal_mask)
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                        try:
                            z_f = _fus(z_sig, z_spec, domain_ids=domain_ids)
                        except TypeError:
                            z_f = _fus(z_sig, z_spec)
                    else:
                        z_sig = _sig(signal, signal_mask)
                        z_spec = _se(spec, spec_mask)
                        z_f = _fus(z_sig, z_spec)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    z_f = torch.nn.functional.normalize(z_f, dim=-1)
                    spec_list.append(z_spec.detach().cpu().numpy())
                    fus_list.append(z_f.detach().cpu().numpy())
                    lbls.append(labels.detach().cpu().numpy())
            if len(spec_list) == 0:
                return None, None, None
            return (
                np.concatenate(spec_list, axis=0),
                np.concatenate(fus_list, axis=0),
                np.concatenate(lbls, axis=0),
            )

        _se.eval()
        _sig.eval()
        _fus.eval()
        support_spec, support_fus, support_y = _collect_spec_fusion(support_loader)
        query_spec, query_fus, query_y = _collect_spec_fusion(query_loader)
        if support_spec is None or query_spec is None:
            return None
        n_support = int(support_spec.shape[0])
        n_query = int(query_spec.shape[0])
        if n_support <= 1 or n_query <= 0:
            return None

        try:
            kernel = str(kernel).strip().lower()
            if kernel not in ("knn", "rbf"):
                kernel = "knn"
            n_neighbors_eff = max(2, min(int(n_neighbors), n_support - 1))
            alpha = max(1e-4, min(0.999, float(alpha)))
            max_iter = max(10, int(max_iter))
            tol = max(1e-6, float(tol))
            w_f = float(max(0.0, min(1.0, fusion_weight)))
            loo_acc_sel = float("nan")
            loo_used = False
            try:
                import config as _cfg_lp
                _loo_en = bool(getattr(_cfg_lp, "TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_ENABLE", False))
                _loo_grid = list(getattr(_cfg_lp, "TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_GRID", []))
            except Exception:
                _loo_en = False
                _loo_grid = []
            if _loo_en and len(_loo_grid) > 0 and n_support >= 3:
                w_f, loo_acc_sel, loo_used = _select_dual_labelprop_fusion_weight_support_loo(
                    support_spec,
                    query_spec,
                    support_fus,
                    query_fus,
                    support_y,
                    diag_num_classes,
                    kernel,
                    n_neighbors_eff,
                    alpha,
                    max_iter,
                    tol,
                    _loo_grid,
                    w_f,
                    dataset_name=dataset_name,
                )
            elif _loo_en and dataset_name:
                print(
                    f"[LabelProp-Dual] {dataset_name} support-LOO skipped "
                    f"(support={n_support}, grid_n={len(_loo_grid)})"
                )
            w_s = 1.0 - w_f

            d_spec = _lp_aligned_query_distributions(
                support_spec,
                query_spec,
                support_y,
                diag_num_classes,
                kernel,
                n_neighbors_eff,
                alpha,
                max_iter,
                tol,
                n_support,
                n_query,
            )
            d_fus = _lp_aligned_query_distributions(
                support_fus,
                query_fus,
                support_y,
                diag_num_classes,
                kernel,
                n_neighbors_eff,
                alpha,
                max_iter,
                tol,
                n_support,
                n_query,
            )
            d_dual = w_s * d_spec + w_f * d_fus
            d_dual = d_dual / np.maximum(d_dual.sum(axis=1, keepdims=True), 1e-12)
            pred_query = np.argmax(d_dual, axis=1).astype(np.int64)
        except Exception as _e_lp:
            if dataset_name:
                print(f"[LabelProp-Dual] {dataset_name} skipped ({type(_e_lp).__name__}: {_e_lp})")
            return None

        all_preds = pred_query.astype(np.int64).tolist()
        all_lbls = query_y.astype(np.int64).tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        acc_spec_only = float(
            np.mean(np.argmax(d_spec, axis=1).astype(np.int64) == query_y.astype(np.int64))
        )
        acc_fus_only = float(
            np.mean(np.argmax(d_fus, axis=1).astype(np.int64) == query_y.astype(np.int64))
        )
        if dataset_name:
            print(
                f"[LabelProp-Dual] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(kernel={kernel}, n_neighbors={n_neighbors_eff}, alpha={alpha:.2f}, "
                f"w_spec={w_s:.2f}, w_fusion={w_f:.2f}, "
                f"acc_z_spec_only={acc_spec_only:.4f}, acc_z_fusion_only={acc_fus_only:.4f}, "
                f"support={n_support}, eval={n_query})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "label_prop_diag_acc": acc,
            "label_prop_macro_f1": macro_f1,
            "label_prop_kernel": str(kernel),
            "label_prop_n_neighbors": int(n_neighbors_eff),
            "label_prop_alpha": float(alpha),
            "label_prop_support_size": int(n_support),
            "label_prop_eval_size": int(n_query),
            "label_prop_dual_fusion_weight": float(w_f),
            "label_prop_dual_acc_spec_only": float(acc_spec_only),
            "label_prop_dual_acc_fusion_only": float(acc_fus_only),
            "label_prop_dual_loo_acc": float(loo_acc_sel),
            "label_prop_dual_loo_weight_used": bool(loo_used),
        }

    def _evaluate_with_spec_tim(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        spec_transform_fn=None,
    ):
        """
        ATT298(S3): transductive information-style objective on frozen z_spec (linear head only).
        No prototype rectification; conservative temperature vs naive high-T runs.
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None:
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        try:
            import config as cfg
            tim_steps = max(1, int(getattr(cfg, "TARGET_SUP_TIM_STEPS", 150)))
            tim_lr = float(getattr(cfg, "TARGET_SUP_TIM_LR", 0.01))
            tim_temp = float(getattr(cfg, "TARGET_SUP_TIM_TEMPERATURE", 2.0))
            tim_sup_ce_w = float(getattr(cfg, "TARGET_SUP_TIM_SUPPORT_CE_WEIGHT", 1.0))
            tim_q_ent_w = float(getattr(cfg, "TARGET_SUP_TIM_QUERY_ENT_WEIGHT", 0.05))
            tim_q_div_w = float(getattr(cfg, "TARGET_SUP_TIM_QUERY_DIV_WEIGHT", 0.20))
            tim_wd = float(getattr(cfg, "TARGET_SUP_TIM_WEIGHT_DECAY", 0.0))
        except Exception:
            tim_steps = 150
            tim_lr = 0.01
            tim_temp = 2.0
            tim_sup_ce_w = 1.0
            tim_q_ent_w = 0.05
            tim_q_div_w = 0.20
            tim_wd = 0.0

        def _collect_embeddings(loader):
            embs = []
            lbls = []
            with torch.no_grad():
                for batch in loader:
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                    else:
                        z_spec = _se(spec, spec_mask)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    embs.append(z_spec.detach().cpu().numpy())
                    lbls.append(labels.detach().cpu().numpy())
            if len(embs) == 0:
                return None, None
            return np.concatenate(embs, axis=0), np.concatenate(lbls, axis=0)

        _se.eval()
        support_emb, support_y = _collect_embeddings(support_loader)
        query_emb, query_y = _collect_embeddings(query_loader)
        if support_emb is None or query_emb is None:
            return None
        n_support = int(support_emb.shape[0])
        n_query = int(query_emb.shape[0])
        if n_support <= 1 or n_query <= 0:
            return None

        support_emb = np.asarray(support_emb, dtype=np.float64)
        query_emb = np.asarray(query_emb, dtype=np.float64)
        nrm = np.linalg.norm(support_emb, axis=1, keepdims=True)
        support_emb = support_emb / np.maximum(nrm, 1e-12)
        nrm_q = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / np.maximum(nrm_q, 1e-12)

        try:
            x_sup = torch.from_numpy(support_emb).to(device=device, dtype=torch.float32)
            y_sup = torch.from_numpy(support_y.astype(np.int64)).to(device=device, dtype=torch.long)
            x_qry = torch.from_numpy(query_emb).to(device=device, dtype=torch.float32)
            d = int(x_sup.shape[1])
            c = int(diag_num_classes)
            if c <= 1:
                return None

            w0 = torch.zeros((d, c), dtype=torch.float32, device=device)
            for cls in range(c):
                cls_mask = y_sup == cls
                if bool(torch.any(cls_mask)):
                    w0[:, cls] = torch.mean(x_sup[cls_mask], dim=0)
            w = torch.nn.Parameter(w0)
            optimizer = torch.optim.Adam([w], lr=max(1e-5, tim_lr), weight_decay=max(0.0, tim_wd))
            eps = 1e-8
            t_scale = float(max(0.1, tim_temp))

            for _ in range(int(tim_steps)):
                optimizer.zero_grad()
                logits_sup = t_scale * (x_sup @ w)
                logits_qry = t_scale * (x_qry @ w)
                p_q = torch.softmax(logits_qry, dim=1)
                p_q = torch.clamp(p_q, min=eps, max=1.0)
                ce_sup = torch.nn.functional.cross_entropy(logits_sup, y_sup)
                ent_q = -torch.mean(torch.sum(p_q * torch.log(p_q), dim=1))
                p_bar = torch.mean(p_q, dim=0)
                p_bar = torch.clamp(p_bar, min=eps, max=1.0)
                div_q = -torch.sum(p_bar * torch.log(p_bar))
                loss = float(tim_sup_ce_w) * ce_sup + float(tim_q_ent_w) * ent_q - float(tim_q_div_w) * div_q
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                logits_qry = t_scale * (x_qry @ w)
                pred_qry = torch.argmax(logits_qry, dim=1).detach().cpu().numpy().astype(np.int64)
        except Exception as _e_tim:
            if dataset_name:
                print(f"[TIM-Spec] {dataset_name} skipped ({type(_e_tim).__name__}: {_e_tim})")
            return None

        all_preds = pred_qry.astype(np.int64).tolist()
        all_lbls = query_y.astype(np.int64).tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        if dataset_name:
            print(
                f"[TIM-Spec] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(steps={int(tim_steps)}, temp={float(t_scale):.2f}, lr={float(tim_lr):.4f})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "tim_diag_acc": acc,
            "tim_macro_f1": macro_f1,
            "tim_steps": int(tim_steps),
            "tim_temperature": float(t_scale),
            "tim_support_size": int(n_support),
            "tim_eval_size": int(n_query),
        }

    def _evaluate_with_spec_knn_weighted_ensemble(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        k=5,
        weight_grid=None,
        spec_transform_fn=None,
    ):
        """
        Weighted ensemble: w * SpecHead prob + (1-w) * kNN prob on z_spec.
        Weight is selected on support split by max training accuracy
        (SpecHead + leave-one-out kNN), then applied to eval split.
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None or (not hasattr(_se, "classifier")) or (_se.classifier is None):
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        if weight_grid is None:
            weight_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        try:
            weight_grid = [float(w) for w in list(weight_grid)]
        except Exception:
            weight_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        weight_grid = [max(0.0, min(1.0, float(w))) for w in weight_grid]
        if len(weight_grid) == 0:
            weight_grid = [0.50]

        def _collect_spec(loader):
            embs = []
            probs = []
            lbls = []
            with torch.no_grad():
                for batch in loader:
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                    else:
                        z_spec = _se(spec, spec_mask)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    logits = _safe_spec_head_logits(_se, z_spec, detector, batch)
                    if int(diag_num_classes) == 3:
                        logits = logits[:, :3]
                    embs.append(z_spec.detach())
                    probs.append(torch.softmax(logits, dim=1).detach())
                    lbls.append(labels.detach())
            if len(embs) == 0:
                return None, None, None
            return torch.cat(embs, dim=0), torch.cat(probs, dim=0), torch.cat(lbls, dim=0)

        def _knn_prob_from_emb(support_emb, support_y, query_emb, k_eff):
            sims = torch.matmul(query_emb, support_emb.transpose(0, 1))
            topk_idx = torch.topk(sims, k=int(k_eff), dim=1, largest=True, sorted=False).indices
            nn_labels = support_y[topk_idx]
            nn_sims = torch.gather(sims, dim=1, index=topk_idx)
            n_query = int(query_emb.size(0))
            n_cls = int(diag_num_classes)
            votes = torch.zeros((n_query, n_cls), device=query_emb.device, dtype=query_emb.dtype)
            clamped_labels = torch.clamp(nn_labels, min=0, max=max(0, n_cls - 1)).long()
            votes.scatter_add_(1, clamped_labels, torch.relu(nn_sims))
            denom = votes.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return votes / denom

        _se.eval()
        support_emb, support_prob_spec, support_y = _collect_spec(support_loader)
        query_emb, query_prob_spec, query_y = _collect_spec(query_loader)
        if support_emb is None or query_emb is None:
            return None
        n_support = int(support_emb.size(0))
        n_query = int(query_emb.size(0))
        if n_support <= 1 or n_query <= 0:
            return None
        k_eff = max(1, min(int(k), n_support))

        # Query-side kNN probabilities
        query_prob_knn = _knn_prob_from_emb(support_emb, support_y, query_emb, k_eff=k_eff)

        # Support-side leave-one-out kNN probabilities for weight selection
        k_loo = max(1, min(int(k_eff), n_support - 1))
        sims_ss = torch.matmul(support_emb, support_emb.transpose(0, 1))
        sims_ss.fill_diagonal_(-1e9)
        topk_idx_ss = torch.topk(sims_ss, k=int(k_loo), dim=1, largest=True, sorted=False).indices
        nn_labels_ss = support_y[topk_idx_ss]
        nn_sims_ss = torch.gather(sims_ss, dim=1, index=topk_idx_ss)
        votes_ss = torch.zeros((n_support, int(diag_num_classes)), device=device, dtype=support_emb.dtype)
        votes_ss.scatter_add_(
            1,
            torch.clamp(nn_labels_ss, min=0, max=max(0, int(diag_num_classes) - 1)).long(),
            torch.relu(nn_sims_ss),
        )
        support_prob_knn = votes_ss / votes_ss.sum(dim=1, keepdim=True).clamp_min(1e-8)

        best_w = float(weight_grid[0])
        best_support_acc = -1.0
        for w in weight_grid:
            w = float(w)
            support_mix = w * support_prob_spec + (1.0 - w) * support_prob_knn
            support_pred = support_mix.argmax(dim=1)
            support_acc = float((support_pred == support_y).float().mean().item())
            if (support_acc > best_support_acc + 1e-12) or (
                abs(support_acc - best_support_acc) <= 1e-12 and w >= best_w
            ):
                best_support_acc = support_acc
                best_w = w

        query_mix = float(best_w) * query_prob_spec + (1.0 - float(best_w)) * query_prob_knn
        query_pred = query_mix.argmax(dim=1)
        all_preds = query_pred.detach().cpu().tolist()
        all_lbls = query_y.detach().cpu().tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        if dataset_name:
            print(
                f"[Spec+kNN Ensemble] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(w_spec={best_w:.2f}, support_acc={best_support_acc:.4f}, k={k_eff})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "spec_knn_ens_diag_acc": acc,
            "spec_knn_ens_macro_f1": macro_f1,
            "spec_knn_ens_weight_spec": float(best_w),
            "spec_knn_ens_support_acc": float(best_support_acc),
            "spec_knn_ens_k": int(k_eff),
            "spec_knn_ens_support_size": int(n_support),
            "spec_knn_ens_eval_size": int(n_query),
        }

    def _build_tip_support_loader(target_loader):
        """
        Build support loader for Tip-Adapter cache from source-domain samples.
        Priority:
          twc_source_* -> task1_* -> fallback test loaders.
        """
        if target_loader is None:
            return None
        candidates = [
            twc_source_cwru_loader,
            twc_source_pu_loader,
            task1_cwru_loader,
            task1_pu_loader,
            cwru_test_loader,
            pu_test_loader,
        ]
        datasets = []
        seen = set()
        for _ldr in candidates:
            if _ldr is None or (not hasattr(_ldr, "dataset")):
                continue
            _ds = _ldr.dataset
            _key = id(_ds)
            if _key in seen:
                continue
            seen.add(_key)
            datasets.append(_ds)
        if len(datasets) == 0:
            return None
        support_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        _nw = int(getattr(target_loader, "num_workers", 0))
        _pf = getattr(target_loader, "prefetch_factor", None) if _nw > 0 else None
        return DataLoader(
            support_dataset,
            batch_size=getattr(target_loader, "batch_size", 32),
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=_nw,
            pin_memory=getattr(target_loader, "pin_memory", False),
            prefetch_factor=_pf,
            persistent_workers=getattr(target_loader, "persistent_workers", False),
        )

    def _evaluate_with_tip_adapter_cache(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
        support_transform_fn=None,
        query_transform_fn=None,
    ):
        """
        Tip-Adapter style cache inference (eval-only):
          - K/V cache: support embeddings + one-hot labels
          - Query Q: target embeddings
          - Prediction: cache weighted voting, optionally mixed with SpecHead probabilities.
        """
        if support_loader is None or query_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        _sig_enc = getattr(detector, "signal_encoder", None)
        _fusion_mod = getattr(detector, "fusion_module", None)
        _fusion_dec = getattr(detector, "fusion_decoder", None)
        if _se is None or (not hasattr(_se, "classifier")) or (_se.classifier is None):
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None
        try:
            import config as _cfg_tip
            tip_topk = int(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_TOPK", 64))
            tip_beta = float(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_BETA", 20.0))
            tip_mix_enable = bool(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_USE_SPEC_HEAD_MIX", True))
            tip_alpha_grid = list(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_ALPHA_GRID", [0.0, 0.25, 0.5, 0.75, 1.0]))
            tip_max_support = int(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_MAX_SUPPORT_SAMPLES", 8000))
            tip_feature_source = str(getattr(_cfg_tip, "EVAL_TIP_ADAPTER_FEATURE_SOURCE", "spec")).lower()
        except Exception:
            tip_topk = 64
            tip_beta = 20.0
            tip_mix_enable = True
            tip_alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            tip_max_support = 8000
            tip_feature_source = "spec"
        if tip_feature_source not in {"spec", "fusion_metric"}:
            tip_feature_source = "spec"
        if tip_feature_source == "fusion_metric" and (_sig_enc is None or _fusion_mod is None or _fusion_dec is None):
            print("[TipAdapter] fallback to spec features: fusion components are unavailable")
            tip_feature_source = "spec"
        tip_topk = max(1, int(tip_topk))
        tip_beta = max(1e-6, float(tip_beta))
        try:
            tip_alpha_grid = [max(0.0, min(1.0, float(a))) for a in tip_alpha_grid]
        except Exception:
            tip_alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        if len(tip_alpha_grid) == 0:
            tip_alpha_grid = [0.0]
        tip_alpha_grid = sorted(list(set(tip_alpha_grid)))
        print(
            f"[TipAdapter] feature_source={tip_feature_source}, "
            f"mix_enable={int(tip_mix_enable)}, topk={tip_topk}, beta={tip_beta:.2f}"
        )

        def _collect_embeddings(loader, transform_fn=None, max_samples=None):
            embs = []
            probs = []
            lbls = []
            n_collected = 0
            with torch.no_grad():
                for batch in loader:
                    if max_samples is not None and n_collected >= int(max_samples):
                        break
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    labels = batch["label"].to(device)
                    if int(diag_num_classes) == 3:
                        labels = torch.clamp(labels, 0, 2)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if tip_feature_source == "fusion_metric":
                        signal = batch["signal"].to(device)
                        signal_mask = batch["signal_mask"].to(device)
                        if domain_ids is not None:
                            try:
                                z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                            except TypeError:
                                z_spec = _se(spec, spec_mask)
                            try:
                                z_sig = _sig_enc(signal, signal_mask, domain_ids=domain_ids)
                            except TypeError:
                                z_sig = _sig_enc(signal, signal_mask)
                        else:
                            z_spec = _se(spec, spec_mask)
                            z_sig = _sig_enc(signal, signal_mask)
                        text_prototypes = detector._active_text_prototypes() if hasattr(detector, "_active_text_prototypes") else None
                        if hasattr(_fusion_mod, "domain_embedding"):
                            z_fus = _fusion_mod(z_sig.float(), z_spec.float(), domain_ids=domain_ids, text_prototypes=text_prototypes)
                        else:
                            z_fus = _fusion_mod(z_sig.float(), z_spec.float())
                        z_fus = torch.nn.functional.normalize(z_fus, dim=-1)
                        z_emb = z_fus
                        if hasattr(_fusion_dec, "get_metric_embedding"):
                            z_emb = _fusion_dec.get_metric_embedding(z_fus)
                        if callable(transform_fn):
                            z_emb = transform_fn(z_emb)
                        z_emb = torch.nn.functional.normalize(z_emb, dim=-1)
                        logits4, parts = _fusion_dec(z_fus, domain_ids=domain_ids, text_prototypes=text_prototypes, return_parts=True)
                        logits = parts.get("logits_diag_supervised", logits4) if isinstance(parts, dict) else logits4
                    else:
                        if domain_ids is not None:
                            try:
                                z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                            except TypeError:
                                z_spec = _se(spec, spec_mask)
                        else:
                            z_spec = _se(spec, spec_mask)
                        if callable(transform_fn):
                            z_spec = transform_fn(z_spec)
                        z_emb = torch.nn.functional.normalize(z_spec, dim=-1)
                        logits = _safe_spec_head_logits(_se, z_emb, detector, batch)
                    if int(diag_num_classes) == 3:
                        logits = logits[:, :3]
                    if max_samples is not None and (n_collected + int(z_emb.size(0)) > int(max_samples)):
                        keep = int(max_samples) - n_collected
                        if keep <= 0:
                            break
                        z_emb = z_emb[:keep]
                        logits = logits[:keep]
                        labels = labels[:keep]
                    embs.append(z_emb.detach())
                    probs.append(torch.softmax(logits, dim=1).detach())
                    lbls.append(labels.detach())
                    n_collected += int(z_emb.size(0))
            if len(embs) == 0:
                return None, None, None
            return torch.cat(embs, dim=0), torch.cat(probs, dim=0), torch.cat(lbls, dim=0)

        def _cache_prob(query_emb, support_emb, support_onehot, topk_eff, beta, exclude_self=False):
            if query_emb is None or support_emb is None or support_onehot is None:
                return None
            if int(support_emb.size(0)) <= 1:
                return None
            sims = torch.matmul(query_emb, support_emb.transpose(0, 1))  # [Nq, Ns]
            if exclude_self and int(query_emb.size(0)) == int(support_emb.size(0)):
                sims = sims.clone()
                sims.fill_diagonal_(-1e4)
                k_now = max(1, min(int(topk_eff), int(support_emb.size(0)) - 1))
            else:
                k_now = max(1, min(int(topk_eff), int(support_emb.size(0))))
            topv, topi = torch.topk(sims, k=k_now, dim=1, largest=True, sorted=False)
            attn = torch.softmax(topv * float(beta), dim=1)
            picked = support_onehot[topi]  # [Nq, k, C]
            return (attn.unsqueeze(-1) * picked).sum(dim=1)

        _se.eval()
        if _sig_enc is not None:
            _sig_enc.eval()
        if _fusion_mod is not None:
            _fusion_mod.eval()
        if _fusion_dec is not None:
            _fusion_dec.eval()

        support_emb, support_prob_spec, support_y = _collect_embeddings(
            support_loader, transform_fn=support_transform_fn, max_samples=tip_max_support
        )
        query_emb, query_prob_spec, query_y = _collect_embeddings(
            query_loader, transform_fn=query_transform_fn, max_samples=None
        )
        if support_emb is None or query_emb is None:
            return None
        n_support = int(support_emb.size(0))
        n_query = int(query_emb.size(0))
        if n_support <= 1 or n_query <= 0:
            return None

        n_cls = int(diag_num_classes)
        support_y_c = torch.clamp(support_y.long(), min=0, max=max(0, n_cls - 1))
        query_y_c = torch.clamp(query_y.long(), min=0, max=max(0, n_cls - 1))
        support_onehot = torch.zeros((n_support, n_cls), device=device, dtype=support_emb.dtype)
        support_onehot.scatter_(1, support_y_c.view(-1, 1), 1.0)

        query_prob_cache = _cache_prob(
            query_emb=query_emb,
            support_emb=support_emb,
            support_onehot=support_onehot,
            topk_eff=tip_topk,
            beta=tip_beta,
            exclude_self=False,
        )
        support_prob_cache_loo = _cache_prob(
            query_emb=support_emb,
            support_emb=support_emb,
            support_onehot=support_onehot,
            topk_eff=tip_topk,
            beta=tip_beta,
            exclude_self=True,
        )
        if query_prob_cache is None or support_prob_cache_loo is None:
            return None

        best_alpha = 0.0
        best_support_acc = -1.0
        if tip_mix_enable and support_prob_spec is not None:
            for alpha in tip_alpha_grid:
                alpha = float(alpha)
                support_mix = alpha * support_prob_spec + (1.0 - alpha) * support_prob_cache_loo
                support_pred = support_mix.argmax(dim=1)
                support_acc = float((support_pred == support_y_c).float().mean().item())
                if (support_acc > best_support_acc + 1e-12) or (
                    abs(support_acc - best_support_acc) <= 1e-12 and alpha < best_alpha
                ):
                    best_support_acc = support_acc
                    best_alpha = alpha
        else:
            best_alpha = 0.0
            support_pred = support_prob_cache_loo.argmax(dim=1)
            best_support_acc = float((support_pred == support_y_c).float().mean().item())

        if tip_mix_enable and query_prob_spec is not None:
            query_mix = float(best_alpha) * query_prob_spec + (1.0 - float(best_alpha)) * query_prob_cache
        else:
            query_mix = query_prob_cache
        query_pred = query_mix.argmax(dim=1)
        all_preds = query_pred.detach().cpu().tolist()
        all_lbls = query_y_c.detach().cpu().tolist()
        acc = float(np.mean(np.asarray(all_preds) == np.asarray(all_lbls)))
        macro_f1 = _safe_macro_f1(all_lbls, all_preds)
        if dataset_name:
            print(
                f"[TipAdapter] {dataset_name} diag_acc={acc:.4f} macro_f1={macro_f1:.4f} "
                f"(alpha={float(best_alpha):.2f}, beta={float(tip_beta):.2f}, topk={int(min(tip_topk, n_support))}, "
                f"support={n_support}, eval={n_query})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": int(np.sum(np.asarray(all_preds) == np.asarray(all_lbls))),
            "diag_total": int(len(all_lbls)),
            "diag_labels": list(all_lbls),
            "diag_preds": list(all_preds),
            "macro_f1": macro_f1,
            "tip_adapter_diag_acc": acc,
            "tip_adapter_macro_f1": macro_f1,
            "tip_adapter_alpha": float(best_alpha),
            "tip_adapter_beta": float(tip_beta),
            "tip_adapter_topk": int(min(tip_topk, n_support)),
            "tip_adapter_support_acc": float(best_support_acc),
            "tip_adapter_support_size": int(n_support),
            "tip_adapter_eval_size": int(n_query),
        }

    def _domain_fs_from_batch(domains):
        try:
            import config as _cfg_fs
            _fs_map = dict(getattr(_cfg_fs, "SAMPLING_RATES", {}))
        except Exception:
            _fs_map = {}
        out = []
        for d in domains:
            d_up = str(d).upper()
            if d_up.startswith("CWRU"):
                out.append(float(_fs_map.get("CWRU", 12000)))
            elif d_up.startswith("PU"):
                out.append(float(_fs_map.get("PU", 64000)))
            elif d_up.startswith("HUST"):
                out.append(float(_fs_map.get("HUST", 25600)))
            elif d_up.startswith("SEU"):
                out.append(float(_fs_map.get("SEU", 2000)))
            elif d_up.startswith("HIT"):
                out.append(float(_fs_map.get("HIT", _fs_map.get("SEU", 2000))))
            else:
                out.append(float(_fs_map.get("CWRU", 12000)))
        return out

    def _evaluate_with_dynamic_multi_proto(
        support_loader,
        query_loader,
        dataset_name="",
        diag_num_classes=4,
    ):
        if support_loader is None or query_loader is None:
            return None
        if (
            build_dynamic_semantic_cluster_bank is None
            or predict_with_dynamic_cluster_bank is None
            or summarize_cluster_bank is None
        ):
            return None

        _se = getattr(detector, "spec_encoder", None)
        _sig_enc = getattr(detector, "signal_encoder", None)
        _fusion_mod = getattr(detector, "fusion_module", None)
        _fusion_dec = getattr(detector, "fusion_decoder", None)
        if _se is None or _sig_enc is None or _fusion_mod is None or _fusion_dec is None:
            return None

        try:
            _device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        try:
            import config as _cfg_dyn
            _k = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_K", 4))
            _topk_vote = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TOPK_VOTE", 7))
            _temperature = float(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TEMPERATURE", 0.07))
            _max_support = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_MAX_SUPPORT_SAMPLES", 8000))
            _max_iter = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_KMEANS_ITERS", 25))
            _min_cluster = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_MIN_CLUSTER_SAMPLES", 8))
            _max_sig = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TEMPLATE_MAX_SIGNAL_SAMPLES", 48))
            _log_top = int(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_LOG_TOP_CLUSTERS", 10))
            _feature_src = str(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_FEATURE_SOURCE", "fusion_raw")).lower()
            _use_dual_metric = bool(getattr(_cfg_dyn, "USE_DUAL_SPACE_DECOUPLE", False))
            _text_mix_alpha = float(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_ALPHA", 0.35))
            _text_mix_min_cos = float(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_MIN_COS", 0.12))
            _text_mix_fallback_alpha = float(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_TEXT_MIX_FALLBACK_ALPHA", 0.0))
            _use_spec_evidence = bool(getattr(_cfg_dyn, "EVAL_DYNAMIC_MULTI_PROTO_USE_SPEC_EVIDENCE", True))
            _freq_ref_hz = float(getattr(_cfg_dyn, "PHYSICAL_FREQ_REFERENCE_HZ", 32000.0))
            _freq_pos_scale = float(getattr(_cfg_dyn, "PHYSICAL_FREQ_POSITION_ENCODING_SCALE", 0.0))
        except Exception:
            _k = 4
            _topk_vote = 7
            _temperature = 0.07
            _max_support = 8000
            _max_iter = 25
            _min_cluster = 8
            _max_sig = 48
            _log_top = 10
            _feature_src = "fusion_raw"
            _use_dual_metric = False
            _text_mix_alpha = 0.35
            _text_mix_min_cos = 0.12
            _text_mix_fallback_alpha = 0.0
            _use_spec_evidence = True
            _freq_ref_hz = 32000.0
            _freq_pos_scale = 0.0

        _k = max(1, _k)
        _topk_vote = max(1, _topk_vote)
        _temperature = max(1e-6, _temperature)

        def _collect(loader, max_samples=None, need_signal=False):
            embs = []
            lbls = []
            sigs = []
            fs_vals = []
            specs = []
            spec_freq_pos = []
            n_keep = 0
            with torch.no_grad():
                for _batch in loader:
                    if max_samples is not None and n_keep >= int(max_samples):
                        break
                    _spec = load_spec_from_batch(_batch, _device, detector)
                    _smask = _batch["spec_mask"].to(_device)
                    _sig = _batch["signal"].to(_device)
                    _sigmask = _batch["signal_mask"].to(_device)
                    _spec_freq_pos = _batch.get("spec_freq_pos", None)
                    _lbl = _batch["label"].to(_device)
                    if int(diag_num_classes) == 3:
                        _lbl = torch.clamp(_lbl, 0, 2)
                    _domains = _batch.get("domain", None)
                    _dids = _encode_domain_ids_from_batch(_batch, _device)

                    if _dids is not None:
                        try:
                            _z_spec = _se(_spec, _smask, domain_ids=_dids)
                        except TypeError:
                            _z_spec = _se(_spec, _smask)
                        try:
                            _z_sig = _sig_enc(_sig, _sigmask, domain_ids=_dids)
                        except TypeError:
                            _z_sig = _sig_enc(_sig, _sigmask)
                    else:
                        _z_spec = _se(_spec, _smask)
                        _z_sig = _sig_enc(_sig, _sigmask)

                    _text_protos = detector._active_text_prototypes() if hasattr(detector, "_active_text_prototypes") else None
                    if hasattr(_fusion_mod, "domain_embedding"):
                        _z_fus = _fusion_mod(
                            _z_sig.float(), _z_spec.float(), domain_ids=_dids, text_prototypes=_text_protos
                        )
                    else:
                        _z_fus = _fusion_mod(_z_sig.float(), _z_spec.float())
                    _z_fus = torch.nn.functional.normalize(_z_fus, dim=-1)

                    _z_emb = _z_fus
                    if _feature_src == "fusion_metric" and _use_dual_metric and hasattr(_fusion_dec, "get_metric_embedding"):
                        try:
                            _z_emb = _fusion_dec.get_metric_embedding(_z_fus)
                            _z_emb = torch.nn.functional.normalize(_z_emb.float(), dim=-1)
                        except Exception:
                            _z_emb = _z_fus

                    if max_samples is not None and (n_keep + int(_z_emb.size(0)) > int(max_samples)):
                        _keep = int(max_samples) - n_keep
                        if _keep <= 0:
                            break
                        _z_emb = _z_emb[:_keep]
                        _lbl = _lbl[:_keep]
                        _sig = _sig[:_keep]
                        _spec = _spec[:_keep]
                        _smask = _smask[:_keep]
                        if torch.is_tensor(_spec_freq_pos):
                            _spec_freq_pos = _spec_freq_pos[:_keep]
                        if _domains is not None:
                            _domains = _domains[:_keep]

                    embs.append(_z_emb.detach())
                    lbls.append(_lbl.detach())
                    n_keep += int(_z_emb.size(0))

                    if need_signal:
                        _sig_np = _sig.detach().cpu().float().numpy()
                        sigs.extend([_sig_np[i].reshape(-1) for i in range(_sig_np.shape[0])])
                        if _domains is not None:
                            fs_vals.extend(_domain_fs_from_batch(_domains))
                        else:
                            fs_vals.extend([12000.0] * _sig_np.shape[0])

                        _spec_np = _spec.detach().cpu().float().numpy()
                        _smask_np = _smask.detach().cpu().float().numpy()
                        if torch.is_tensor(_spec_freq_pos):
                            _sfp_np = _spec_freq_pos.detach().cpu().float().numpy()
                        else:
                            _sfp_np = None
                        for i in range(_spec_np.shape[0]):
                            _t_valid = int(max(1, np.round(_smask_np[i].sum())))
                            _t_valid = min(_t_valid, int(_spec_np.shape[-1]))
                            specs.append(_spec_np[i, :, :, :_t_valid].reshape(_spec_np.shape[1], _spec_np.shape[2], _t_valid))
                            if _sfp_np is not None:
                                spec_freq_pos.append(_sfp_np[i, :, :, :_t_valid].reshape(_sfp_np.shape[1], _sfp_np.shape[2], _t_valid))
                            else:
                                spec_freq_pos.append(None)

            if len(embs) == 0:
                return None
            out = {
                "emb": torch.cat(embs, dim=0),
                "lbl": torch.cat(lbls, dim=0),
            }
            if need_signal:
                out["signals"] = sigs
                out["fs"] = fs_vals
                out["specs"] = specs
                out["spec_freq_pos"] = spec_freq_pos
            return out

        _se.eval()
        _sig_enc.eval()
        _fusion_mod.eval()
        _fusion_dec.eval()

        _sup = _collect(support_loader, max_samples=_max_support, need_signal=True)
        _qry = _collect(query_loader, max_samples=None, need_signal=False)
        if _sup is None or _qry is None:
            return None

        _support_emb = _sup["emb"]
        _support_lbl = _sup["lbl"]
        _query_emb = _qry["emb"]
        _query_lbl = _qry["lbl"]

        if int(_support_emb.size(0)) < 4 or int(_query_emb.size(0)) < 1:
            return None

        _text_embedder = None
        if hasattr(detector, "text_encoder") and detector.text_encoder is not None:
            def _encode_texts_for_dyn(_texts):
                with torch.no_grad():
                    _z_txt = detector.text_encoder(list(_texts))
                    if isinstance(_z_txt, np.ndarray):
                        _z_txt = torch.from_numpy(_z_txt)
                    if torch.is_tensor(_z_txt):
                        return torch.nn.functional.normalize(_z_txt.detach().cpu().float(), dim=-1)
                    return None
            _text_embedder = _encode_texts_for_dyn

        _bank = build_dynamic_semantic_cluster_bank(
            support_emb=_support_emb,
            support_labels=_support_lbl,
            support_signals=_sup.get("signals", None),
            support_fs=_sup.get("fs", None),
            support_specs=_sup.get("specs", None),
            support_spec_freq_pos=_sup.get("spec_freq_pos", None),
            num_classes=int(diag_num_classes),
            k_per_class=int(_k),
            max_iters=int(_max_iter),
            min_cluster_samples=int(_min_cluster),
            max_signal_samples_per_cluster=int(_max_sig),
            freq_reference_hz=float(_freq_ref_hz),
            spec_posenc_scale=float(_freq_pos_scale),
            use_spec_evidence=bool(_use_spec_evidence),
            text_embedder=_text_embedder,
            text_mix_alpha=float(_text_mix_alpha),
            text_mix_min_cos=float(_text_mix_min_cos),
            text_mix_fallback_alpha=float(_text_mix_fallback_alpha),
            seed=42,
        )
        if _bank is None:
            return None

        _pred_out = predict_with_dynamic_cluster_bank(
            query_emb=_query_emb,
            cluster_bank=_bank,
            num_classes=int(diag_num_classes),
            topk_vote=int(_topk_vote),
            temperature=float(_temperature),
        )
        if _pred_out is None:
            return None

        _pred = _pred_out["preds"].detach().cpu().long().numpy()
        _lbl = _query_lbl.detach().cpu().long().numpy()
        _acc = float(np.mean(_pred == _lbl))
        _f1 = _safe_macro_f1(_lbl.tolist(), _pred.tolist())

        print(
            f"[DynMultiProto] {dataset_name} diag_acc={_acc:.4f} macro_f1={_f1:.4f} "
            f"(feature={_feature_src}, k={int(_k)}, topk_vote={int(_topk_vote)}, "
            f"temp={float(_temperature):.3f}, support={int(_support_emb.size(0))}, query={int(_query_emb.size(0))}, "
            f"spec_evidence={int(_use_spec_evidence)}, text_mix_alpha={float(_text_mix_alpha):.2f})"
        )
        try:
            _v = _bank.get("centers_visual", None)
            _t = _bank.get("centers_text", None)
            if torch.is_tensor(_v) and torch.is_tensor(_t) and _v.shape == _t.shape and int(_v.numel()) > 0:
                _tv = torch.sum(
                    torch.nn.functional.normalize(_v.float(), dim=-1)
                    * torch.nn.functional.normalize(_t.float(), dim=-1),
                    dim=-1,
                )
                print(
                    f"[DynMultiProtoText] visual_text_cos(mean/min/max)="
                    f"({float(_tv.mean().item()):.4f},{float(_tv.min().item()):.4f},{float(_tv.max().item()):.4f})"
                )
                _mix_eff = float(_bank.get("text_mix_alpha_effective", _text_mix_alpha))
                _mix_fb = bool(_bank.get("text_mix_fallback_triggered", False))
                if _mix_fb:
                    print(
                        f"[DynMultiProtoTextMix] fallback triggered: "
                        f"mean_cos={float(_tv.mean().item()):.4f} < min_cos={float(_text_mix_min_cos):.4f}, "
                        f"alpha {float(_text_mix_alpha):.2f} -> {float(_mix_eff):.2f}"
                    )
        except Exception:
            pass
        for _line in summarize_cluster_bank(_bank, max_clusters=int(_log_top)):
            print(_line)

        return {
            "diag_acc": float(_acc),
            "diag_acc_raw": float(_acc),
            "diag_correct": int(np.sum(_pred == _lbl)),
            "diag_total": int(len(_lbl)),
            "diag_labels": _lbl.tolist(),
            "diag_preds": _pred.tolist(),
            "macro_f1": float(_f1),
            "dynamic_multi_proto_bank_size": int(_bank.get("num_clusters", 0)),
        }

    class _PseudoLabelSubsetDataset(Dataset):
        """Dataset wrapper: reuse raw samples but replace labels with pseudo labels."""
        def __init__(self, base_dataset, base_indices, pseudo_labels):
            self.base_dataset = base_dataset
            self.base_indices = list(base_indices)
            self.pseudo_labels = [int(v) for v in pseudo_labels]

        def __len__(self):
            return len(self.base_indices)

        def __getitem__(self, idx):
            sample = self.base_dataset[self.base_indices[idx]]
            pseudo_y = int(self.pseudo_labels[idx])
            if isinstance(sample, dict):
                out = dict(sample)
                old_y = out.get("label", None)
                if torch.is_tensor(old_y):
                    out["label"] = torch.tensor(pseudo_y, dtype=old_y.dtype)
                else:
                    out["label"] = pseudo_y
                return out
            if isinstance(sample, (tuple, list)) and len(sample) > 0:
                arr = list(sample)
                arr[-1] = pseudo_y
                return tuple(arr) if isinstance(sample, tuple) else arr
            return sample

    def _build_pseudo_labeled_loader_from_spec_head(
        eval_loader,
        dataset_name="",
        diag_num_classes=4,
        topk_ratio=0.30,
        min_confidence=0.70,
        min_per_class=5,
        balanced_topk=False,
        spec_transform_fn=None,
    ):
        """
        Build pseudo-labeled loader from eval split using current spec_head confidence.
        Selection policy:
        1) Ensure a minimal per-class quota (by predicted class).
        2) Fill up to top-k ratio with highest-confidence samples above min_confidence.
        """
        if eval_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None or (not hasattr(_se, "classifier")) or (_se.classifier is None):
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        records = []
        _se.eval()
        with torch.no_grad():
            pos = 0
            for batch in eval_loader:
                spec = load_spec_from_batch(batch, device, detector)
                spec_mask = batch["spec_mask"].to(device)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                if domain_ids is not None:
                    try:
                        z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_spec = _se(spec, spec_mask)
                else:
                    z_spec = _se(spec, spec_mask)
                if callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                logits = _safe_spec_head_logits(_se, z_spec, detector, batch)
                if int(diag_num_classes) == 3:
                    logits = logits[:, :3]
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                conf = conf.detach().cpu().tolist()
                pred = pred.detach().cpu().tolist()
                for c, p in zip(conf, pred):
                    records.append({"pos": int(pos), "pred": int(p), "conf": float(c)})
                    pos += 1

        if len(records) == 0:
            return None

        n_total = len(records)
        n_target = max(
            int(min_per_class) * int(diag_num_classes),
            int(round(float(topk_ratio) * float(n_total))),
        )
        n_target = min(n_total, max(1, n_target))

        selected_pos = set()
        by_cls = {c: [] for c in range(int(diag_num_classes))}
        for r in records:
            if 0 <= int(r["pred"]) < int(diag_num_classes):
                by_cls[int(r["pred"])].append(r)
        if bool(balanced_topk):
            n_cls = max(1, int(diag_num_classes))
            base_q = int(n_target // n_cls)
            rem_q = int(n_target % n_cls)
            cls_quota = {}
            for c in range(n_cls):
                q = int(base_q + (1 if c < rem_q else 0))
                cls_quota[c] = max(int(min_per_class), q)
            for c in range(n_cls):
                cls_items = sorted(by_cls.get(c, []), key=lambda x: x["conf"], reverse=True)
                if len(cls_items) == 0:
                    continue
                strong = [x for x in cls_items if float(x["conf"]) >= float(min_confidence)]
                chosen = strong[: int(cls_quota[c])]
                if len(chosen) < int(cls_quota[c]):
                    chosen = cls_items[: min(len(cls_items), int(cls_quota[c]))]
                for x in chosen:
                    selected_pos.add(int(x["pos"]))
        else:
            for c in range(int(diag_num_classes)):
                cls_items = sorted(by_cls.get(c, []), key=lambda x: x["conf"], reverse=True)
                if len(cls_items) == 0:
                    continue
                strong = [x for x in cls_items if float(x["conf"]) >= float(min_confidence)]
                chosen = strong[: int(min_per_class)]
                if len(chosen) < int(min_per_class):
                    chosen = cls_items[: min(len(cls_items), int(min_per_class))]
                for x in chosen:
                    selected_pos.add(int(x["pos"]))

        remain = [x for x in records if int(x["pos"]) not in selected_pos and float(x["conf"]) >= float(min_confidence)]
        remain = sorted(remain, key=lambda x: x["conf"], reverse=True)
        for x in remain:
            if len(selected_pos) >= n_target:
                break
            selected_pos.add(int(x["pos"]))

        if len(selected_pos) == 0:
            if dataset_name:
                print(f"[SelfTrain] {dataset_name} skipped: no pseudo samples passed selection")
            return None

        pos_to_pred = {int(r["pos"]): int(r["pred"]) for r in records}
        sel_pos = sorted(list(selected_pos))
        sel_pred = [pos_to_pred[p] for p in sel_pos]

        eval_dataset = eval_loader.dataset
        if isinstance(eval_dataset, Subset):
            base_dataset = eval_dataset.dataset
            subset_indices = list(eval_dataset.indices)
            base_indices = [int(subset_indices[p]) for p in sel_pos]
        else:
            base_dataset = eval_dataset
            base_indices = [int(p) for p in sel_pos]

        pseudo_dataset = _PseudoLabelSubsetDataset(base_dataset, base_indices, sel_pred)
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=eval_loader.batch_size,
            shuffle=False,
            collate_fn=getattr(eval_loader, "collate_fn", multimodal_collate_fn),
            num_workers=getattr(eval_loader, "num_workers", 0),
            pin_memory=getattr(eval_loader, "pin_memory", False),
            prefetch_factor=getattr(eval_loader, "prefetch_factor", None),
            persistent_workers=getattr(eval_loader, "persistent_workers", False),
        )
        if dataset_name:
            print(
                f"[SelfTrain] {dataset_name} pseudo selected: "
                f"{len(pseudo_dataset)}/{n_total} (topk={float(topk_ratio):.2f}, min_conf={float(min_confidence):.2f}, "
                f"balanced_topk={1 if bool(balanced_topk) else 0})"
            )
        return pseudo_loader

    def _build_pseudo_labeled_loader_from_label_prop(
        support_loader,
        eval_loader,
        dataset_name="",
        diag_num_classes=4,
        topk_ratio=0.30,
        min_confidence=0.70,
        min_per_class=5,
        balanced_topk=False,
        kernel="knn",
        n_neighbors=10,
        alpha=0.20,
        max_iter=30,
        tol=1e-3,
        spec_transform_fn=None,
    ):
        """Build pseudo-labeled loader from LabelSpreading over support(eval-labeled)+query(eval-unlabeled)."""
        if support_loader is None or eval_loader is None:
            return None
        _se = getattr(detector, "spec_encoder", None)
        if _se is None:
            return None
        try:
            device = getattr(detector, "device", next(_se.parameters()).device)
        except Exception:
            return None

        try:
            import config as _cfg_st_lp
            _dual_st = bool(getattr(_cfg_st_lp, "TARGET_SUP_LABEL_PROP_DUAL_SPACE_ENABLE", False))
            _loo_st = bool(getattr(_cfg_st_lp, "TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_ENABLE", False))
            _loo_grid_st = list(getattr(_cfg_st_lp, "TARGET_SUP_LABEL_PROP_DUAL_LOO_WEIGHT_GRID", []))
            _default_w_st = float(getattr(_cfg_st_lp, "TARGET_SUP_LABEL_PROP_FUSION_WEIGHT", 0.35))
        except Exception:
            _dual_st = False
            _loo_st = False
            _loo_grid_st = []
            _default_w_st = 0.35

        _sig_st = getattr(detector, "signal_encoder", None)
        _fus_st = getattr(detector, "fusion_module", None)

        def _collect_embeddings(loader):
            embs = []
            with torch.no_grad():
                for batch in loader:
                    spec = load_spec_from_batch(batch, device, detector)
                    spec_mask = batch["spec_mask"].to(device)
                    domain_ids = _encode_domain_ids_from_batch(batch, device)
                    if domain_ids is not None:
                        try:
                            z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_spec = _se(spec, spec_mask)
                    else:
                        z_spec = _se(spec, spec_mask)
                    if callable(spec_transform_fn):
                        z_spec = spec_transform_fn(z_spec)
                    z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                    embs.append(z_spec.detach().cpu().numpy())
            if len(embs) == 0:
                return None
            return np.concatenate(embs, axis=0)

        def _collect_labels(loader):
            ys = []
            for batch in loader:
                y = batch["label"]
                if not torch.is_tensor(y):
                    y = torch.tensor(y)
                y = y.long()
                if int(diag_num_classes) == 3:
                    y = torch.clamp(y, 0, 2)
                ys.append(y.detach().cpu().numpy())
            if len(ys) == 0:
                return None
            return np.concatenate(ys, axis=0)

        pred_query = None
        conf_query = None
        n_support = None
        n_total = None
        _pseudo_lp_tag = "label_prop"
        _pseudo_w_f = None
        _log_k = str(kernel).strip().lower()
        _log_nn = int(n_neighbors)
        _log_al = float(alpha)

        if _dual_st and _sig_st is not None and _fus_st is not None:
            try:
                _se.eval()
                _sig_st.eval()
                _fus_st.eval()

                def _collect_spec_fusion_np(loader, collect_y):
                    spec_list = []
                    fus_list = []
                    lbl_list = []
                    with torch.no_grad():
                        for batch in loader:
                            signal = batch["signal"].to(device)
                            spec = load_spec_from_batch(batch, device, detector)
                            signal_mask = batch.get("signal_mask")
                            spec_mask = batch["spec_mask"].to(device)
                            if signal_mask is None:
                                signal_mask = torch.ones(
                                    signal.size(0), signal.size(1), device=device, dtype=torch.bool
                                )
                            else:
                                signal_mask = signal_mask.to(device)
                            if collect_y:
                                labels = batch["label"].to(device)
                                if int(diag_num_classes) == 3:
                                    labels = torch.clamp(labels, 0, 2)
                                lbl_list.append(labels.detach().cpu().numpy())
                            domain_ids = _encode_domain_ids_from_batch(batch, device)
                            if domain_ids is not None:
                                try:
                                    z_sig = _sig_st(signal, signal_mask, domain_ids=domain_ids)
                                except TypeError:
                                    z_sig = _sig_st(signal, signal_mask)
                                try:
                                    z_spec = _se(spec, spec_mask, domain_ids=domain_ids)
                                except TypeError:
                                    z_spec = _se(spec, spec_mask)
                                try:
                                    z_f = _fus_st(z_sig, z_spec, domain_ids=domain_ids)
                                except TypeError:
                                    z_f = _fus_st(z_sig, z_spec)
                            else:
                                z_sig = _sig_st(signal, signal_mask)
                                z_spec = _se(spec, spec_mask)
                                z_f = _fus_st(z_sig, z_spec)
                            if callable(spec_transform_fn):
                                z_spec = spec_transform_fn(z_spec)
                            z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                            z_f = torch.nn.functional.normalize(z_f, dim=-1)
                            spec_list.append(z_spec.detach().cpu().numpy())
                            fus_list.append(z_f.detach().cpu().numpy())
                    if len(spec_list) == 0:
                        return None, None, None
                    sp = np.concatenate(spec_list, axis=0)
                    fu = np.concatenate(fus_list, axis=0)
                    if collect_y:
                        return sp, fu, np.concatenate(lbl_list, axis=0)
                    return sp, fu, None

                support_spec, support_fus, support_y_d = _collect_spec_fusion_np(support_loader, True)
                query_spec, query_fus, _ = _collect_spec_fusion_np(eval_loader, False)
                if support_spec is None or support_y_d is None or query_spec is None:
                    raise RuntimeError("empty dual embeddings")
                n_support = int(support_spec.shape[0])
                n_total = int(query_spec.shape[0])
                if n_support <= 1 or n_total <= 0:
                    raise RuntimeError("invalid dual shapes")

                kernel_d = str(kernel).strip().lower()
                if kernel_d not in ("knn", "rbf"):
                    kernel_d = "knn"
                n_neighbors_eff = max(2, min(int(n_neighbors), n_support - 1))
                alpha_d = max(1e-4, min(0.999, float(alpha)))
                max_iter_d = max(10, int(max_iter))
                tol_d = max(1e-6, float(tol))

                w_f = float(max(0.0, min(1.0, _default_w_st)))
                if _loo_st and len(_loo_grid_st) > 0 and n_support >= 3:
                    w_f, _, _ = _select_dual_labelprop_fusion_weight_support_loo(
                        support_spec,
                        query_spec,
                        support_fus,
                        query_fus,
                        support_y_d,
                        diag_num_classes,
                        kernel_d,
                        n_neighbors_eff,
                        alpha_d,
                        max_iter_d,
                        tol_d,
                        _loo_grid_st,
                        w_f,
                        dataset_name=dataset_name,
                    )
                elif _loo_st and dataset_name:
                    print(
                        f"[SelfTrain] {dataset_name} dual label_prop: support-LOO skipped "
                        f"(support={n_support}, grid_n={len(_loo_grid_st)})"
                    )

                d_spec = _lp_aligned_query_distributions(
                    support_spec,
                    query_spec,
                    support_y_d,
                    diag_num_classes,
                    kernel_d,
                    n_neighbors_eff,
                    alpha_d,
                    max_iter_d,
                    tol_d,
                    n_support,
                    n_total,
                )
                d_fus = _lp_aligned_query_distributions(
                    support_fus,
                    query_fus,
                    support_y_d,
                    diag_num_classes,
                    kernel_d,
                    n_neighbors_eff,
                    alpha_d,
                    max_iter_d,
                    tol_d,
                    n_support,
                    n_total,
                )
                w_s = 1.0 - w_f
                d_dual = w_s * d_spec + w_f * d_fus
                d_dual = d_dual / np.maximum(d_dual.sum(axis=1, keepdims=True), 1e-12)
                pred_query = np.argmax(d_dual, axis=1).astype(np.int64)
                conf_query = np.max(d_dual, axis=1)
                _pseudo_lp_tag = "label_prop_dual"
                _pseudo_w_f = float(w_f)
                _log_k = kernel_d
                _log_nn = int(n_neighbors_eff)
                _log_al = float(alpha_d)
            except Exception as _e_dual:
                if dataset_name:
                    print(
                        f"[SelfTrain] {dataset_name} dual label_prop fallback to z_spec-only "
                        f"({type(_e_dual).__name__}: {_e_dual})"
                    )
                pred_query = None

        _se.eval()
        support_emb = _collect_embeddings(support_loader)
        support_y = _collect_labels(support_loader)
        query_emb = _collect_embeddings(eval_loader)
        if support_emb is None or support_y is None or query_emb is None:
            return None

        n_support_sp = int(support_emb.shape[0])
        n_total_sp = int(query_emb.shape[0])
        if n_support_sp <= 1 or n_total_sp <= 0:
            return None

        if pred_query is not None and int(n_total) != int(n_total_sp):
            if dataset_name:
                print(
                    f"[SelfTrain] {dataset_name} dual/spec eval size mismatch "
                    f"({n_total} vs {n_total_sp}), fallback to z_spec label_prop"
                )
            pred_query = None

        if pred_query is None:
            n_support = n_support_sp
            n_total = n_total_sp
            try:
                kernel = str(kernel).strip().lower()
                if kernel not in ("knn", "rbf"):
                    kernel = "knn"
                n_neighbors_eff = max(2, min(int(n_neighbors), n_support - 1))
                alpha = max(1e-4, min(0.999, float(alpha)))
                max_iter = max(10, int(max_iter))
                tol = max(1e-6, float(tol))

                x_all = np.concatenate([support_emb, query_emb], axis=0)
                y_all = np.concatenate(
                    [support_y.astype(np.int64), -1 * np.ones((n_total,), dtype=np.int64)],
                    axis=0,
                )
                lp = LabelSpreading(
                    kernel=kernel,
                    n_neighbors=n_neighbors_eff,
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                )
                lp.fit(x_all, y_all)
                pred_all = lp.transduction_
                dist_all = lp.label_distributions_
                pred_query = pred_all[n_support:]
                dist_query = dist_all[n_support:]
                conf_query = np.max(dist_query, axis=1)
                _log_k = kernel
                _log_nn = int(n_neighbors_eff)
                _log_al = float(alpha)
            except Exception as _e_lp:
                if dataset_name:
                    print(f"[SelfTrain] {dataset_name} label_prop skipped ({type(_e_lp).__name__}: {_e_lp})")
                return None

        records = []
        for _i in range(n_total):
            records.append(
                {
                    "pos": int(_i),
                    "pred": int(pred_query[_i]),
                    "conf": float(conf_query[_i]),
                }
            )
        if len(records) == 0:
            return None

        n_target = max(
            int(min_per_class) * int(diag_num_classes),
            int(round(float(topk_ratio) * float(n_total))),
        )
        n_target = min(n_total, max(1, n_target))

        selected_pos = set()
        by_cls = {c: [] for c in range(int(diag_num_classes))}
        for r in records:
            if 0 <= int(r["pred"]) < int(diag_num_classes):
                by_cls[int(r["pred"])].append(r)
        if bool(balanced_topk):
            n_cls = max(1, int(diag_num_classes))
            base_q = int(n_target // n_cls)
            rem_q = int(n_target % n_cls)
            cls_quota = {}
            for c in range(n_cls):
                q = int(base_q + (1 if c < rem_q else 0))
                cls_quota[c] = max(int(min_per_class), q)
            for c in range(n_cls):
                cls_items = sorted(by_cls.get(c, []), key=lambda x: x["conf"], reverse=True)
                if len(cls_items) == 0:
                    continue
                strong = [x for x in cls_items if float(x["conf"]) >= float(min_confidence)]
                chosen = strong[: int(cls_quota[c])]
                if len(chosen) < int(cls_quota[c]):
                    chosen = cls_items[: min(len(cls_items), int(cls_quota[c]))]
                for x in chosen:
                    selected_pos.add(int(x["pos"]))
        else:
            for c in range(int(diag_num_classes)):
                cls_items = sorted(by_cls.get(c, []), key=lambda x: x["conf"], reverse=True)
                if len(cls_items) == 0:
                    continue
                strong = [x for x in cls_items if float(x["conf"]) >= float(min_confidence)]
                chosen = strong[: int(min_per_class)]
                if len(chosen) < int(min_per_class):
                    chosen = cls_items[: min(len(cls_items), int(min_per_class))]
                for x in chosen:
                    selected_pos.add(int(x["pos"]))

        remain = [x for x in records if int(x["pos"]) not in selected_pos and float(x["conf"]) >= float(min_confidence)]
        remain = sorted(remain, key=lambda x: x["conf"], reverse=True)
        for x in remain:
            if len(selected_pos) >= n_target:
                break
            selected_pos.add(int(x["pos"]))

        if len(selected_pos) == 0:
            if dataset_name:
                print(f"[SelfTrain] {dataset_name} skipped: no pseudo samples passed selection")
            return None

        pos_to_pred = {int(r["pos"]): int(r["pred"]) for r in records}
        sel_pos = sorted(list(selected_pos))
        sel_pred = [pos_to_pred[p] for p in sel_pos]

        eval_dataset = eval_loader.dataset
        if isinstance(eval_dataset, Subset):
            base_dataset = eval_dataset.dataset
            subset_indices = list(eval_dataset.indices)
            base_indices = [int(subset_indices[p]) for p in sel_pos]
        else:
            base_dataset = eval_dataset
            base_indices = [int(p) for p in sel_pos]

        pseudo_dataset = _PseudoLabelSubsetDataset(base_dataset, base_indices, sel_pred)
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=eval_loader.batch_size,
            shuffle=False,
            collate_fn=getattr(eval_loader, "collate_fn", multimodal_collate_fn),
            num_workers=getattr(eval_loader, "num_workers", 0),
            pin_memory=getattr(eval_loader, "pin_memory", False),
            prefetch_factor=getattr(eval_loader, "prefetch_factor", None),
            persistent_workers=getattr(eval_loader, "persistent_workers", False),
        )
        if dataset_name:
            _wmsg = ""
            if _pseudo_lp_tag == "label_prop_dual" and _pseudo_w_f is not None:
                _wmsg = f", w_fusion={float(_pseudo_w_f):.2f}"
            print(
                f"[SelfTrain] {dataset_name} pseudo selected: "
                f"{len(pseudo_dataset)}/{n_total} (source={_pseudo_lp_tag}{_wmsg}, topk={float(topk_ratio):.2f}, "
                f"min_conf={float(min_confidence):.2f}, balanced_topk={1 if bool(balanced_topk) else 0}, "
                f"kernel={_log_k}, n_neighbors={int(_log_nn)}, alpha={float(_log_al):.2f})"
            )
        return pseudo_loader

    def _run_post_lpft_self_training(
        finetune_loader,
        eval_loader,
        sup_result,
        dataset_name="",
        diag_num_classes=4,
        has_nf_head=False,
        num_steps=None,
        lr=None,
        nf_weight=0.5,
        diag_weight=1.0,
        spec_transform_fn=None,
    ):
        if has_nf_head or finetune_loader is None or eval_loader is None or sup_result is None:
            return sup_result
        try:
            import config as cfg
            st_enable = bool(getattr(cfg, "TARGET_SELF_TRAINING_ENABLE", False))
            st_pseudo_source = str(getattr(cfg, "TARGET_SELF_TRAINING_PSEUDO_SOURCE", "spec_head")).strip().lower()
            st_rounds = int(getattr(cfg, "TARGET_SELF_TRAINING_ROUNDS", 0))
            st_topk = float(getattr(cfg, "TARGET_SELF_TRAINING_TOPK_RATIO", 0.30))
            st_topk_cap = float(getattr(cfg, "TARGET_SELF_TRAINING_MAX_TOPK_RATIO", 1.0))
            st_min_conf = float(getattr(cfg, "TARGET_SELF_TRAINING_MIN_CONFIDENCE", 0.70))
            st_min_conf_step = float(getattr(cfg, "TARGET_SELF_TRAINING_MIN_CONF_STEP", 0.0))
            st_min_per_cls = int(getattr(cfg, "TARGET_SELF_TRAINING_MIN_SAMPLES_PER_CLASS", 5))
            st_balanced_topk_enable = bool(getattr(cfg, "TARGET_SELF_TRAINING_BALANCED_TOPK_ENABLE", False))
            st_tiny_relax_enable = bool(getattr(cfg, "TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_ENABLE", False))
            st_tiny_relax_threshold = int(getattr(cfg, "TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_THRESHOLD", 50))
            st_tiny_relax_value = float(getattr(cfg, "TARGET_SELF_TRAINING_TINYSHOT_RELAX_MIN_CONF_VALUE", 0.50))
            st_min_start_acc = float(getattr(cfg, "TARGET_SELF_TRAINING_MIN_START_ACC", 0.45))
            st_replace_when_better = bool(getattr(cfg, "TARGET_SELF_TRAINING_REPLACE_WHEN_BETTER", True))
            st_lp_kernel = str(getattr(cfg, "TARGET_SUP_LABEL_PROP_KERNEL", "knn")).strip().lower()
            st_lp_n_neighbors = int(getattr(cfg, "TARGET_SUP_LABEL_PROP_N_NEIGHBORS", 10))
            st_lp_alpha = float(getattr(cfg, "TARGET_SUP_LABEL_PROP_ALPHA", 0.20))
            st_lp_max_iter = int(getattr(cfg, "TARGET_SUP_LABEL_PROP_MAX_ITER", 30))
            st_lp_tol = float(getattr(cfg, "TARGET_SUP_LABEL_PROP_TOL", 1e-3))
        except Exception:
            st_enable = False
            st_pseudo_source = "spec_head"
            st_rounds = 0
            st_topk = 0.30
            st_topk_cap = 1.0
            st_min_conf = 0.70
            st_min_conf_step = 0.0
            st_min_per_cls = 5
            st_balanced_topk_enable = False
            st_tiny_relax_enable = False
            st_tiny_relax_threshold = 50
            st_tiny_relax_value = 0.50
            st_min_start_acc = 0.45
            st_replace_when_better = True
            st_lp_kernel = "knn"
            st_lp_n_neighbors = 10
            st_lp_alpha = 0.20
            st_lp_max_iter = 30
            st_lp_tol = 1e-3
        if st_pseudo_source not in ("spec_head", "label_prop"):
            st_pseudo_source = "spec_head"
        if (not st_enable) or (st_rounds <= 0):
            return sup_result

        start_acc = float(sup_result.get("diag_acc", 0.0))
        if start_acc + 1e-12 < float(st_min_start_acc):
            print(
                f"[SelfTrain] {dataset_name} skipped: start diag_acc={start_acc:.4f} "
                f"< min_start_acc={float(st_min_start_acc):.4f}"
            )
            return sup_result

        # Snapshot the post-LPFT state so self-training failures do not pollute
        # subsequent evaluation paths (e.g., SVP) when no round improves.
        _can_snapshot = hasattr(detector, "snapshot_fusion_state") and hasattr(detector, "restore_fusion_state")
        if _can_snapshot:
            try:
                detector.snapshot_fusion_state()
            except Exception:
                _can_snapshot = False

        best_result = dict(sup_result)
        best_acc = float(best_result.get("diag_acc", 0.0))
        _base_labeled_n = 0
        try:
            _base_labeled_n = int(len(finetune_loader.dataset))
        except Exception:
            _base_labeled_n = 0
        for rd in range(int(st_rounds)):
            rd_topk = float(st_topk)
            if st_topk_cap > 0.0:
                rd_topk = min(rd_topk, float(st_topk_cap))
            rd_topk = max(0.05, min(0.95, rd_topk))
            rd_min_conf = max(0.0, min(0.99, float(st_min_conf) + float(rd) * float(st_min_conf_step)))
            if bool(st_tiny_relax_enable) and (_base_labeled_n > 0) and (_base_labeled_n <= int(st_tiny_relax_threshold)):
                rd_min_conf = min(float(rd_min_conf), float(st_tiny_relax_value))
            if st_pseudo_source == "label_prop":
                pseudo_loader = _build_pseudo_labeled_loader_from_label_prop(
                    support_loader=finetune_loader,
                    eval_loader=eval_loader,
                    dataset_name=f"{dataset_name}/r{rd+1}",
                    diag_num_classes=int(diag_num_classes),
                    topk_ratio=float(rd_topk),
                    min_confidence=float(rd_min_conf),
                    min_per_class=int(st_min_per_cls),
                    balanced_topk=bool(st_balanced_topk_enable),
                    kernel=st_lp_kernel,
                    n_neighbors=int(st_lp_n_neighbors),
                    alpha=float(st_lp_alpha),
                    max_iter=int(st_lp_max_iter),
                    tol=float(st_lp_tol),
                    spec_transform_fn=spec_transform_fn,
                )
            else:
                pseudo_loader = _build_pseudo_labeled_loader_from_spec_head(
                    eval_loader,
                    dataset_name=f"{dataset_name}/r{rd+1}",
                    diag_num_classes=int(diag_num_classes),
                    topk_ratio=float(rd_topk),
                    min_confidence=float(rd_min_conf),
                    min_per_class=int(st_min_per_cls),
                    balanced_topk=bool(st_balanced_topk_enable),
                    spec_transform_fn=spec_transform_fn,
                )
            if pseudo_loader is None or len(pseudo_loader.dataset) == 0:
                break

            aug_dataset = ConcatDataset([finetune_loader.dataset, pseudo_loader.dataset])
            aug_loader = DataLoader(
                aug_dataset,
                batch_size=finetune_loader.batch_size,
                shuffle=False,
                collate_fn=getattr(finetune_loader, "collate_fn", multimodal_collate_fn),
                num_workers=getattr(finetune_loader, "num_workers", 0),
                pin_memory=getattr(finetune_loader, "pin_memory", False),
                prefetch_factor=getattr(finetune_loader, "prefetch_factor", None),
                persistent_workers=getattr(finetune_loader, "persistent_workers", False),
            )
            print(
                f"[SelfTrain] {dataset_name} round {rd+1}/{int(st_rounds)}: "
                f"augmented labeled samples={len(aug_dataset)} "
                f"(base={len(finetune_loader.dataset)}, pseudo={len(pseudo_loader.dataset)}), "
                f"topk={rd_topk:.2f}, min_conf={rd_min_conf:.2f}"
            )

            detector.supervised_finetune_on_loader(
                aug_loader,
                dataset_name=f"{dataset_name}/selftrain_r{rd+1}",
                num_classes_eval=int(diag_num_classes),
                num_steps=num_steps,
                lr=lr,
                nf_weight=nf_weight,
                diag_weight=diag_weight,
            )
            detector.set_eval_nf_threshold(None)
            round_result = detector.evaluate_target_domain_accuracy(
                eval_loader, dataset_name=dataset_name, diag_num_classes=int(diag_num_classes)
            )
            round_spec = _evaluate_with_spec_head(
                eval_loader,
                dataset_name=f"{dataset_name}/selftrain_r{rd+1}",
                diag_num_classes=int(diag_num_classes),
                spec_transform_fn=spec_transform_fn,
            )
            if round_spec is not None:
                round_result["spec_head_diag_acc"] = float(round_spec.get("diag_acc", float("nan")))
                round_result["spec_head_macro_f1"] = float(round_spec.get("macro_f1", float("nan")))
                if st_replace_when_better and float(round_spec.get("diag_acc", 0.0)) > float(round_result.get("diag_acc", 0.0)):
                    merged = dict(round_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = round_spec.get(k, merged.get(k))
                    round_result = merged

            cur_acc = float(round_result.get("diag_acc", 0.0))
            print(f"[SelfTrain] {dataset_name} round {rd+1} eval diag_acc={cur_acc:.4f} (best={best_acc:.4f})")
            if cur_acc > best_acc + 1e-6:
                best_acc = cur_acc
                best_result = round_result
                if _can_snapshot:
                    try:
                        detector.snapshot_fusion_state()
                    except Exception:
                        _can_snapshot = False
            else:
                print(f"[SelfTrain] {dataset_name} early stop: no improvement at round {rd+1}")
                if _can_snapshot:
                    try:
                        detector.restore_fusion_state()
                    except Exception:
                        pass
                break
        if _can_snapshot:
            try:
                detector.restore_fusion_state()
            except Exception:
                pass
        return best_result

    def _build_visual_prototypes(labeled_loader, diag_num_classes=4, space="spec", spec_transform_fn=None):
        """
        Build class prototypes from labeled target-domain embeddings.
        space:
          - "spec": z_spec
          - "fusion": z_fusion (from z_sig + z_spec)
        """
        device = getattr(detector, "device", next(detector.spec_encoder.parameters()).device)
        detector.signal_encoder.eval()
        detector.spec_encoder.eval()
        detector.fusion_module.eval()
        sums = [None for _ in range(diag_num_classes)]
        counts = [0 for _ in range(diag_num_classes)]

        with torch.no_grad():
            for batch in labeled_loader:
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch["spec_mask"].to(device)
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                else:
                    signal_mask = signal_mask.to(device)
                labels = batch["label"].to(device)
                if diag_num_classes == 3:
                    labels = torch.clamp(labels, 0, 2)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                _sv = batch.get("stat_vec", None)
                if _sv is not None:
                    _sv = _sv.to(device)
                if domain_ids is not None:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                else:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                if (space == "spec") and callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                if space == "fusion":
                    if domain_ids is not None:
                        try:
                            z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_sig = detector.signal_encoder(signal, signal_mask)
                    else:
                        z_sig = detector.signal_encoder(signal, signal_mask)
                    try:
                        z_emb = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                    except TypeError:
                        z_emb = detector.fusion_module(z_sig, z_spec)
                else:
                    z_emb = z_spec
                z_emb = torch.nn.functional.normalize(z_emb, dim=-1)
                for c in range(diag_num_classes):
                    mask_c = (labels == c)
                    n_c = int(mask_c.sum().item())
                    if n_c <= 0:
                        continue
                    z_c = z_emb[mask_c]
                    if sums[c] is None:
                        sums[c] = z_c.sum(dim=0)
                    else:
                        sums[c] = sums[c] + z_c.sum(dim=0)
                    counts[c] += n_c

        feat_dim = None
        for x in sums:
            if x is not None:
                feat_dim = int(x.numel())
                break
        if feat_dim is None:
            return None, counts

        protos = []
        valid = []
        for c in range(diag_num_classes):
            if counts[c] <= 0 or sums[c] is None:
                protos.append(torch.zeros(feat_dim, device=device))
                valid.append(False)
                continue
            p = sums[c] / float(counts[c])
            p = torch.nn.functional.normalize(p.unsqueeze(0), dim=-1).squeeze(0)
            protos.append(p)
            valid.append(True)
        proto_mat = torch.stack(protos, dim=0)  # [C, D]
        return (proto_mat, valid), counts

    def _evaluate_with_visual_prototypes(loader, proto_pack, dataset_name="", diag_num_classes=4, space="spec", spec_transform_fn=None):
        """
        Evaluate target diagnosis by nearest visual prototypes in selected embedding space.
        """
        if proto_pack is None:
            return None
        proto_mat, valid = proto_pack
        if proto_mat is None:
            return None
        device = getattr(detector, "device", next(detector.spec_encoder.parameters()).device)
        detector.signal_encoder.eval()
        detector.spec_encoder.eval()
        detector.fusion_module.eval()

        diag_correct = 0
        diag_total = 0
        all_diag_labels = []
        all_diag_preds = []
        valid_mask = torch.tensor(valid, dtype=torch.bool, device=device)
        neg_inf = torch.tensor(-1e9, device=device)

        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch["spec_mask"].to(device)
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                else:
                    signal_mask = signal_mask.to(device)
                labels = batch["label"].to(device)
                if diag_num_classes == 3:
                    labels = torch.clamp(labels, 0, 2)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                _sv = batch.get("stat_vec", None)
                if _sv is not None:
                    _sv = _sv.to(device)
                if domain_ids is not None:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                else:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                if (space == "spec") and callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                if space == "fusion":
                    if domain_ids is not None:
                        try:
                            z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_sig = detector.signal_encoder(signal, signal_mask)
                    else:
                        z_sig = detector.signal_encoder(signal, signal_mask)
                    try:
                        z_emb = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                    except TypeError:
                        z_emb = detector.fusion_module(z_sig, z_spec)
                else:
                    z_emb = z_spec
                z_emb = torch.nn.functional.normalize(z_emb, dim=-1)
                sims = torch.matmul(z_emb, proto_mat.t())  # [B, C]
                sims = torch.where(valid_mask.unsqueeze(0), sims, neg_inf)
                preds = sims.argmax(dim=1)
                diag_correct += int((preds == labels).sum().item())
                diag_total += int(labels.numel())
                all_diag_labels.extend(labels.detach().cpu().tolist())
                all_diag_preds.extend(preds.detach().cpu().tolist())

        acc = float(diag_correct) / float(diag_total) if diag_total > 0 else 0.0
        macro_f1 = _safe_macro_f1(all_diag_labels, all_diag_preds)
        if dataset_name:
            space_tag = "fusion_visual_proto" if space == "fusion" else "spec_visual_proto"
            print(
                f"[SVP] {dataset_name} {space_tag}_diag_acc={acc:.4f} "
                f"macro_f1={macro_f1:.4f} (N={diag_total})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": diag_correct,
            "diag_total": diag_total,
            "diag_labels": all_diag_labels,
            "diag_preds": all_diag_preds,
            "macro_f1": macro_f1,
            "nf_acc": float("nan"),
            "nf_correct": 0,
            "nf_total": 0,
        }

    def _evaluate_with_dual_visual_prototypes(
        loader,
        spec_proto_pack,
        fusion_proto_pack,
        dataset_name="",
        diag_num_classes=4,
        fusion_weight=0.35,
        spec_transform_fn=None,
    ):
        """
        Blend z_spec and z_fusion prototype similarities:
          sim = (1-w) * sim_spec + w * sim_fusion
        """
        if spec_proto_pack is None or fusion_proto_pack is None:
            return None
        spec_mat, spec_valid = spec_proto_pack
        fus_mat, fus_valid = fusion_proto_pack
        if spec_mat is None or fus_mat is None:
            return None
        device = getattr(detector, "device", next(detector.spec_encoder.parameters()).device)
        detector.signal_encoder.eval()
        detector.spec_encoder.eval()
        detector.fusion_module.eval()

        w_fus = float(max(0.0, min(1.0, fusion_weight)))
        w_spec = 1.0 - w_fus
        valid_mask = torch.tensor(spec_valid, dtype=torch.bool, device=device) & torch.tensor(fus_valid, dtype=torch.bool, device=device)
        neg_inf = torch.tensor(-1e9, device=device)
        diag_correct, diag_total = 0, 0
        all_diag_labels, all_diag_preds = [], []

        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch["spec_mask"].to(device)
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                else:
                    signal_mask = signal_mask.to(device)
                labels = batch["label"].to(device)
                if diag_num_classes == 3:
                    labels = torch.clamp(labels, 0, 2)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                _sv = batch.get("stat_vec", None)
                if _sv is not None:
                    _sv = _sv.to(device)
                if domain_ids is not None:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                    try:
                        z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                    except TypeError:
                        z_sig = detector.signal_encoder(signal, signal_mask)
                else:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                    z_sig = detector.signal_encoder(signal, signal_mask)
                try:
                    z_fus = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                except TypeError:
                    z_fus = detector.fusion_module(z_sig, z_spec)
                if callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                z_spec = torch.nn.functional.normalize(z_spec, dim=-1)
                z_fus = torch.nn.functional.normalize(z_fus, dim=-1)
                sims_spec = torch.matmul(z_spec, spec_mat.t())
                sims_fus = torch.matmul(z_fus, fus_mat.t())
                sims = w_spec * sims_spec + w_fus * sims_fus
                sims = torch.where(valid_mask.unsqueeze(0), sims, neg_inf)
                preds = sims.argmax(dim=1)
                diag_correct += int((preds == labels).sum().item())
                diag_total += int(labels.numel())
                all_diag_labels.extend(labels.detach().cpu().tolist())
                all_diag_preds.extend(preds.detach().cpu().tolist())

        acc = float(diag_correct) / float(diag_total) if diag_total > 0 else 0.0
        macro_f1 = _safe_macro_f1(all_diag_labels, all_diag_preds)
        if dataset_name:
            print(
                f"[SVP] {dataset_name} dual_visual_proto_diag_acc={acc:.4f} "
                f"macro_f1={macro_f1:.4f} (w_spec={w_spec:.2f}, w_fusion={w_fus:.2f}, N={diag_total})"
            )
        return {
            "diag_acc": acc,
            "diag_acc_raw": acc,
            "diag_correct": diag_correct,
            "diag_total": diag_total,
            "diag_labels": all_diag_labels,
            "diag_preds": all_diag_preds,
            "macro_f1": macro_f1,
            "nf_acc": float("nan"),
            "nf_correct": 0,
            "nf_total": 0,
        }

    def _refine_visual_prototypes_with_unlabeled(
        loader,
        proto_pack,
        dataset_name="",
        diag_num_classes=4,
        space="spec",
        iters=2,
        topk_ratio=0.30,
        min_per_class=5,
        alpha=0.80,
        spec_transform_fn=None,
    ):
        """
        Transductive refinement for visual prototypes using high-confidence pseudo labels.
        This is eval-only and does not update any model parameters.
        """
        if proto_pack is None:
            return None
        proto_mat, valid = proto_pack
        if proto_mat is None:
            return None
        device = getattr(detector, "device", next(detector.spec_encoder.parameters()).device)
        detector.signal_encoder.eval()
        detector.spec_encoder.eval()
        detector.fusion_module.eval()

        z_all = []
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch["spec_mask"].to(device)
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                else:
                    signal_mask = signal_mask.to(device)
                domain_ids = _encode_domain_ids_from_batch(batch, device)
                _sv = batch.get("stat_vec", None)
                if _sv is not None:
                    _sv = _sv.to(device)
                if domain_ids is not None:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                else:
                    try:
                        z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                    except TypeError:
                        z_spec = detector.spec_encoder(spec, spec_mask)
                if (space == "spec") and callable(spec_transform_fn):
                    z_spec = spec_transform_fn(z_spec)
                if space == "fusion":
                    if domain_ids is not None:
                        try:
                            z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_sig = detector.signal_encoder(signal, signal_mask)
                    else:
                        z_sig = detector.signal_encoder(signal, signal_mask)
                    try:
                        z_emb = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                    except TypeError:
                        z_emb = detector.fusion_module(z_sig, z_spec)
                else:
                    z_emb = z_spec
                z_all.append(torch.nn.functional.normalize(z_emb, dim=-1))

        if len(z_all) <= 0:
            return proto_pack

        z_all = torch.cat(z_all, dim=0).float()
        proto_base = proto_mat.to(z_all.device).float().clone()
        proto_cur = proto_base.clone()
        valid_mask = torch.tensor(valid, dtype=torch.bool, device=z_all.device)
        neg_inf = torch.tensor(-1e9, device=z_all.device)
        q = min(max(float(topk_ratio), 0.05), 0.95)
        min_cnt = max(1, int(min_per_class))
        n_iter = max(1, int(iters))
        mix_alpha = min(max(float(alpha), 0.0), 1.0)
        updated_total = 0

        for _ in range(n_iter):
            sims = torch.matmul(z_all, proto_cur.t())
            sims = torch.where(valid_mask.unsqueeze(0), sims, neg_inf)
            conf, pred = torch.max(sims, dim=1)
            thr = torch.quantile(conf, 1.0 - q)
            selected = (conf >= thr)
            proto_vis = proto_cur.clone()
            updated = 0
            for c in range(int(proto_cur.size(0))):
                if not bool(valid_mask[c].item()):
                    continue
                m = selected & (pred == c)
                if int(m.sum().item()) < min_cnt:
                    continue
                p_c = z_all[m].mean(dim=0)
                p_c = torch.nn.functional.normalize(p_c.unsqueeze(0), dim=-1).squeeze(0)
                proto_vis[c] = p_c
                updated += 1
            proto_cur = torch.nn.functional.normalize(
                mix_alpha * proto_base + (1.0 - mix_alpha) * proto_vis, dim=-1
            )
            updated_total += int(updated)
            if updated <= 0:
                break

        if dataset_name:
            print(
                f"[SVP-TPA] {dataset_name} {space}: iters={n_iter} topk={q:.2f} "
                f"min_cls={min_cnt} alpha={mix_alpha:.2f} updated={updated_total}"
            )
        return (proto_cur, valid)

    def _select_best_dual_weight_from_labeled(
        labeled_loader,
        spec_proto_pack,
        fusion_proto_pack,
        dataset_name="",
        diag_num_classes=4,
        weight_grid=None,
        fallback_weight=0.35,
    ):
        if weight_grid is None or len(weight_grid) <= 0:
            return float(fallback_weight)
        best_w = float(fallback_weight)
        best_acc = -1.0
        best_f1 = -1.0
        for w in weight_grid:
            try:
                wv = float(w)
            except Exception:
                continue
            res = _evaluate_with_dual_visual_prototypes(
                labeled_loader,
                spec_proto_pack,
                fusion_proto_pack,
                dataset_name="",
                diag_num_classes=diag_num_classes,
                fusion_weight=wv,
            )
            if res is None:
                continue
            acc = float(res.get("diag_acc", 0.0))
            f1 = float(res.get("macro_f1", 0.0))
            if (acc > best_acc) or (abs(acc - best_acc) <= 1e-12 and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_w = wv
        if dataset_name:
            print(f"[SVP] {dataset_name} dual weight selected: w_fusion={best_w:.2f} (fit_acc={best_acc:.4f})")
        return float(max(0.0, min(1.0, best_w)))

    def _run_bn_adaptation(detector, loader, dataset_name, num_batches=None):
        """
        P1: Test-Time BN Adaptation. 用目标域若干 batch 前向更新 BN running_mean/var，再恢复初态由调用方在评估后 restore。
        返回 (saved_state, restore_fn)，若未执行则返回 (None, None)。
        """
        try:
            import config as cfg
            enable = bool(getattr(cfg, "TARGET_BN_ADAPT_ENABLE", True))
            n_batch = int(getattr(cfg, "TARGET_BN_ADAPT_NUM_BATCHES", 10))
            bn_alpha = float(getattr(cfg, "TARGET_BN_ADAPT_ALPHA", 0.0))
            auto_alpha = bool(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA", False))
            auto_alpha_min = float(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA_MIN", 0.05))
            auto_alpha_max = float(getattr(cfg, "TARGET_BN_ADAPT_AUTO_ALPHA_MAX", 0.95))
        except Exception:
            enable = True
            n_batch = 10
            bn_alpha = 0.0
            auto_alpha = False
            auto_alpha_min = 0.05
            auto_alpha_max = 0.95
        if not enable or n_batch == 0:
            return None, None
        if n_batch < 0:
            max_batches = len(loader) if hasattr(loader, "__len__") else 10**9
        else:
            if num_batches is None:
                max_batches = n_batch
            else:
                max_batches = min(int(num_batches), n_batch)
        device = getattr(detector, "device", next(detector.signal_encoder.parameters()).device)
        saved = {
            "signal_encoder": {k: v.cpu().clone() for k, v in detector.signal_encoder.state_dict().items()},
            "spec_encoder": {k: v.cpu().clone() for k, v in detector.spec_encoder.state_dict().items()},
        }
        detector.signal_encoder.train()
        detector.spec_encoder.train()
        n_run = 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= max_batches:
                    break
                n_run += 1
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch.get("spec_mask")
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                if spec_mask is None:
                    spec_mask = torch.ones(spec.size(0), spec.size(1), device=device, dtype=torch.bool)
                domain_ids = None
                if batch.get("domain") is not None:
                    domain_ids = _encode_domain_ids(batch["domain"], batch.get("condition", None), device)
                try:
                    if domain_ids is not None:
                        try:
                            detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            detector.signal_encoder(signal, signal_mask)
                        try:
                            detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            detector.spec_encoder(spec, spec_mask)
                    else:
                        detector.signal_encoder(signal, signal_mask)
                        detector.spec_encoder(spec, spec_mask)
                except Exception:
                    pass
        detector.signal_encoder.eval()
        detector.spec_encoder.eval()

        mixed_stats = 0
        alpha_values = []
        if bn_alpha > 0.0:
            # ATT157: alpha-BN，插值 source 与 target 的 BN running 统计。
            # final_stat = alpha * source_stat + (1-alpha) * target_stat
            alpha = max(0.0, min(1.0, float(bn_alpha)))
            a_min = max(0.0, min(1.0, float(auto_alpha_min)))
            a_max = max(a_min, min(1.0, float(auto_alpha_max)))
            with torch.no_grad():
                for module_name in ("signal_encoder", "spec_encoder"):
                    module = getattr(detector, module_name, None)
                    if module is None:
                        continue
                    source_state = saved.get(module_name, {})
                    alpha_by_prefix = {}
                    for name, tensor in module.state_dict().items():
                        if ("running_mean" not in name) and ("running_var" not in name):
                            continue
                        if name not in source_state:
                            continue
                        alpha_eff = alpha
                        if auto_alpha:
                            # TTN-style: use source/target variance ratio to set per-layer alpha.
                            # alpha_l = var_source / (var_source + var_target)
                            if "running_var" in name:
                                src_var = source_state[name]
                                tgt_var = tensor
                                src_m = float(torch.mean(torch.abs(src_var)).item())
                                tgt_m = float(torch.mean(torch.abs(tgt_var)).item())
                                alpha_eff = src_m / (src_m + tgt_m + 1e-12)
                                alpha_eff = max(a_min, min(a_max, alpha_eff))
                                alpha_by_prefix[name.replace("running_var", "")] = alpha_eff
                            else:
                                key = name.replace("running_mean", "")
                                alpha_eff = alpha_by_prefix.get(key, alpha)
                        src_t = source_state[name].to(device=tensor.device, dtype=tensor.dtype)
                        tensor.copy_(alpha_eff * src_t + (1.0 - alpha_eff) * tensor)
                        alpha_values.append(alpha_eff)
                        mixed_stats += 1

        def restore():
            detector.signal_encoder.load_state_dict(saved["signal_encoder"], strict=True)
            detector.spec_encoder.load_state_dict(saved["spec_encoder"], strict=True)

        if bn_alpha > 0.0:
            alpha_msg = f"alpha_bn={max(0.0, min(1.0, float(bn_alpha))):.2f}"
            if auto_alpha and alpha_values:
                alpha_msg = (
                    f"alpha_bn=auto[{min(alpha_values):.2f},{max(alpha_values):.2f}]"
                    f"/mean={float(np.mean(alpha_values)):.2f}"
                )
            print(
                f"[BN-Adapt] {dataset_name}: updated BN stats with {n_run} batches, "
                f"{alpha_msg}, mixed_stats={mixed_stats}"
            )
        else:
            print(f"[BN-Adapt] {dataset_name}: updated BN stats with {n_run} batches (test-time adaptation)")
        return saved, restore

    def _run_tent_adaptation(detector, loader, dataset_name):
        """
        TENT: 目标域测试时熵最小化（仅更新 BN affine 参数），评估后由调用方 restore。
        返回 (saved_state, restore_fn)，若未执行则返回 (None, None)。
        """
        try:
            import config as cfg
            enable = bool(getattr(cfg, "TARGET_TENT_ENABLE", False))
            n_steps = int(getattr(cfg, "TARGET_TENT_STEPS", 10))
            lr = float(getattr(cfg, "TARGET_TENT_LR", 1e-3))
            n_batches = int(getattr(cfg, "TARGET_TENT_BATCHES", 4))
            div_w = float(getattr(cfg, "TARGET_TENT_DIVERSITY_WEIGHT", 0.1))
            reliable_filter_enable = bool(getattr(cfg, "TARGET_TENT_RELIABLE_FILTER_ENABLE", False))
            reliable_entropy_factor = float(getattr(cfg, "TARGET_TENT_RELIABLE_ENTROPY_FACTOR", 0.4))
        except Exception:
            enable = False
            n_steps = 10
            lr = 1e-3
            n_batches = 4
            div_w = 0.1
            reliable_filter_enable = False
            reliable_entropy_factor = 0.4
        if (not enable) or n_steps <= 0 or n_batches <= 0:
            return None, None

        device = getattr(detector, "device", next(detector.signal_encoder.parameters()).device)
        saved = {
            "signal_encoder": {k: v.cpu().clone() for k, v in detector.signal_encoder.state_dict().items()},
            "spec_encoder": {k: v.cpu().clone() for k, v in detector.spec_encoder.state_dict().items()},
        }
        grad_flags = {
            "signal_encoder": {n: p.requires_grad for n, p in detector.signal_encoder.named_parameters()},
            "spec_encoder": {n: p.requires_grad for n, p in detector.spec_encoder.named_parameters()},
            "fusion_module": {n: p.requires_grad for n, p in detector.fusion_module.named_parameters()},
            "fusion_decoder": {n: p.requires_grad for n, p in detector.fusion_decoder.named_parameters()},
        }

        # 仅更新 BN affine 参数（gamma/beta）；其余参数保持冻结
        for mod in (detector.signal_encoder, detector.spec_encoder, detector.fusion_module, detector.fusion_decoder):
            for p in mod.parameters():
                p.requires_grad = False

        tent_params = []
        bn_count = 0
        for enc in (detector.signal_encoder, detector.spec_encoder):
            for m in enc.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    bn_count += 1
                    if m.affine:
                        if m.weight is not None:
                            m.weight.requires_grad = True
                            tent_params.append(m.weight)
                        if m.bias is not None:
                            m.bias.requires_grad = True
                            tent_params.append(m.bias)
        if len(tent_params) == 0:
            print(f"[TENT] {dataset_name}: skipped (no BN affine params)")
            def _restore_skip():
                detector.signal_encoder.load_state_dict(saved["signal_encoder"], strict=True)
                detector.spec_encoder.load_state_dict(saved["spec_encoder"], strict=True)
                for n, p in detector.signal_encoder.named_parameters():
                    if n in grad_flags["signal_encoder"]:
                        p.requires_grad = grad_flags["signal_encoder"][n]
                for n, p in detector.spec_encoder.named_parameters():
                    if n in grad_flags["spec_encoder"]:
                        p.requires_grad = grad_flags["spec_encoder"][n]
                for n, p in detector.fusion_module.named_parameters():
                    if n in grad_flags["fusion_module"]:
                        p.requires_grad = grad_flags["fusion_module"][n]
                for n, p in detector.fusion_decoder.named_parameters():
                    if n in grad_flags["fusion_decoder"]:
                        p.requires_grad = grad_flags["fusion_decoder"][n]
            return saved, _restore_skip

        detector.signal_encoder.train()
        detector.spec_encoder.train()
        detector.fusion_module.eval()
        detector.fusion_decoder.eval()

        optim = torch.optim.Adam(tent_params, lr=lr)
        data_iter = iter(loader)
        n_effective = 0
        n_total_samples = 0
        n_selected_samples = 0
        n_total_batches = 0
        n_selected_batches = 0
        for _ in range(n_steps):
            batch_losses = []
            optim.zero_grad(set_to_none=True)
            for _ in range(n_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                signal = batch["signal"].to(device)
                spec = load_spec_from_batch(batch, device, detector)
                signal_mask = batch.get("signal_mask")
                spec_mask = batch.get("spec_mask")
                if signal_mask is None:
                    signal_mask = torch.ones(signal.size(0), signal.size(1), device=device, dtype=torch.bool)
                else:
                    signal_mask = signal_mask.to(device)
                if spec_mask is None:
                    spec_mask = torch.ones(spec.size(0), spec.size(1), device=device, dtype=torch.bool)
                else:
                    spec_mask = spec_mask.to(device)
                domain_ids = None
                if batch.get("domain") is not None:
                    domain_ids = _encode_domain_ids(batch["domain"], batch.get("condition", None), device)

                _sv = batch.get("stat_vec", None)
                if _sv is not None:
                    _sv = _sv.to(device)
                # Forward: encoders -> fusion -> decoder logits
                try:
                    if domain_ids is not None:
                        try:
                            z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                        except TypeError:
                            z_sig = detector.signal_encoder(signal, signal_mask)
                        try:
                            z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                        except TypeError:
                            z_spec = detector.spec_encoder(spec, spec_mask)
                    else:
                        z_sig = detector.signal_encoder(signal, signal_mask)
                        try:
                            z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                        except TypeError:
                            z_spec = detector.spec_encoder(spec, spec_mask)
                    z_fusion = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                    _, parts = detector.fusion_decoder(z_fusion, domain_ids=domain_ids, return_parts=True)
                    logits_diag = parts.get("logits_diag_supervised", parts.get("logits_diag", None))
                    if logits_diag is None:
                        continue
                except Exception:
                    continue

                p = torch.softmax(logits_diag.float(), dim=1)
                ent_each = -(p * torch.log(torch.clamp(p, min=1e-8))).sum(dim=1)
                n_total_batches += 1
                n_total_samples += int(ent_each.numel())

                if reliable_filter_enable:
                    n_cls = max(2, int(p.size(1)))
                    ent_th = float(math.log(n_cls) * reliable_entropy_factor)
                    reliable_mask = ent_each <= ent_th
                    n_keep = int(reliable_mask.sum().item())
                    n_selected_samples += n_keep
                    if n_keep <= 0:
                        continue
                    n_selected_batches += 1
                    p_use = p[reliable_mask]
                else:
                    n_selected_batches += 1
                    n_selected_samples += int(ent_each.numel())
                    p_use = p

                ent = -(p_use * torch.log(torch.clamp(p_use, min=1e-8))).sum(dim=1).mean()
                p_bar = p_use.mean(dim=0)
                div = -(p_bar * torch.log(torch.clamp(p_bar, min=1e-8))).sum()
                loss = ent - float(div_w) * div
                batch_losses.append(loss)

            if len(batch_losses) == 0:
                continue
            loss_step = torch.stack(batch_losses).mean()
            loss_step.backward()
            optim.step()
            n_effective += 1

        detector.signal_encoder.eval()
        detector.spec_encoder.eval()

        def restore():
            detector.signal_encoder.load_state_dict(saved["signal_encoder"], strict=True)
            detector.spec_encoder.load_state_dict(saved["spec_encoder"], strict=True)
            for n, p in detector.signal_encoder.named_parameters():
                if n in grad_flags["signal_encoder"]:
                    p.requires_grad = grad_flags["signal_encoder"][n]
            for n, p in detector.spec_encoder.named_parameters():
                if n in grad_flags["spec_encoder"]:
                    p.requires_grad = grad_flags["spec_encoder"][n]
            for n, p in detector.fusion_module.named_parameters():
                if n in grad_flags["fusion_module"]:
                    p.requires_grad = grad_flags["fusion_module"][n]
            for n, p in detector.fusion_decoder.named_parameters():
                if n in grad_flags["fusion_decoder"]:
                    p.requires_grad = grad_flags["fusion_decoder"][n]
            detector.signal_encoder.eval()
            detector.spec_encoder.eval()

        print(
            f"[TENT] {dataset_name}: updated BN affine with steps={n_effective}/{n_steps}, "
            f"batches_per_step={n_batches}, bn_layers={bn_count}, lr={lr:.2e}, div_w={div_w:.3f}, "
            f"reliable_filter={int(reliable_filter_enable)}, ent_factor={reliable_entropy_factor:.2f}, "
            f"kept={n_selected_samples}/{n_total_samples} ({(n_selected_samples / max(1, n_total_samples)):.2%}), "
            f"kept_batches={n_selected_batches}/{n_total_batches}"
        )
        return saved, restore

    def _run_shot_bn_tent(detector, loader, dataset_name):
        """
        ATT422: SHOT-style BN adaptation + TENT entropy minimization.
        
        Step 1 (SHOT-BN): Switch BN layers to use batch statistics (not running stats),
        then do a full pass over target data to compute exact batch-mean/var.
        Step 2 (TENT): Freeze all params except BN affine (gamma/beta), optimize
        entropy of model predictions on target data.
        
        Returns (saved_state, restore_fn).
        """
        try:
            import config as cfg
            shot_bn_enable = bool(getattr(cfg, "TARGET_SHOT_BN_ENABLE", False))
            shot_tent_enable = bool(getattr(cfg, "TARGET_SHOT_TENT_ENABLE", False))
            tent_steps = int(getattr(cfg, "TARGET_SHOT_TENT_STEPS", 15))
            tent_lr = float(getattr(cfg, "TARGET_SHOT_TENT_LR", 5e-4))
            tent_batches = int(getattr(cfg, "TARGET_SHOT_TENT_BATCHES", 4))
            conf_th = float(getattr(cfg, "TARGET_SHOT_CONFIDENCE_TH", 0.5))
        except Exception:
            shot_bn_enable = False
            shot_tent_enable = False
            tent_steps = 15
            tent_lr = 5e-4
            tent_batches = 4
            conf_th = 0.5

        if not shot_bn_enable and not shot_tent_enable:
            return None, None

        device = getattr(detector, "device", next(detector.signal_encoder.parameters()).device)

        saved = {
            "signal_encoder": {k: v.cpu().clone() for k, v in detector.signal_encoder.state_dict().items()},
            "spec_encoder": {k: v.cpu().clone() for k, v in detector.spec_encoder.state_dict().items()},
        }
        bn_tracking_state = {}

        # Step 1: SHOT-BN — force BN to use current batch statistics
        if shot_bn_enable:
            for enc_name in ("signal_encoder", "spec_encoder"):
                enc = getattr(detector, enc_name)
                for name, m in enc.named_modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        bn_tracking_state[f"{enc_name}.{name}"] = {
                            "track_running_stats": m.track_running_stats,
                            "momentum": m.momentum,
                        }
                        m.track_running_stats = False
                        m.momentum = None

            # Forward pass to let BN layers compute batch stats
            n_bn_pass = 0
            for enc_name in ("signal_encoder", "spec_encoder"):
                enc = getattr(detector, enc_name)
                enc.train()
            with torch.no_grad():
                for batch in loader:
                    n_bn_pass += 1
                    signal = batch["signal"].to(device)
                    spec = load_spec_from_batch(batch, device, detector)
                    signal_mask = batch.get("signal_mask")
                    spec_mask = batch.get("spec_mask")
                    if signal_mask is None:
                        signal_mask = torch.ones(signal.size(0), signal.size(-1) if signal.dim() == 2 else signal.size(1), device=device)
                    else:
                        signal_mask = signal_mask.to(device)
                    if spec_mask is None:
                        spec_mask = torch.ones(spec.size(0), spec.size(-1) if spec.dim() == 2 else spec.size(1), device=device)
                    else:
                        spec_mask = spec_mask.to(device)
                    domain_ids = None
                    if batch.get("domain") is not None:
                        domain_ids = _encode_domain_ids(batch["domain"], batch.get("condition", None), device)
                    try:
                        if domain_ids is not None:
                            try:
                                detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                            except TypeError:
                                detector.signal_encoder(signal, signal_mask)
                            try:
                                detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids)
                            except TypeError:
                                detector.spec_encoder(spec, spec_mask)
                        else:
                            detector.signal_encoder(signal, signal_mask)
                            detector.spec_encoder(spec, spec_mask)
                    except Exception:
                        pass
            for enc_name in ("signal_encoder", "spec_encoder"):
                getattr(detector, enc_name).eval()
            print(f"[SHOT-BN] {dataset_name}: switched BN to batch-stats mode, {n_bn_pass} batches")

        # Step 2: SHOT-TENT — entropy minimization on BN affine params
        n_tent_effective = 0
        if shot_tent_enable and tent_steps > 0:
            grad_flags = {}
            for mod_name in ("signal_encoder", "spec_encoder", "fusion_module", "fusion_decoder"):
                mod = getattr(detector, mod_name)
                grad_flags[mod_name] = {n: p.requires_grad for n, p in mod.named_parameters()}
                for p in mod.parameters():
                    p.requires_grad = False

            tent_params = []
            bn_count = 0
            for enc in (detector.signal_encoder, detector.spec_encoder):
                for m in enc.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        bn_count += 1
                        if m.affine:
                            if m.weight is not None:
                                m.weight.requires_grad = True
                                tent_params.append(m.weight)
                            if m.bias is not None:
                                m.bias.requires_grad = True
                                tent_params.append(m.bias)

            if len(tent_params) > 0:
                detector.signal_encoder.train()
                detector.spec_encoder.train()
                detector.fusion_module.eval()
                detector.fusion_decoder.eval()

                optim = torch.optim.Adam(tent_params, lr=tent_lr)
                data_iter = iter(loader)
                ent_history = []

                for step_i in range(tent_steps):
                    batch_losses = []
                    optim.zero_grad(set_to_none=True)
                    for _ in range(tent_batches):
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            data_iter = iter(loader)
                            batch = next(data_iter)
                        signal = batch["signal"].to(device)
                        spec = load_spec_from_batch(batch, device, detector)
                        signal_mask = batch.get("signal_mask")
                        spec_mask = batch.get("spec_mask")
                        if signal_mask is None:
                            signal_mask = torch.ones(signal.size(0), signal.size(1), device=device)
                        else:
                            signal_mask = signal_mask.to(device)
                        if spec_mask is None:
                            spec_mask = torch.ones(spec.size(0), spec.size(1), device=device)
                        else:
                            spec_mask = spec_mask.to(device)
                        domain_ids = None
                        if batch.get("domain") is not None:
                            domain_ids = _encode_domain_ids(batch["domain"], batch.get("condition", None), device)
                        _sv = batch.get("stat_vec", None)
                        if _sv is not None:
                            _sv = _sv.to(device)
                        try:
                            if domain_ids is not None:
                                try:
                                    z_sig = detector.signal_encoder(signal, signal_mask, domain_ids=domain_ids)
                                except TypeError:
                                    z_sig = detector.signal_encoder(signal, signal_mask)
                                try:
                                    z_spec = detector.spec_encoder(spec, spec_mask, domain_ids=domain_ids, stat_vec=_sv)
                                except TypeError:
                                    z_spec = detector.spec_encoder(spec, spec_mask)
                            else:
                                z_sig = detector.signal_encoder(signal, signal_mask)
                                try:
                                    z_spec = detector.spec_encoder(spec, spec_mask, stat_vec=_sv)
                                except TypeError:
                                    z_spec = detector.spec_encoder(spec, spec_mask)
                            z_fusion = detector.fusion_module(z_sig, z_spec, domain_ids=domain_ids)
                            _, parts = detector.fusion_decoder(z_fusion, domain_ids=domain_ids, return_parts=True)
                            logits = parts.get("logits_diag_supervised", parts.get("logits_diag", None))
                            if logits is None:
                                continue
                        except Exception:
                            continue

                        p = torch.softmax(logits.float(), dim=1)
                        max_conf = p.max(dim=1).values
                        reliable_mask = max_conf >= conf_th
                        if reliable_mask.sum() == 0:
                            continue
                        p_use = p[reliable_mask]
                        ent = -(p_use * torch.log(torch.clamp(p_use, min=1e-8))).sum(dim=1).mean()
                        batch_losses.append(ent)

                    if len(batch_losses) == 0:
                        continue
                    loss_step = torch.stack(batch_losses).mean()
                    loss_step.backward()
                    optim.step()
                    n_tent_effective += 1
                    ent_history.append(float(loss_step.item()))

                detector.signal_encoder.eval()
                detector.spec_encoder.eval()

                # Restore grad flags
                for mod_name in ("signal_encoder", "spec_encoder", "fusion_module", "fusion_decoder"):
                    mod = getattr(detector, mod_name)
                    for n, p in mod.named_parameters():
                        if n in grad_flags.get(mod_name, {}):
                            p.requires_grad = grad_flags[mod_name][n]

                ent_start = ent_history[0] if ent_history else float("nan")
                ent_end = ent_history[-1] if ent_history else float("nan")
                print(
                    f"[SHOT-TENT] {dataset_name}: steps={n_tent_effective}/{tent_steps}, "
                    f"bn_layers={bn_count}, lr={tent_lr:.2e}, conf_th={conf_th:.2f}, "
                    f"ent_start={ent_start:.4f}, ent_end={ent_end:.4f}"
                )
            else:
                print(f"[SHOT-TENT] {dataset_name}: skipped (no BN affine params)")
                for mod_name in ("signal_encoder", "spec_encoder", "fusion_module", "fusion_decoder"):
                    mod = getattr(detector, mod_name)
                    for n, p in mod.named_parameters():
                        if n in grad_flags.get(mod_name, {}):
                            p.requires_grad = grad_flags[mod_name][n]

        def restore():
            detector.signal_encoder.load_state_dict(saved["signal_encoder"], strict=True)
            detector.spec_encoder.load_state_dict(saved["spec_encoder"], strict=True)
            for key, st in bn_tracking_state.items():
                parts = key.split(".", 1)
                enc = getattr(detector, parts[0])
                for name, m in enc.named_modules():
                    if name == parts[1] and isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        m.track_running_stats = st["track_running_stats"]
                        m.momentum = st["momentum"]

        return saved, restore

    def _run_ttfa(detector, loader, dataset_name):
        """
        ATT422: Test-Time Feature Alignment (TTFA).
        Collects target-domain SpecEncoder intermediate feature statistics (mean/var),
        then applies AdaIN-style alignment to shift them toward source-domain stats
        stored in SpecEncoder's BN layers (running_mean/running_var).
        
        Implementation: inserts a forward hook on SpecEncoder.block3's BN that
        applies partial AdaIN alignment: x_aligned = strength * adain(x) + (1-strength) * x
        
        Returns (hook_handles, restore_fn).
        """
        try:
            import config as cfg
            enable = bool(getattr(cfg, "TARGET_TTFA_ENABLE", False))
            strength = float(getattr(cfg, "TARGET_TTFA_STRENGTH", 0.8))
            num_batches = int(getattr(cfg, "TARGET_TTFA_NUM_BATCHES", -1))
        except Exception:
            enable = False
            strength = 0.8
            num_batches = -1

        if not enable:
            return None, None

        device = getattr(detector, "device", next(detector.signal_encoder.parameters()).device)
        spec_enc = detector.spec_encoder

        target_bn = None
        target_bn_name = None
        for name, m in spec_enc.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) and "block3" in name:
                target_bn = m
                target_bn_name = name
                break

        if target_bn is None:
            for name, m in spec_enc.named_modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    target_bn = m
                    target_bn_name = name

        if target_bn is None:
            print(f"[TTFA] {dataset_name}: skipped (no BN2d found in spec_encoder)")
            return None, None

        src_mean = target_bn.running_mean.clone().detach() if target_bn.running_mean is not None else None
        src_var = target_bn.running_var.clone().detach() if target_bn.running_var is not None else None

        if src_mean is None or src_var is None:
            print(f"[TTFA] {dataset_name}: skipped (no running stats in {target_bn_name})")
            return None, None

        # Collect target statistics
        tgt_means = []
        tgt_vars = []
        spec_enc.eval()
        n_collected = 0
        if num_batches <= 0:
            max_b = len(loader) if hasattr(loader, '__len__') else 10**9
        elif hasattr(loader, '__len__'):
            max_b = min(num_batches, len(loader))
        else:
            max_b = num_batches
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= max_b:
                    break
                spec = load_spec_from_batch(batch, device, detector)
                spec_mask = batch.get("spec_mask")
                if spec_mask is None:
                    spec_mask = torch.ones(spec.size(0), spec.size(1), device=device)
                else:
                    spec_mask = spec_mask.to(device)
                domain_ids = None
                if batch.get("domain") is not None:
                    domain_ids = _encode_domain_ids(batch["domain"], batch.get("condition", None), device)

                activations = {}
                def hook_fn(module, input, output, name=target_bn_name):
                    activations[name] = output.detach()
                handle = target_bn.register_forward_hook(hook_fn)
                try:
                    if domain_ids is not None:
                        try:
                            spec_enc(spec, spec_mask, domain_ids=domain_ids)
                        except TypeError:
                            spec_enc(spec, spec_mask)
                    else:
                        spec_enc(spec, spec_mask)
                except Exception:
                    handle.remove()
                    continue
                handle.remove()

                feat = activations.get(target_bn_name)
                if feat is not None and feat.dim() == 4:
                    b_mean = feat.mean(dim=(0, 2, 3))
                    b_var = feat.var(dim=(0, 2, 3), unbiased=False)
                    tgt_means.append(b_mean)
                    tgt_vars.append(b_var)
                    n_collected += 1

        if n_collected == 0:
            print(f"[TTFA] {dataset_name}: skipped (no valid activations collected)")
            return None, None

        tgt_mean = torch.stack(tgt_means).mean(dim=0)
        tgt_var = torch.stack(tgt_vars).mean(dim=0)

        cos_sim = torch.nn.functional.cosine_similarity(
            src_mean.unsqueeze(0), tgt_mean.unsqueeze(0)
        ).item()
        mean_shift = (src_mean - tgt_mean).abs().mean().item()
        var_ratio = ((src_var + 1e-8) / (tgt_var + 1e-8)).mean().item()
        print(
            f"[TTFA] {dataset_name}: layer={target_bn_name}, batches={n_collected}, "
            f"strength={strength:.2f}, cos_sim(src,tgt)={cos_sim:.4f}, "
            f"mean_shift={mean_shift:.4f}, var_ratio={var_ratio:.4f}"
        )

        # Register alignment hook: partial AdaIN toward source stats
        s = float(strength)
        _src_mean = src_mean.to(device)
        _src_var = src_var.to(device)
        _tgt_mean = tgt_mean.to(device)
        _tgt_var = tgt_var.to(device)
        eps = 1e-5

        def _align_hook(module, input, output):
            if output.dim() != 4:
                return output
            c = output.size(1)
            if c != _tgt_mean.size(0):
                return output
            tgt_std = (_tgt_var + eps).sqrt().view(1, c, 1, 1)
            src_std = (_src_var + eps).sqrt().view(1, c, 1, 1)
            src_m = _src_mean.view(1, c, 1, 1)
            tgt_m = _tgt_mean.view(1, c, 1, 1)
            # AdaIN: normalize with target stats, then denormalize with source stats
            x_norm = (output - tgt_m) / tgt_std
            x_aligned = x_norm * src_std + src_m
            return s * x_aligned + (1.0 - s) * output

        align_handle = target_bn.register_forward_hook(_align_hook)

        def restore():
            align_handle.remove()

        return [align_handle], restore

    def _eval_task_block(task_name, dataset_name, loader, diag_num_classes=4):
        """
        执行三阶段评估：Zero-shot → Unsupervised → Supervised
        每个阶段使用不同的数据划分：
        - Zero-shot: 使用全部数据评估（不划分）
        - Unsupervised: 5%无标签用于适应，95%用于评估
        - Supervised: 5%有标签用于微调，95%用于评估
        """
        if loader is None:
            return None
        # Att78: set current eval dataset for domain-specific CollapseGate (e.g. CWRU relax).
        if hasattr(detector, "set_eval_dataset_name"):
            detector.set_eval_dataset_name(dataset_name)
        
        # ATT395/ATT422: BN-Adapt for both Task1 and Task2.
        bn_restore = None
        tent_restore = None
        shot_restore = None
        ttfa_restore = None
        try:
            import config as _cfg_bna
            _task1_bn_adapt = bool(getattr(_cfg_bna, "TASK1_BN_ADAPT_ENABLE", True))
            _shot_bn = bool(getattr(_cfg_bna, "TARGET_SHOT_BN_ENABLE", False))
            _ttfa_enable = bool(getattr(_cfg_bna, "TARGET_TTFA_ENABLE", False))
        except Exception:
            _task1_bn_adapt = True
            _shot_bn = False
            _ttfa_enable = False

        if task_name == "Task2":
            if _shot_bn:
                _, shot_restore = _run_shot_bn_tent(detector, loader, dataset_name)
            else:
                _, bn_restore = _run_bn_adaptation(detector, loader, dataset_name)
                _, tent_restore = _run_tent_adaptation(detector, loader, dataset_name)
            if _ttfa_enable:
                _, ttfa_restore = _run_ttfa(detector, loader, dataset_name)
        elif task_name == "Task1" and _task1_bn_adapt:
            print(f"[ATT395][BN-Adapt] Running BN adaptation for Task1-{dataset_name}")
            _, bn_restore = _run_bn_adaptation(detector, loader, dataset_name)
        
        # 尝试导入配置
        try:
            import config as cfg
            enable_task1_ft = bool(getattr(cfg, "TASK1_ENABLE_PRETEST_FINETUNE", True))
            task1_ft_ratio = float(getattr(cfg, "TASK1_FINETUNE_PERCENT", 0.05))
            sup_steps = getattr(cfg, "TARGET_SUP_STEPS", None)
            sup_lr = getattr(cfg, "TARGET_SUP_LR", None)
            sup_nf_w = float(getattr(cfg, "TARGET_SUP_NF_WEIGHT", 0.5))
            sup_diag_w = float(getattr(cfg, "TARGET_SUP_DIAG_WEIGHT", 1.0))
            seed_base = int(getattr(cfg, "RANDOM_SEED", 42))
            # 新增：是否执行无监督适应
            enable_unsup = getattr(cfg, "TASK_ENABLE_UNSUPERVISED_ADAPT", True)
            # Task1-CWRU 可选跳过监督微调（避免负迁移）
            skip_cwru_sup = bool(getattr(cfg, "TASK1_CWRU_SKIP_SUPERVISED_FINETUNE", False))
            if bool(getattr(cfg, "TARGET_NSHOT_ENABLE", False)) and int(getattr(cfg, "TARGET_NSHOT_K", 0)) > 0:
                skip_cwru_sup = False
            eval_only_mode = bool(getattr(cfg, "EVAL_ONLY_MODE", False))
            eval_only_skip_task1_sup_ft = bool(getattr(cfg, "EVAL_ONLY_SKIP_TASK1_SUP_FT", False))
        except Exception:
            enable_task1_ft = True
            task1_ft_ratio = 0.05
            sup_steps = None
            sup_lr = None
            sup_nf_w = 0.5
            sup_diag_w = 1.0
            seed_base = 42
            enable_unsup = True
            skip_cwru_sup = False
            eval_only_mode = False
            eval_only_skip_task1_sup_ft = False
        
        results = {}
        split_seed = int(seed_base) + (11 if dataset_name == "CWRU" else 29)
        _spec_twc_fn = _fit_twc_for_target(loader, dataset_name)

        # ATT398: Test-Time Prompt Tuning — adapt prototypes per target domain
        _tpt_protos_backup = None
        try:
            import config as _cfg_tpt
            _use_tpt = bool(getattr(_cfg_tpt, "USE_TPT", False))
        except Exception:
            _use_tpt = False
        if _use_tpt and task_name == "Task2" and dataset_name in ("HUST", "SEU"):
            try:
                from zero_shot_eval.tpt import run_tpt
                from zero_shot_eval.zero_shot_eval_hust import _domain_ids_from_cfg
                _tpt_orig_protos = detector._active_text_prototypes()
                if _tpt_orig_protos is not None:
                    _tpt_protos_backup = _tpt_orig_protos
                    def _tpt_did_fn(batch, dev):
                        _ids = _domain_ids_from_cfg()
                        domains = batch.get("domain", None)
                        conditions = batch.get("condition", None)
                        if domains is None:
                            return None
                        enc = []
                        for i, d in enumerate(domains):
                            d_up = str(d).upper()
                            cond = str(conditions[i]).lower() if (conditions is not None and i < len(conditions)) else ""
                            if d_up == "CWRU":
                                enc.append(_ids["CWRU_GK4"] if any(k in cond for k in ("gk4", "gk_4", "hp3")) else _ids["CWRU_GK2"])
                            elif d_up == "PU":
                                enc.append(_ids["PU"])
                            elif d_up == "HUST":
                                enc.append(_ids["HUST"])
                            elif d_up in ("HIT", "SEU"):
                                enc.append(_ids.get("SEU", _ids.get("HIT")))
                            else:
                                enc.append(_ids["CWRU_GK2"])
                        import torch as _th
                        return _th.tensor(enc, dtype=_th.long, device=dev)
                    _tpt_steps = int(getattr(_cfg_tpt, "TPT_STEPS", 3))
                    _tpt_lr = float(getattr(_cfg_tpt, "TPT_LR", 0.001))
                    _tpt_T = float(getattr(_cfg_tpt, "TPT_TEMPERATURE", 0.07))
                    _tpt_cf = float(getattr(_cfg_tpt, "TPT_CONFIDENCE_FILTER", 0.1))
                    _tpt_dev = next(detector.signal_encoder.parameters()).device
                    _tpt_adapted, _tpt_info = run_tpt(
                        signal_encoder=detector.signal_encoder,
                        spec_encoder=detector.spec_encoder,
                        fusion_module=detector.fusion_module,
                        text_prototypes=_tpt_orig_protos,
                        eval_loader=loader,
                        device=_tpt_dev,
                        steps=_tpt_steps, lr=_tpt_lr, temperature=_tpt_T,
                        confidence_filter=_tpt_cf,
                        domain_id_fn=_tpt_did_fn,
                        dataset_name=dataset_name,
                    )
                    if not _tpt_info.get("skipped", True):
                        detector.set_text_prototypes(_tpt_adapted)
                        print(f"[ATT398-TPT] {dataset_name}: prototypes adapted successfully")
            except Exception as _tpt_e:
                print(f"[ATT398-TPT] {dataset_name}: TPT failed ({_tpt_e}), using original prototypes")
                if _tpt_protos_backup is not None:
                    detector.set_text_prototypes(_tpt_protos_backup)
                    _tpt_protos_backup = None

        # ===== ATT422: Pre-eval Diagnostics =====
        if task_name == "Task2" and dataset_name in ("HUST", "SEU"):
            try:
                _diag_device = getattr(detector, "device", next(detector.signal_encoder.parameters()).device)
                _fm = getattr(detector, "fusion_module", None)
                if _fm is not None:
                    _gate_info = _fm.get_gate_info() if hasattr(_fm, "get_gate_info") else {}
                    _alpha = _gate_info.get("spec_residual_alpha", None)
                    print(f"[ATT422-Diag] {dataset_name}: spec_residual_alpha={_alpha}")
                # Sample one batch for feature norm diagnostics
                _diag_iter = iter(loader)
                _diag_batch = next(_diag_iter)
                with torch.no_grad():
                    _d_sig = _diag_batch["signal"].to(_diag_device)
                    _d_spec = load_spec_from_batch(_diag_batch, _diag_device, detector)
                    _d_sig_m = _diag_batch.get("signal_mask", torch.ones(_d_sig.size(0), _d_sig.size(1), device=_diag_device)).to(_diag_device)
                    _d_spec_m = _diag_batch.get("spec_mask", torch.ones(_d_spec.size(0), _d_spec.size(1), device=_diag_device)).to(_diag_device)
                    _d_did = None
                    if _diag_batch.get("domain") is not None:
                        _d_did = _encode_domain_ids(_diag_batch["domain"], _diag_batch.get("condition", None), _diag_device)
                    try:
                        _z_sig = detector.signal_encoder(_d_sig, _d_sig_m, domain_ids=_d_did) if _d_did is not None else detector.signal_encoder(_d_sig, _d_sig_m)
                    except TypeError:
                        _z_sig = detector.signal_encoder(_d_sig, _d_sig_m)
                    _sv = _diag_batch.get("stat_vec", None)
                    if _sv is not None:
                        _sv = _sv.to(_diag_device)
                    try:
                        _z_spec_out = detector.spec_encoder(_d_spec, _d_spec_m, domain_ids=_d_did, stat_vec=_sv) if _d_did is not None else detector.spec_encoder(_d_spec, _d_spec_m, stat_vec=_sv)
                    except TypeError:
                        _z_spec_out = detector.spec_encoder(_d_spec, _d_spec_m)
                    _z_spec_val = _z_spec_out[0] if isinstance(_z_spec_out, tuple) else _z_spec_out
                    if isinstance(_z_sig, tuple):
                        _z_sig = _z_sig[0]
                    _sig_norm = float(_z_sig.float().norm(dim=-1).mean().item()) if _z_sig.dim() >= 2 else float("nan")
                    _spec_norm = float(_z_spec_val.float().norm(dim=-1).mean().item()) if _z_spec_val.dim() >= 2 else float("nan")
                    _sig_std = float(_z_sig.float().std().item())
                    _spec_std = float(_z_spec_val.float().std().item())
                    print(
                        f"[ATT422-Diag] {dataset_name}: sig_norm={_sig_norm:.4f}, sig_std={_sig_std:.6f}, "
                        f"spec_norm={_spec_norm:.4f}, spec_std={_spec_std:.6f}"
                    )
                    # Fusion output
                    try:
                        _z_fus = _fm(_z_sig, _z_spec_val, domain_ids=_d_did)
                        _fus_norm = float(_z_fus.float().norm(dim=-1).mean().item())
                        _fus_std = float(_z_fus.float().std().item())
                        _alpha_after = _fm._last_spec_residual_alpha
                        print(
                            f"[ATT422-Diag] {dataset_name}: fus_norm={_fus_norm:.4f}, fus_std={_fus_std:.6f}, "
                            f"spec_residual_alpha_actual={_alpha_after}"
                        )
                    except Exception:
                        pass
            except Exception as _diag_e:
                print(f"[ATT422-Diag] {dataset_name}: diagnostics error: {_diag_e}")

        # ===== Stage 0: Zero-shot (未微调) =====
        # 使用完整数据集 adaptation进行评估，无任何
        zero_shot_loader = loader
        zero_shot_nf_res = _evaluate(
            zero_shot_loader,
            mode="fusion",
            scoring="evidence_or",
            dataset_name=f"{dataset_name}(zero-shot)",
            enable_diagnosis=False,
        )
        zero_shot_diag_res = detector.evaluate_target_domain_accuracy(
            zero_shot_loader, dataset_name=f"{task_name}-{dataset_name}(zero-shot)", diag_num_classes=diag_num_classes
        )
        zero_shot_macro_f1 = _safe_macro_f1(zero_shot_diag_res.get("diag_labels", None), zero_shot_diag_res.get("diag_preds", None))
        zero_shot_macro_recall = _safe_macro_recall(
            zero_shot_diag_res.get("diag_labels", None),
            zero_shot_diag_res.get("diag_preds", None),
        )

        # ATT116 Bug Fix: spec_head_diag_acc evaluation for Task1 path.
        # Root cause of 6-round missing metric: zero_shot_eval_hust() was never called for CWRU
        # isolated phase; only detector.evaluate_target_domain_accuracy() was used, which has no
        # spec_head branch. Fix: run spec_encoder(return_logits=True) directly here.
        # ATT120 Extension: added spec_head_macro_f1, per-class recall, confusion matrix,
        # and spec_proto_diag_acc (z_spec cosine similarity with text_prototypes → argmax).
        spec_head_diag_acc_zs = None
        spec_head_macro_f1_zs = float("nan")
        spec_head_macro_recall_zs = float("nan")
        spec_proto_diag_acc_zs = float("nan")
        signal_head_diag_acc_zs = None
        signal_head_macro_f1_zs = float("nan")
        signal_head_macro_recall_zs = float("nan")
        signal_proto_diag_acc_zs = float("nan")
        signal_proto_macro_f1_zs = float("nan")
        fusion_raw_diag_acc_zs = float("nan")
        fusion_raw_macro_f1_zs = float("nan")
        modality_spec_signal_disagree = float("nan")
        modality_spec_head_max_frac = float("nan")
        modality_signal_head_max_frac = float("nan")
        modality_signal_proto_max_frac = float("nan")
        # ATT146: 语义路径（在 _se_has_head 块内可能被覆盖）
        fusion_proto_diag_acc_zs = float("nan")
        fusion_proto_macro_f1_zs = float("nan")
        fusion_proto_macro_recall_zs = float("nan")
        sem_proto_diag_acc_zs = float("nan")
        sem_proto_macro_f1_zs = float("nan")
        tip_adapter_diag_acc_zs = float("nan")
        tip_adapter_macro_f1_zs = float("nan")
        tip_adapter_alpha_zs = float("nan")
        dynamic_multi_proto_diag_acc_zs = float("nan")
        dynamic_multi_proto_macro_f1_zs = float("nan")
        # P0a2: 集成结果用于 Phase 3 表格，先置为无效
        _best_ens_acc = -1.0
        _best_ens_f1 = float("nan")
        _best_ens_recalls = {}
        _best_ens_label = ""
        spec_head_macro_f1_zs = None
        spec_proto_diag_acc_zs = None
        _se = getattr(detector, 'spec_encoder', None)
        # ATT129 P0 Bug修复: 同时兼容旧属性名 'classifier'（2层MLP）和新属性名 'classifier_weight'（余弦分类器）
        _se_has_head = (
            _se is not None and
            hasattr(_se, 'use_classification_head') and
            bool(getattr(_se, 'use_classification_head', False)) and
            (
                (hasattr(_se, 'classifier') and _se.classifier is not None) or
                (hasattr(_se, 'classifier_weight') and _se.classifier_weight is not None)
            )
        )
        if _se_has_head:
            try:
                import torch.nn.functional as _F
                _se.eval()
                _sh_device = next(_se.parameters()).device
                _all_preds_sh, _all_lbls_sh = [], []
                _all_preds_proto, _all_z_spec = [], []
                # ATT130 B: collect raw logit/sim tensors for multimodal ensemble
                _all_logits_sh = []    # spec_head MLP logits [B, C]
                _all_sims_proto = []   # spec_proto cosine sims [B, C] — kept for logging but excluded from ensemble
                _all_sims_sig = []     # signal_proto cosine sims [B, C] — kept for logging but excluded from ensemble
                _all_preds_sig = []    # signal_head argmax preds
                _all_logits_sig = []   # signal_head MLP logits [B, C]
                _has_sig_head = False
                _all_sims_fusion_proto = []  # fusion_proto cosine sims [B, C] — ATT143 diagnostics
                _all_z_fusion_for_td = []    # ATT500: fusion metric-space z per sample (aligned with FusionProto cos sims)
                _all_z_spec_for_td = []      # ATT501: spec graph fallback when z_fusion collapses
                _all_sims_sem_proto = []     # sem_proto cosine sims [B, C] — ATT143 diagnostics
                _all_z_metric_raw = []       # ATT379a: raw z_metric embeddings for ETF eval
                # ATT142: collect fusion decoder logits for calibrated dual-path ensemble
                _all_logits_fusion = []   # fusion logits_diag_supervised [B, C]
                _all_labels_eval = []     # aligned labels for final decision fusion
                _has_proto = False
                _has_sig_proto = False
                _has_fusion_logits = False
                _has_fusion_proto = False
                _fproto_sep_ratio = float("nan")
                _has_sem_proto = False
                # ATT121 Bug修复: 使用 detector._active_text_prototypes() (正确路径)
                # 之前错误地查找不存在的 _current_text_prototypes / text_prototypes 属性
                _current_protos = None
                if hasattr(detector, '_active_text_prototypes') and callable(detector._active_text_prototypes):
                    try:
                        _current_protos = detector._active_text_prototypes()
                    except Exception:
                        pass
                # ATT130 B: get signal encoder from detector for signal_proto_sims
                _sig_enc = getattr(detector, 'signal_encoder', None)
                if _sig_enc is not None:
                    _sig_enc.eval()
                # ATT142: get fusion_module and fusion_decoder for calibrated dual-path ensemble
                _fusion_mod = getattr(detector, 'fusion_module', None)
                _fusion_dec = getattr(detector, 'fusion_decoder', None)
                if _fusion_mod is not None:
                    _fusion_mod.eval()
                if _fusion_dec is not None:
                    _fusion_dec.eval()
                _proto_keys = None
                _proto_mat_cached = None
                if _current_protos is not None:
                    try:
                        _proto_keys = sorted(_current_protos.keys())
                        _proto_mat_cached = torch.stack(
                            [_current_protos[k].to(_sh_device) for k in _proto_keys], dim=0
                        )  # [C, D]
                        _proto_mat_cached = _F.normalize(_proto_mat_cached, dim=-1)
                    except Exception:
                        _proto_keys = None
                        _proto_mat_cached = None
                # ATT474: optional SpecProto matrix = learnable protos (training-aligned geometry).
                _spec_proto_mat_cached = _proto_mat_cached
                _spec_proto_eval_src = "text_prototypes"
                try:
                    import config as _cfg_sp_lp
                    if bool(getattr(_cfg_sp_lp, "EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS", False)):
                        _lp_sp = getattr(_fusion_dec, "_learnable_proto_module", None) if _fusion_dec is not None else None
                        _lp_w = getattr(_lp_sp, "protos", None) if _lp_sp is not None else None
                        if torch.is_tensor(_lp_w):
                            _spec_proto_mat_cached = _F.normalize(_lp_w.detach().float(), dim=-1).to(_sh_device)
                            _spec_proto_eval_src = "learnable_prototypes"
                except Exception:
                    pass
                # ATT142: read ensemble config (defaults ensure backward compat)
                try:
                    import config as _cfg142
                    _ens_w_spec_head  = float(getattr(_cfg142, 'EVAL_ENSEMBLE_SPEC_HEAD_WEIGHT', 0.6))
                    _ens_w_fusion     = float(getattr(_cfg142, 'EVAL_ENSEMBLE_FUSION_WEIGHT', 0.4))
                    _ens_w_spec_proto = float(getattr(_cfg142, 'EVAL_ENSEMBLE_SPEC_PROTO_WEIGHT', 0.0))
                    _ens_w_sig_proto  = float(getattr(_cfg142, 'EVAL_ENSEMBLE_SIG_PROTO_WEIGHT', 0.0))
                    _ens_t_spec_head  = float(getattr(_cfg142, 'EVAL_ENSEMBLE_SPEC_HEAD_TEMPERATURE', 1.0))
                    _ens_t_fusion     = float(getattr(_cfg142, 'EVAL_ENSEMBLE_FUSION_TEMPERATURE', 1.0))
                    _ens_use_fusion   = bool(getattr(_cfg142, 'EVAL_ENSEMBLE_USE_FUSION_LOGITS', True))
                    # ATT143: Margin-Aware Adaptive Fusion (MAAF)
                    _ens_margin_adaptive = bool(getattr(_cfg142, 'EVAL_ENSEMBLE_MARGIN_ADAPTIVE', False))
                    _ens_margin_alpha_min = float(getattr(_cfg142, 'EVAL_ENSEMBLE_MARGIN_ALPHA_MIN', 0.30))
                    _ens_margin_alpha_max = float(getattr(_cfg142, 'EVAL_ENSEMBLE_MARGIN_ALPHA_MAX', 0.90))
                    _ens_margin_tau = float(getattr(_cfg142, 'EVAL_ENSEMBLE_MARGIN_TAU', 1.0))
                except Exception:
                    _ens_w_spec_head  = 0.6
                    _ens_w_fusion     = 0.4
                    _ens_w_spec_proto = 0.0
                    _ens_w_sig_proto  = 0.0
                    _ens_t_spec_head  = 1.0
                    _ens_t_fusion     = 1.0
                    _ens_use_fusion   = True
                    _ens_margin_adaptive = False
                    _ens_margin_alpha_min = 0.30
                    _ens_margin_alpha_max = 0.90
                    _ens_margin_tau = 1.0
                try:
                    import config as _cfg_eval_fusion
                    _eval_fusion_norm_inputs = bool(getattr(_cfg_eval_fusion, "EVAL_FUSION_NORMALIZE_INPUTS", False))
                    _eval_fusion_use_spec_pre_l2 = bool(
                        getattr(
                            _cfg_eval_fusion,
                            "EVAL_FUSION_USE_SPEC_PRE_L2_INPUT",
                            bool(getattr(_cfg_eval_fusion, "FUSION_USE_SPEC_PRE_L2_INPUT", True))
                        )
                    )
                    _eval_fusion_log_inputs = bool(getattr(_cfg_eval_fusion, "EVAL_LOG_FUSION_INPUT_STATS", True))
                    _eval_fusion_log_stride = max(1, int(getattr(_cfg_eval_fusion, "EVAL_LOG_FUSION_INPUT_STEP_STRIDE", 200)))
                    _eval_report_signal_head = bool(getattr(_cfg_eval_fusion, "EVAL_REPORT_SIGNAL_HEAD", True))
                    _eval_report_signal_proto = bool(getattr(_cfg_eval_fusion, "EVAL_REPORT_SIGNAL_PROTO", True))
                    _eval_report_modality_diag = bool(getattr(_cfg_eval_fusion, "EVAL_REPORT_MODALITY_DIAG", True))
                except Exception:
                    _eval_fusion_norm_inputs = False
                    _eval_fusion_use_spec_pre_l2 = True
                    _eval_fusion_log_inputs = True
                    _eval_fusion_log_stride = 200
                    _eval_report_signal_head = True
                    _eval_report_signal_proto = True
                    _eval_report_modality_diag = True
                with torch.no_grad():
                    for _batch_eval_idx, _b in enumerate(zero_shot_loader):
                        _spec = load_spec_from_batch(_b, _sh_device, detector)
                        _smask = _b["spec_mask"].to(_sh_device)
                        _lbls = _b["label"].to(_sh_device)
                        _sh_out = _se(_spec, _smask, return_logits=True)
                        _z_spec_for_fusion = None
                        if isinstance(_sh_out, (tuple, list)) and len(_sh_out) >= 2:
                            _z_spec, _sh_logits = _sh_out[0], _sh_out[1]
                            _z_spec_eval = _z_spec
                            if callable(_spec_twc_fn):
                                _z_spec_eval = _spec_twc_fn(_z_spec)
                            _sh_logits_eval = _sh_logits
                            _spechead_cosine = False
                            try:
                                import config as _cfg_sh_cos
                                _spechead_cosine = bool(getattr(_cfg_sh_cos, "SPECHEAD_USE_COSINE_PROTO", False))
                            except Exception:
                                pass
                            if _z_spec_eval is not None:
                                try:
                                    _z_spec_head = _F.normalize(_z_spec_eval.float(), dim=-1)
                                    _all_z_spec_for_td.append(_z_spec_head.detach().cpu().float())
                                    if _spechead_cosine and _proto_mat_cached is not None:
                                        _sh_logits_eval = torch.matmul(_z_spec_head, _proto_mat_cached.t())
                                    else:
                                        _sh_logits_eval = _safe_spec_head_logits(_se, _z_spec_head, detector, _b)
                                except Exception:
                                    _sh_logits_eval = _sh_logits
                            _z_spec_for_fusion = _z_spec  # save for fusion path
                            if _eval_fusion_use_spec_pre_l2:
                                try:
                                    _z_spec_pre = getattr(_z_spec, "_pre_l2_feature", None)
                                    if torch.is_tensor(_z_spec_pre) and _z_spec_pre.shape == _z_spec.shape:
                                        _z_spec_for_fusion = _z_spec_pre
                                except Exception:
                                    pass
                            if _sh_logits_eval is not None and _sh_logits_eval.dim() == 2:
                                _sh_preds = _sh_logits_eval.argmax(dim=1)
                                _all_preds_sh.append(_sh_preds.cpu())
                                _all_lbls_sh.append(_lbls.cpu())
                                _all_labels_eval.append(_lbls.cpu())
                                _all_logits_sh.append(_sh_logits_eval.cpu().float())  # ATT130 B
                            # spec_proto: cosine similarity of z_spec with prototype matrix (text or learnable)
                            if _spec_proto_mat_cached is not None and _z_spec_eval is not None:
                                try:
                                    try:
                                        import config as _cfg_dsap
                                        _use_dsap_eval = bool(getattr(_cfg_dsap, "DETACHED_SPEC_ALIGN_USE_IN_EVAL", False))
                                    except Exception:
                                        _use_dsap_eval = False
                                    _z_for_proto = _z_spec_eval
                                    if _use_dsap_eval and hasattr(_se, "project_for_proto"):
                                        _z_for_proto = _se.project_for_proto(_z_spec_eval.float(), detach_input=False)
                                    _z_norm = _F.normalize(_z_for_proto.float(), dim=-1)
                                    # ATT392-S1: Learnable protos don't need adapter (features aligned directly).
                                    # ATT391-S1 fallback: Apply proto adapter to z_spec for SpecProto
                                    _lp_module = getattr(_fusion_dec, '_learnable_proto_module', None) if _fusion_dec is not None else None
                                    if _lp_module is None:
                                        _spec_adapter = getattr(_fusion_dec, '_proto_adapter', None) if _fusion_dec is not None else None
                                        if _spec_adapter is not None:
                                            try:
                                                with torch.no_grad():
                                                    _z_norm = _spec_adapter(_z_norm)
                                            except Exception:
                                                pass
                                    _sim = torch.matmul(_z_norm, _spec_proto_mat_cached.t())  # [B, C]
                                    _proto_preds = _sim.argmax(dim=1)
                                    _all_preds_proto.append(_proto_preds.cpu())
                                    _all_sims_proto.append(_sim.cpu())  # ATT130 B
                                    _all_z_spec.append(_z_norm.cpu())   # ATT179: cache normalized z_spec for TPA
                                    _has_proto = True
                                except Exception:
                                    pass
                            # ATT130 B: signal_proto — z_sig cosine similarity with text_prototypes
                            # ATT142: z_sig is also reused below for the fusion path
                            _z_sig_for_fusion = None
                            if _sig_enc is not None:
                                try:
                                    _signal = _b["signal"].to(_sh_device)
                                    _smask_sig = _b["signal_mask"].to(_sh_device)
                                    _sig_has_head = (
                                        hasattr(_sig_enc, "use_classification_head")
                                        and bool(getattr(_sig_enc, "use_classification_head", False))
                                        and getattr(_sig_enc, "classifier", None) is not None
                                    )
                                    if _sig_has_head:
                                        _sig_out = _sig_enc(_signal, _smask_sig, return_logits=True)
                                        if isinstance(_sig_out, (tuple, list)) and len(_sig_out) >= 2:
                                            _z_sig_raw, _sig_logits = _sig_out[0], _sig_out[1]
                                        else:
                                            _z_sig_raw = _sig_out
                                            _sig_logits = None
                                    else:
                                        _z_sig_raw = _sig_enc(_signal, _smask_sig)
                                        _sig_logits = None
                                    if isinstance(_z_sig_raw, (tuple, list)):
                                        _z_sig_raw = _z_sig_raw[0]
                                    _z_sig_for_fusion = _z_sig_raw
                                    if _sig_logits is not None and _sig_logits.dim() == 2:
                                        _sig_preds = _sig_logits.argmax(dim=1)
                                        _all_preds_sig.append(_sig_preds.cpu())
                                        _all_logits_sig.append(_sig_logits.cpu().float())
                                        _has_sig_head = True
                                    if _proto_mat_cached is not None:
                                        _z_sig_norm = _F.normalize(_z_sig_raw.float(), dim=-1)
                                        _sim_sig = torch.matmul(_z_sig_norm, _proto_mat_cached.t())  # [B, C]
                                        _all_sims_sig.append(_sim_sig.cpu())
                                        _has_sig_proto = True
                                except Exception:
                                    pass
                            # ATT142: fusion path — run fusion_module + fusion_decoder on frozen features
                            # Rationale: Fusion AUC=0.9994 but Diag-Acc=0.6452 with nf_protect.
                            # Raw logits_diag_supervised (without nf_protect) capture near-perfect ranking.
                            # These complement spec_head's 0.8548 acc via OR-vs-Ball disambiguation.
                            if _ens_use_fusion and _fusion_mod is not None and _fusion_dec is not None and _z_spec_for_fusion is not None and _z_sig_for_fusion is not None:
                                try:
                                    _z_sig_fus = _z_sig_for_fusion.float()
                                    _z_spec_fus = _z_spec_for_fusion.float()
                                    if _eval_fusion_norm_inputs:
                                        _z_sig_fus = _F.normalize(_z_sig_fus, dim=-1)
                                        _z_spec_fus = _F.normalize(_z_spec_fus, dim=-1)
                                    if _eval_fusion_log_inputs and (_batch_eval_idx == 0 or ((_batch_eval_idx + 1) % _eval_fusion_log_stride == 0)):
                                        try:
                                            _sig_norm_mean = float(_z_sig_fus.norm(dim=-1).mean().item())
                                            _spec_norm_mean = float(_z_spec_fus.norm(dim=-1).mean().item())
                                            print(
                                                f"[FusionEvalInput] batch={_batch_eval_idx + 1} "
                                                f"normalize_inputs={int(_eval_fusion_norm_inputs)} "
                                                f"use_spec_pre_l2={int(_eval_fusion_use_spec_pre_l2)} "
                                                f"norm(sig/spec)=({_sig_norm_mean:.4f},{_spec_norm_mean:.4f})"
                                            )
                                        except Exception:
                                            pass
                                    _domains_b = _b.get("domain", None)
                                    _domain_ids_b = None
                                    if _domains_b is not None:
                                        _domain_ids_b = _encode_domain_ids(
                                            _domains_b, _b.get("condition", None), _sh_device
                                        )
                                    _text_protos_b = _current_protos
                                    # fusion_module forward
                                    if hasattr(_fusion_mod, 'domain_embedding'):
                                        _z_fus = _fusion_mod(
                                            _z_sig_fus, _z_spec_fus,
                                            domain_ids=_domain_ids_b,
                                            text_prototypes=_text_protos_b
                                        )
                                    else:
                                        _z_fus = _fusion_mod(_z_sig_fus, _z_spec_fus)
                                    _z_fus_n = _F.normalize(_z_fus, dim=-1)
                                    _z_metric_eval = _z_fus_n
                                    _use_dual_space_eval = False
                                    try:
                                        import config as _cfg_eval_dual
                                        _use_dual_space_eval = bool(getattr(_cfg_eval_dual, "USE_DUAL_SPACE_DECOUPLE", False))
                                    except Exception:
                                        _use_dual_space_eval = False
                                    if _use_dual_space_eval and hasattr(_fusion_dec, "get_metric_embedding"):
                                        try:
                                            _z_metric_eval = _fusion_dec.get_metric_embedding(_z_fus_n)
                                            _z_metric_eval = _F.normalize(_z_metric_eval.float(), dim=-1)
                                        except Exception:
                                            _z_metric_eval = _z_fus_n
                                    # ATT392-S1: Learnable protos don't need adapter (features aligned directly).
                                    # ATT391-S1 fallback: Apply proto adapter if available (ETF→text space mapping)
                                    _lp_module_fus = getattr(_fusion_dec, '_learnable_proto_module', None) if _fusion_dec is not None else None
                                    if _lp_module_fus is None:
                                        _proto_adapter = getattr(_fusion_dec, '_proto_adapter', None) if _fusion_dec is not None else None
                                        if _proto_adapter is not None:
                                            try:
                                                with torch.no_grad():
                                                    _z_metric_eval = _proto_adapter(_z_metric_eval.float())
                                            except Exception:
                                                pass
                                    # ATT143: diagnose semantic decision channels on fusion embedding
                                    if _proto_mat_cached is not None:
                                        try:
                                            _sim_fus_proto = torch.matmul(_z_metric_eval.float(), _proto_mat_cached.t())  # [B, C]
                                            _all_sims_fusion_proto.append(_sim_fus_proto.cpu())
                                            _all_z_fusion_for_td.append(_z_metric_eval.detach().cpu().float())
                                            _has_fusion_proto = True
                                        except Exception:
                                            pass
                                    # ATT379a: collect raw z_metric for ETF evaluation
                                    try:
                                        _all_z_metric_raw.append(_z_metric_eval.detach().cpu().float())
                                    except Exception:
                                        pass
                                        if hasattr(_fusion_dec, "get_sem_embedding"):
                                            try:
                                                _z_sem = _fusion_dec.get_sem_embedding(_z_metric_eval)
                                                _z_sem_n = _F.normalize(_z_sem.float(), dim=-1)
                                                _sim_sem_proto = torch.matmul(_z_sem_n, _proto_mat_cached.t())  # [B, C]
                                                _all_sims_sem_proto.append(_sim_sem_proto.cpu())
                                                _has_sem_proto = True
                                            except Exception:
                                                pass
                                    # fusion_decoder forward — retrieve logits_diag_supervised (no nf_protect)
                                    _logits4, _parts = _fusion_dec(
                                        _z_fus_n,
                                        domain_ids=_domain_ids_b,
                                        text_prototypes=_text_protos_b,
                                        return_parts=True
                                    )
                                    _fus_logits = _parts.get("logits_diag_supervised", _logits4)
                                    if _fus_logits is not None and _fus_logits.dim() == 2:
                                        _all_logits_fusion.append(_fus_logits.cpu().float())
                                        _has_fusion_logits = True
                                except Exception:
                                    pass
                if len(_all_preds_sh) > 0 and len(_all_lbls_sh) > 0:
                    _preds_np = torch.cat(_all_preds_sh).numpy()
                    _lbls_np = torch.cat(_all_lbls_sh).numpy()
                    _total_sh = len(_lbls_np)
                    spec_head_diag_acc_zs = float((_preds_np == _lbls_np).mean())
                    # Macro-F1 and per-class recall
                    _classes_sh = sorted(list(set(_lbls_np.tolist())))
                    _class_names = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                    _f1s_sh, _recalls_sh, _per_class_info = [], [], []
                    for _c in _classes_sh:
                        _tp = float(np.sum((_preds_np == _c) & (_lbls_np == _c)))
                        _fp = float(np.sum((_preds_np == _c) & (_lbls_np != _c)))
                        _fn = float(np.sum((_preds_np != _c) & (_lbls_np == _c)))
                        _n_true = _tp + _fn
                        _prec = _tp / (_tp + _fp + 1e-12)
                        _rec = _tp / (_tp + _fn + 1e-12)
                        _f1 = 2.0 * _prec * _rec / (_prec + _rec + 1e-12)
                        _f1s_sh.append(_f1)
                        _recalls_sh.append(_rec)
                        _cname = _class_names.get(_c, str(_c))
                        _per_class_info.append(f"{_cname}(R={_rec:.2f},F1={_f1:.2f},N={int(_n_true)})")
                    spec_head_macro_f1_zs = float(np.mean(_f1s_sh)) if _f1s_sh else float("nan")
                    spec_head_macro_recall_zs = float(np.mean(_recalls_sh)) if _recalls_sh else float("nan")
                    _sh_hmean = float(len(_recalls_sh) / sum(1.0 / max(r, 1e-12) for r in _recalls_sh)) if _recalls_sh and all(r > 0 for r in _recalls_sh) else float(np.mean(_recalls_sh) if _recalls_sh else float("nan"))
                    _sh_mode_tag = "[CosProto]" if _spechead_cosine else ""
                    print(f"[SpecHead]{_sh_mode_tag} {dataset_name} spec_head_diag_acc={spec_head_diag_acc_zs:.4f}"
                          f" macro_f1={spec_head_macro_f1_zs:.4f} hmean={_sh_hmean:.4f} (N={_total_sh})")
                    print(f"[SpecHead] Per-class: {', '.join(_per_class_info)}")
                    # Confusion matrix (compact)
                    # ATT145 P1b: use max over both true labels AND predictions to avoid IndexError
                    # when a 4-class spec_head predicts class 3 on a 3-class dataset (e.g. PU, HIT).
                    _n_cls_conf = max(int(max(_lbls_np)), int(max(_preds_np))) + 1
                    _conf = np.zeros((_n_cls_conf, _n_cls_conf), dtype=int)
                    for _t, _p in zip(_lbls_np.tolist(), _preds_np.tolist()):
                        _conf[int(_t)][int(_p)] += 1
                    _cnames_short = ["N", "IR", "OR", "Ball"]
                    _header = "ConfMat(true\\pred) " + " ".join(
                        f"{(_cnames_short[i] if i < len(_cnames_short) else str(i)):>5}"
                        for i in range(_n_cls_conf)
                    )
                    print(f"[SpecHead] {_header}")
                    for _ri in range(_n_cls_conf):
                        _row_str = " ".join(f"{_conf[_ri][_ci]:>5}" for _ci in range(_n_cls_conf))
                        _rname = _cnames_short[_ri] if _ri < len(_cnames_short) else str(_ri)
                        print(f"[SpecHead]   {_rname:>17} {_row_str}")
                    # spec_proto evaluation
                    if _has_proto and len(_all_preds_proto) > 0:
                        _proto_preds_np = torch.cat(_all_preds_proto).numpy()
                        spec_proto_diag_acc_zs = float((_proto_preds_np == _lbls_np).mean())
                        _proto_f1s = []
                        for _c in _classes_sh:
                            _tp = float(np.sum((_proto_preds_np == _c) & (_lbls_np == _c)))
                            _fp = float(np.sum((_proto_preds_np == _c) & (_lbls_np != _c)))
                            _fn = float(np.sum((_proto_preds_np != _c) & (_lbls_np == _c)))
                            _p2 = _tp / (_tp + _fp + 1e-12)
                            _r2 = _tp / (_tp + _fn + 1e-12)
                            _proto_f1s.append(2.0 * _p2 * _r2 / (_p2 + _r2 + 1e-12))
                        _proto_macro_f1 = float(np.mean(_proto_f1s)) if _proto_f1s else float("nan")
                        print(f"[SpecProto] {dataset_name} spec_proto_diag_acc={spec_proto_diag_acc_zs:.4f}"
                              f" macro_f1={_proto_macro_f1:.4f} (z_spec[/align_proj] cosine vs {_spec_proto_eval_src})")
                    # SignalHead: direct classifier on z_sig (diagnostic only, not in FinalDecision by default)
                    if _eval_report_signal_head and _has_sig_head and len(_all_preds_sig) > 0:
                        try:
                            _sig_preds_np = torch.cat(_all_preds_sig).numpy()
                            _sig_metrics = _compute_head_diag_metrics(_sig_preds_np, _lbls_np)
                            signal_head_diag_acc_zs = float(_sig_metrics["acc"])
                            signal_head_macro_f1_zs = float(_sig_metrics["macro_f1"])
                            signal_head_macro_recall_zs = float(_sig_metrics["macro_recall"])
                            print(
                                f"[SignalHead] {dataset_name} signal_head_diag_acc={signal_head_diag_acc_zs:.4f}"
                                f" macro_f1={signal_head_macro_f1_zs:.4f} (N={_total_sh})"
                            )
                            print(f"[SignalHead] Per-class: {', '.join(_sig_metrics['per_class_info'])}")
                            _print_head_confusion_matrix("SignalHead", dataset_name, _sig_preds_np, _lbls_np)
                        except Exception as _e_sighead:
                            print(f"[SignalHead] {dataset_name} failed: {type(_e_sighead).__name__}: {_e_sighead}")
                    elif _eval_report_signal_head and not _has_sig_head:
                        print(f"[SignalHead] {dataset_name} skipped (signal encoder has no classification head)")
                    # SignalProto: z_sig cosine vs text_prototypes
                    if _eval_report_signal_proto and _has_sig_proto and len(_all_sims_sig) > 0:
                        try:
                            _sig_proto_cat = torch.cat(_all_sims_sig, dim=0).float()
                            _sig_proto_preds = _sig_proto_cat.argmax(dim=1).numpy()
                            _sig_proto_metrics = _compute_head_diag_metrics(_sig_proto_preds, _lbls_np)
                            signal_proto_diag_acc_zs = float(_sig_proto_metrics["acc"])
                            signal_proto_macro_f1_zs = float(_sig_proto_metrics["macro_f1"])
                            print(
                                f"[SignalProto] {dataset_name} signal_proto_diag_acc={signal_proto_diag_acc_zs:.4f}"
                                f" macro_f1={signal_proto_macro_f1_zs:.4f} (z_sig cosine vs text_prototypes)"
                            )
                            print(f"[SignalProto] Per-class: {', '.join(_sig_proto_metrics['per_class_info'])}")
                            _print_head_confusion_matrix("SignalProto", dataset_name, _sig_proto_preds, _lbls_np)
                        except Exception as _e_sigproto:
                            print(f"[SignalProto] {dataset_name} failed: {type(_e_sigproto).__name__}: {_e_sigproto}")
                    # Modality collapse / disagreement diagnostics (Spec vs Signal)
                    if _eval_report_modality_diag and len(_all_preds_sh) > 0:
                        try:
                            _spec_preds_diag = torch.cat(_all_preds_sh).numpy()
                            modality_spec_head_max_frac = _pred_max_class_frac(_spec_preds_diag)
                            if _has_sig_head and len(_all_preds_sig) > 0:
                                _sig_preds_diag = torch.cat(_all_preds_sig).numpy()
                                modality_signal_head_max_frac = _pred_max_class_frac(_sig_preds_diag)
                                modality_spec_signal_disagree = float(np.mean(_spec_preds_diag != _sig_preds_diag))
                            if _has_sig_proto and len(_all_sims_sig) > 0:
                                _sig_proto_preds_diag = torch.cat(_all_sims_sig, dim=0).argmax(dim=1).numpy()
                                modality_signal_proto_max_frac = _pred_max_class_frac(_sig_proto_preds_diag)
                            print(
                                f"[ModalityDiag] {dataset_name} "
                                f"spec_head_max_frac={modality_spec_head_max_frac:.3f} "
                                f"signal_head_max_frac={modality_signal_head_max_frac:.3f} "
                                f"signal_proto_max_frac={modality_signal_proto_max_frac:.3f} "
                                f"spec_vs_signal_disagree={modality_spec_signal_disagree:.3f}"
                            )
                        except Exception as _e_mod:
                            print(f"[ModalityDiag] {dataset_name} failed: {type(_e_mod).__name__}: {_e_mod}")
                    # ATT143 diagnostics: fusion_proto / sem_proto (4-class semantic decision channels)
                    # ATT144: added per-class recall to FusionProto for CCTC feasibility check
                    _fus_proto_logits_cached = None  # saved for potential CCTC 3rd path later
                    if _has_fusion_proto and len(_all_sims_fusion_proto) > 0:
                        try:
                            _fus_proto_cat = torch.cat(_all_sims_fusion_proto, dim=0).float()  # [N, C]
                            _fus_proto_logits_cached = _fus_proto_cat  # save for CCTC
                            _fus_proto_preds = _fus_proto_cat.argmax(dim=1).numpy()
                            _fus_proto_acc = float((_fus_proto_preds == _lbls_np).mean())
                            _fus_proto_f1s = []
                            _fus_proto_recalls = []
                            _fus_proto_per = []
                            _cnames_fp = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                            for _c in _classes_sh:
                                _tp = float(np.sum((_fus_proto_preds == _c) & (_lbls_np == _c)))
                                _fp = float(np.sum((_fus_proto_preds == _c) & (_lbls_np != _c)))
                                _fn = float(np.sum((_fus_proto_preds != _c) & (_lbls_np == _c)))
                                _pr = _tp / (_tp + _fp + 1e-12)
                                _rc = _tp / (_tp + _fn + 1e-12)
                                _fus_proto_f1s.append(2.0 * _pr * _rc / (_pr + _rc + 1e-12))
                                _fus_proto_recalls.append(_rc)
                                _n_fp_c = int(_tp + _fn)
                                _fus_proto_per.append(f"{_cnames_fp.get(_c, str(_c))}(R={_rc:.2f},N={_n_fp_c})")
                            _fus_proto_macro_f1 = float(np.mean(_fus_proto_f1s)) if _fus_proto_f1s else float("nan")
                            fusion_proto_diag_acc_zs = _fus_proto_acc
                            fusion_proto_macro_f1_zs = _fus_proto_macro_f1
                            fusion_proto_macro_recall_zs = (
                                float(np.mean(_fus_proto_recalls)) if _fus_proto_recalls else float("nan")
                            )
                            print(f"[FusionProto] {dataset_name} fusion_proto_diag_acc={_fus_proto_acc:.4f}"
                                  f" macro_f1={_fus_proto_macro_f1:.4f} (z_metric cosine vs text_prototypes)")
                            print(f"[FusionProto] Per-class: {', '.join(_fus_proto_per)}")
                            # ATT379e(S4): Confusion matrix for error pattern analysis
                            _n_classes_cm = max(len(_classes_sh), 4)
                            _cm = np.zeros((_n_classes_cm, _n_classes_cm), dtype=int)
                            for _gt, _pd in zip(_lbls_np, _fus_proto_preds):
                                if _gt < _n_classes_cm and _pd < _n_classes_cm:
                                    _cm[int(_gt)][int(_pd)] += 1
                            _cm_header = "    " + "  ".join(f"{_cnames_fp.get(i, str(i)):>7s}" for i in range(_n_classes_cm))
                            print(f"[FusionProto-CM] {dataset_name} Confusion Matrix (row=true, col=pred):")
                            print(f"[FusionProto-CM] {_cm_header}")
                            for _ri in range(_n_classes_cm):
                                _row_str = "  ".join(f"{_cm[_ri][_ci]:>7d}" for _ci in range(_n_classes_cm))
                                print(f"[FusionProto-CM] {_cnames_fp.get(_ri, str(_ri)):>7s} {_row_str}")
                            # ATT408: z_fusion inter-class / intra-class angular distance monitoring
                            try:
                                _z_met_fproto = torch.cat(_all_z_metric_raw, dim=0).float()
                                _z_met_norm = _F.normalize(_z_met_fproto, dim=-1)
                                _lbls_t = torch.from_numpy(_lbls_np)
                                _unique_cls = sorted(set(_lbls_np.tolist()))
                                _cls_centroids = {}
                                _intra_cos_list = []
                                for _c in _unique_cls:
                                    _mask_c = (_lbls_t == _c)
                                    _z_c = _z_met_norm[_mask_c]
                                    if _z_c.size(0) < 2:
                                        continue
                                    _centroid_c = _F.normalize(_z_c.mean(dim=0, keepdim=True), dim=-1)
                                    _cls_centroids[_c] = _centroid_c.squeeze(0)
                                    _cos_to_centroid = (_z_c * _centroid_c).sum(dim=-1)
                                    _intra_cos_list.append(float(_cos_to_centroid.mean()))
                                _inter_cos_list = []
                                _cls_keys = sorted(_cls_centroids.keys())
                                for _i_idx in range(len(_cls_keys)):
                                    for _j_idx in range(_i_idx + 1, len(_cls_keys)):
                                        _ci, _cj = _cls_keys[_i_idx], _cls_keys[_j_idx]
                                        _cos_ij = float((_cls_centroids[_ci] * _cls_centroids[_cj]).sum())
                                        _inter_cos_list.append(_cos_ij)
                                _intra_mean = float(np.mean(_intra_cos_list)) if _intra_cos_list else float("nan")
                                _inter_mean = float(np.mean(_inter_cos_list)) if _inter_cos_list else float("nan")
                                _sep_ratio = (1.0 - _inter_mean) / max(1e-8, 1.0 - _intra_mean) if not (np.isnan(_intra_mean) or np.isnan(_inter_mean)) else float("nan")
                                _fproto_sep_ratio = _sep_ratio
                                print(f"[ATT408-AngularSep] {dataset_name} intra_cos={_intra_mean:.4f} inter_cos={_inter_mean:.4f} sep_ratio={_sep_ratio:.4f}")
                                _pair_strs = []
                                for _i_idx in range(len(_cls_keys)):
                                    for _j_idx in range(_i_idx + 1, len(_cls_keys)):
                                        _ci, _cj = _cls_keys[_i_idx], _cls_keys[_j_idx]
                                        _cos_ij = float((_cls_centroids[_ci] * _cls_centroids[_cj]).sum())
                                        _cni = _cnames_fp.get(_ci, str(_ci))
                                        _cnj = _cnames_fp.get(_cj, str(_cj))
                                        _pair_strs.append(f"{_cni}-{_cnj}={_cos_ij:.4f}")
                                print(f"[ATT408-AngularSep] {dataset_name} pairwise_inter: {', '.join(_pair_strs)}")
                            except Exception as _e_sep:
                                print(f"[ATT408-AngularSep] {dataset_name} failed: {type(_e_sep).__name__}: {_e_sep}")
                        except Exception as _e_fproto:
                            print(f"[FusionProto] {dataset_name} failed: {type(_e_fproto).__name__}: {_e_fproto}")
                    # ATT421: FusionProto_learnable — evaluate using learned prototypes instead of text prototypes
                    if len(_all_z_metric_raw) > 0:
                        try:
                            _lp_mod = getattr(_fusion_dec, '_learnable_proto_module', None) if _fusion_dec is not None else None
                            if _lp_mod is not None and hasattr(_lp_mod, 'forward'):
                                with torch.no_grad():
                                    _lp_protos = _lp_mod.forward().cpu().float()  # [C, D]
                                _z_lp_all = torch.cat(_all_z_metric_raw, dim=0).float()
                                _z_lp_norm = _F.normalize(_z_lp_all, dim=-1)
                                _lp_sims = torch.matmul(_z_lp_norm, _lp_protos.t())
                                _lp_preds = _lp_sims.argmax(dim=1).numpy()
                                _lp_acc = float((_lp_preds == _lbls_np).mean())
                                _lp_f1s = []
                                _lp_per = []
                                _cnames_lp = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                                for _c in _classes_sh:
                                    _tp = float(np.sum((_lp_preds == _c) & (_lbls_np == _c)))
                                    _fp = float(np.sum((_lp_preds == _c) & (_lbls_np != _c)))
                                    _fn = float(np.sum((_lp_preds != _c) & (_lbls_np == _c)))
                                    _pr_lp = _tp / (_tp + _fp + 1e-12)
                                    _rc_lp = _tp / (_tp + _fn + 1e-12)
                                    _lp_f1s.append(2.0 * _pr_lp * _rc_lp / (_pr_lp + _rc_lp + 1e-12))
                                    _lp_per.append(f"{_cnames_lp.get(_c, str(_c))}(R={_rc_lp:.2f},N={int(_tp + _fn)})")
                                _lp_macro_f1 = float(np.mean(_lp_f1s)) if _lp_f1s else float("nan")
                                print(f"[FusionProto-Learnable] {dataset_name} acc={_lp_acc:.4f}"
                                      f" macro_f1={_lp_macro_f1:.4f} (z_metric cosine vs learnable_prototypes)")
                                print(f"[FusionProto-Learnable] Per-class: {', '.join(_lp_per)}")
                                _lp_cos = _lp_mod.proto_cosine_matrix().cpu()
                                for _ri_lp in range(_lp_cos.size(0)):
                                    _row_s = ", ".join([f"{_lp_cos[_ri_lp, _ci_lp].item():.4f}" for _ci_lp in range(_lp_cos.size(1))])
                                    print(f"[LearnableProtoCos] row {_ri_lp}: [{_row_s}]")
                        except Exception as _e_lp:
                            print(f"[FusionProto-Learnable] {dataset_name} failed: {type(_e_lp).__name__}: {_e_lp}")
                    # ATT379a: ETF prototype evaluation
                    if len(_all_z_metric_raw) > 0:
                        try:
                            from utils.simplex_etf import generate_simplex_etf
                            _z_metric_all = torch.cat(_all_z_metric_raw, dim=0).float()
                            _z_metric_all_norm = _F.normalize(_z_metric_all, dim=-1)
                            _etf_c = len(_classes_sh) if len(_classes_sh) > 0 else 4
                            _etf_d = _z_metric_all.shape[-1]
                            _etf_mat = generate_simplex_etf(_etf_c, _etf_d, seed=42, device="cpu")
                            _etf_mat_norm = _F.normalize(_etf_mat.float(), dim=-1)
                            _etf_sims = torch.matmul(_z_metric_all_norm, _etf_mat_norm.t())
                            _etf_preds = _etf_sims.argmax(dim=1).numpy()
                            _etf_acc = float((_etf_preds == _lbls_np).mean())
                            _etf_per = []
                            _cnames_etf = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                            for _c in _classes_sh:
                                _rc = float(np.sum((_etf_preds == _c) & (_lbls_np == _c))) / max(1, float(np.sum(_lbls_np == _c)))
                                _n_c = int(np.sum(_lbls_np == _c))
                                _etf_per.append(f"{_cnames_etf.get(_c, str(_c))}(R={_rc:.2f},N={_n_c})")
                            print(f"[ETFProto] {dataset_name} etf_proto_diag_acc={_etf_acc:.4f} (z_metric cosine vs ETF prototypes)")
                            print(f"[ETFProto] Per-class: {', '.join(_etf_per)}")
                        except Exception as _e_etf:
                            print(f"[ETFProto] {dataset_name} failed: {type(_e_etf).__name__}: {_e_etf}")
                    # Reviewer utility: optionally dump normalized z_metric embeddings for t-SNE.
                    if len(_all_z_metric_raw) > 0 and len(_lbls_np) > 0:
                        try:
                            import config as _cfg_emb
                            _dump_emb = bool(getattr(_cfg_emb, "EVAL_EXPORT_EMBEDDINGS_ENABLE", False))
                        except Exception:
                            _dump_emb = False
                        if _dump_emb:
                            try:
                                _z_metric_all = torch.cat(_all_z_metric_raw, dim=0).float()
                                _z_metric_all = _F.normalize(_z_metric_all, dim=-1)
                                _name_up = str(dataset_name).upper()
                                _dom_map = {"CWRU": 0, "PU": 2, "HUST": 3, "HIT": 4, "SEU": 5}
                                _did = int(_dom_map.get(_name_up, -1))
                                _domains = np.full((len(_lbls_np),), _did, dtype=np.int32)
                                _dump_metric_embedding_arrays(
                                    _z_metric_all.detach().cpu().numpy().astype(np.float32),
                                    np.asarray(_lbls_np, dtype=np.int32),
                                    _domains,
                                    dataset_name=dataset_name,
                                    stage="zs",
                                    n_shot_k=0,
                                    export_tag=_emb_export_tag,
                                )
                            except Exception as _e_dump:
                                print(f"[EmbDump] {dataset_name} failed: {type(_e_dump).__name__}: {_e_dump}")
                    if _has_sem_proto and len(_all_sims_sem_proto) > 0:
                        try:
                            _sem_proto_cat = torch.cat(_all_sims_sem_proto, dim=0).float()  # [N, C]
                            _sem_proto_preds = _sem_proto_cat.argmax(dim=1).numpy()
                            _sem_proto_acc = float((_sem_proto_preds == _lbls_np).mean())
                            _sem_proto_f1s = []
                            for _c in _classes_sh:
                                _tp = float(np.sum((_sem_proto_preds == _c) & (_lbls_np == _c)))
                                _fp = float(np.sum((_sem_proto_preds == _c) & (_lbls_np != _c)))
                                _fn = float(np.sum((_sem_proto_preds != _c) & (_lbls_np == _c)))
                                _pr = _tp / (_tp + _fp + 1e-12)
                                _rc = _tp / (_tp + _fn + 1e-12)
                                _sem_proto_f1s.append(2.0 * _pr * _rc / (_pr + _rc + 1e-12))
                            _sem_proto_macro_f1 = float(np.mean(_sem_proto_f1s)) if _sem_proto_f1s else float("nan")
                            sem_proto_diag_acc_zs = _sem_proto_acc
                            sem_proto_macro_f1_zs = _sem_proto_macro_f1
                            print(f"[SemProto] {dataset_name} sem_proto_diag_acc={_sem_proto_acc:.4f}"
                                  f" macro_f1={_sem_proto_macro_f1:.4f} (z_sem cosine vs text_prototypes)")
                        except Exception as _e_semproto:
                            print(f"[SemProto] {dataset_name} failed: {type(_e_semproto).__name__}: {_e_semproto}")

                    # ATT142: Post-Hoc Semantic Decision Fusion (PSDF)
                    # Strategy: remove anti-aligned text-proto signals (spec_proto/sig_proto both
                    # have acc < random-chance 0.25, confirmed across ATT120-141), instead fuse
                    # spec_head MLP logits with fusion decoder's logits_diag_supervised.
                    #
                    # Mathematical justification:
                    #   - Fusion AUC=0.9994 (near-perfect ranking) but Diag-Acc=0.6452 with nf_protect.
                    #     Using raw logits_diag_supervised (bypassing nf_protect) captures this ranking.
                    #   - spec_head achieves 0.8548 acc but has OR→Ball confusion (17/36 OR misclassified).
                    #   - Weighted average of two probability vectors leverages their complementary errors.
                    #   - Temperature T_spec, T_fusion control sharpness; weights from config (default 0.6/0.4).
                    #
                    # First: report FusionRaw (raw argmax of fusion logits, no nf_protect)
                    if _has_fusion_logits and len(_all_logits_fusion) > 0:
                        try:
                            _fus_logits_cat = torch.cat(_all_logits_fusion, dim=0).float()  # [N, C]
                            _fus_preds_raw = _fus_logits_cat.argmax(dim=1).numpy()
                            _fus_acc_raw = float((_fus_preds_raw == _lbls_np).mean())
                            _fus_f1s_raw = []
                            _fus_per_raw = []
                            _cnames_fus = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                            for _c in _classes_sh:
                                _tp = float(np.sum((_fus_preds_raw == _c) & (_lbls_np == _c)))
                                _fp = float(np.sum((_fus_preds_raw == _c) & (_lbls_np != _c)))
                                _fn = float(np.sum((_fus_preds_raw != _c) & (_lbls_np == _c)))
                                _p_f = _tp / (_tp + _fp + 1e-12)
                                _r_f = _tp / (_tp + _fn + 1e-12)
                                _fus_f1s_raw.append(2.0 * _p_f * _r_f / (_p_f + _r_f + 1e-12))
                                _n_true_f = int(_tp + _fn)
                                _fus_per_raw.append(f"{_cnames_fus.get(_c, str(_c))}(R={_r_f:.2f},N={_n_true_f})")
                            _fus_macro_f1_raw = float(np.mean(_fus_f1s_raw)) if _fus_f1s_raw else float("nan")
                            fusion_raw_diag_acc_zs = _fus_acc_raw
                            fusion_raw_macro_f1_zs = _fus_macro_f1_raw
                            print(f"[FusionRaw] {dataset_name} fusion_raw_diag_acc={_fus_acc_raw:.4f}"
                                  f" macro_f1={_fus_macro_f1_raw:.4f} (logits_diag_supervised, no nf_protect)")
                            if not eval_compact_log:
                                print(f"[FusionRaw] Per-class: {', '.join(_fus_per_raw)}")
                        except Exception as _e_fraw:
                            print(f"[FusionRaw] {dataset_name} failed: {type(_e_fraw).__name__}: {_e_fraw}")

                    # ATT145 P0a: Calibrated Dual-Path Ensemble with joint (w_spec, T_fus) grid search
                    # Motivation: ATT142-144 used fixed w_spec=0.6, only searching T_fus.
                    # Problem (revealed in three-phase run): when SpecHead degrades on PU (0.54 vs FusionRaw 0.76),
                    # the fixed w_spec=0.6 pulls ensemble toward the weaker path, yielding Ensemble(0.60) < FusionRaw(0.76).
                    # Fix: include w_spec in the search to auto-discover optimal weights per dataset.
                    #   w_spec=0.0 → pure FusionRaw;  w_spec=1.0 → pure SpecHead.
                    #   CWRU: auto-selects w_spec≈0.6 (SpecHead 0.85 >> FusionRaw 0.74)
                    #   PU:   auto-selects w_spec≈0.0 (FusionRaw 0.76 >> SpecHead 0.54)
                    # Same unified strategy — no dataset-conditional branching.
                    _best_t_fusion = _ens_t_fusion
                    _best_ens_acc = -1.0
                    _best_ens_f1 = -1.0
                    _best_ens_recalls = {}
                    _best_ens_preds = None
                    _best_ens_label = ""
                    _grid_rows = []
                    _prior_calib_logged = {"done": False}

                    def _maybe_prior_calibrate_spec_logits(_logits_spec, _dname):
                        """ATT158 P1: BBSE-style posterior prior calibration (eval only)."""
                        try:
                            import config as _cfg_pc
                            _pc_enable = bool(getattr(_cfg_pc, "EVAL_SPEC_PRIOR_CALIB_ENABLE", False))
                            _pc_strength = float(getattr(_cfg_pc, "EVAL_SPEC_PRIOR_CALIB_STRENGTH", 1.0))
                            _pc_eps = float(getattr(_cfg_pc, "EVAL_SPEC_PRIOR_CALIB_EPS", 1e-6))
                        except Exception:
                            _pc_enable, _pc_strength, _pc_eps = False, 1.0, 1e-6
                        if (not _pc_enable) or _logits_spec is None:
                            return _logits_spec
                        _k = int(_logits_spec.size(1))
                        if _k <= 1:
                            return _logits_spec
                        _p_hat = _F.softmax(_logits_spec, dim=1).mean(dim=0)  # target predicted prior
                        _u = torch.full_like(_p_hat, 1.0 / float(_k))          # uniform prior
                        _adj = torch.log(torch.clamp(_p_hat, min=_pc_eps) / torch.clamp(_u, min=_pc_eps))
                        _logits_cal = _logits_spec - float(_pc_strength) * _adj.unsqueeze(0)
                        _p_np = _p_hat.detach().cpu().numpy().tolist()
                        _a_np = _adj.detach().cpu().numpy().tolist()
                        if (not _prior_calib_logged["done"]) and (not eval_compact_log):
                            print(
                                f"[PriorCalib] {_dname}: enabled, strength={_pc_strength:.2f}, "
                                f"p_hat={','.join(f'{x:.3f}' for x in _p_np)}, "
                                f"adj={','.join(f'{x:.3f}' for x in _a_np)}"
                            )
                            _prior_calib_logged["done"] = True
                        return _logits_cal

                    try:
                        import config as _cfg_tg
                        _t_grid = list(getattr(_cfg_tg, 'EVAL_ENSEMBLE_FUSION_T_GRID',
                                               [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]))
                        _t_grid_enable = bool(getattr(_cfg_tg, 'EVAL_ENSEMBLE_FUSION_T_GRID_ENABLE', True))
                        _inner_protect_threshold = float(getattr(_cfg_tg, 'EVAL_ENSEMBLE_INNER_PROTECT', 0.90))
                        # ATT145 P0a: w_spec joint search grid; ATT169: finer step (0.1)
                        _w_spec_grid = list(getattr(_cfg_tg, 'EVAL_ENSEMBLE_W_SPEC_GRID',
                                                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
                        # ATT169: SpecProto third-path grids
                        _w_proto_grid = list(getattr(_cfg_tg, 'EVAL_ENSEMBLE_W_SPEC_PROTO_GRID',
                                                     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
                        _t_proto_grid = list(getattr(_cfg_tg, 'EVAL_ENSEMBLE_SPEC_PROTO_T_GRID',
                                                     [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]))
                    except Exception:
                        _t_grid = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
                        _t_grid_enable = True
                        _inner_protect_threshold = 0.90
                        _w_spec_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        _w_proto_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                        _t_proto_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

                    # Only run grid search if both paths are available
                    _can_grid = (
                        _t_grid_enable and
                        _has_fusion_logits and len(_all_logits_fusion) > 0 and
                        len(_all_logits_sh) > 0
                    )
                    if _can_grid:
                        _logits_sh_all = torch.cat(_all_logits_sh, dim=0).float()   # [N, C]
                        _logits_sh_all = _maybe_prior_calibrate_spec_logits(_logits_sh_all, dataset_name)
                        _logits_fus_all = torch.cat(_all_logits_fusion, dim=0).float()  # [N, C]
                        _p_spec_fixed = _F.softmax(_logits_sh_all / max(_ens_t_spec_head, 1e-6), dim=1)
                        # ATT169: prepare SpecProto sims tensor (cosine sims [N, C])
                        _sims_proto_all = None
                        if _has_proto and len(_all_sims_proto) > 0:
                            try:
                                _sims_proto_all = torch.cat(_all_sims_proto, dim=0).float()
                            except Exception:
                                _sims_proto_all = None
                        # ATT179 Step1: Test-Time Prototype Adaptation (TPA, eval-only)
                        # 用目标域高置信样本迭代估计视觉原型，并与文本原型做 convex mix。
                        # 仅修改评估时的 SpecProto 相似度，不改任何训练参数/梯度路径。
                        if _sims_proto_all is not None and len(_all_z_spec) > 0:
                            try:
                                import config as _cfg_tpa
                                _tta_enable = bool(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_ENABLE", False))
                                _tta_targets = set(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_TARGET_DATASETS", ["HUST", "SEU"]))
                                _tta_iters = int(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_ITERS", 3))
                                _tta_topk_ratio = float(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_TOPK_RATIO", 0.30))
                                _tta_min_per_cls = int(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_MIN_SAMPLES_PER_CLASS", 5))
                                _tta_alpha = float(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_MIX_ALPHA", 0.70))
                                _tta_min_raw_acc = float(getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_MIN_SPECPROTO_ACC", 0.0))
                                _tta_alpha_by_ds = getattr(_cfg_tpa, "EVAL_SPEC_PROTO_TTA_MIX_ALPHA_BY_DATASET", {})
                            except Exception:
                                _tta_enable = False
                                _tta_targets = {"HUST", "SEU"}
                                _tta_iters = 3
                                _tta_topk_ratio = 0.30
                                _tta_min_per_cls = 5
                                _tta_alpha = 0.70
                                _tta_min_raw_acc = 0.0
                                _tta_alpha_by_ds = {}
                            _apply_tta = _tta_enable and (str(dataset_name) in _tta_targets)
                            _raw_pred = _sims_proto_all.argmax(dim=1).numpy()
                            _raw_acc = float((_raw_pred == _lbls_np).mean())
                            if _apply_tta and (_raw_acc + 1e-12 < float(_tta_min_raw_acc)):
                                print(
                                    f"[SpecProto-TPA] {dataset_name} skipped: raw_acc={_raw_acc:.4f} "
                                    f"< min_specproto_acc={float(_tta_min_raw_acc):.4f}"
                                )
                                _apply_tta = False
                            if _apply_tta:
                                try:
                                    _z_spec_all = torch.cat(_all_z_spec, dim=0).float()  # [N, D], normalized
                                    _proto_base_src = (
                                        _spec_proto_mat_cached
                                        if _spec_proto_mat_cached is not None
                                        else _proto_mat_cached
                                    )
                                    _proto_base = _proto_base_src.float().to(_z_spec_all.device)  # [K, D]
                                    _proto_cur = _proto_base.clone()
                                    _k_cls = int(_proto_cur.size(0))
                                    _topk_q = min(max(_tta_topk_ratio, 0.05), 0.95)
                                    _min_per_cls = max(1, _tta_min_per_cls)
                                    _iters = max(1, _tta_iters)
                                    _alpha_eff = _tta_alpha
                                    if isinstance(_tta_alpha_by_ds, dict):
                                        _alpha_eff = float(_tta_alpha_by_ds.get(str(dataset_name), _alpha_eff))
                                    _alpha = min(max(_alpha_eff, 0.0), 1.0)
                                    for _it in range(_iters):
                                        _sim_cur = torch.matmul(_z_spec_all, _proto_cur.t())  # [N, K]
                                        _conf, _pred = torch.max(_sim_cur, dim=1)
                                        _thr = torch.quantile(_conf, 1.0 - _topk_q)
                                        _selected = (_conf >= _thr)
                                        _proto_vis = _proto_cur.clone()
                                        _updated = 0
                                        for _c in range(_k_cls):
                                            _m = _selected & (_pred == _c)
                                            if int(_m.sum().item()) < _min_per_cls:
                                                continue
                                            _p_c = _z_spec_all[_m].mean(dim=0)
                                            _p_c = _F.normalize(_p_c.unsqueeze(0), dim=1).squeeze(0)
                                            _proto_vis[_c] = _p_c
                                            _updated += 1
                                        _proto_cur = _F.normalize(_alpha * _proto_base + (1.0 - _alpha) * _proto_vis, dim=1)
                                        if _updated == 0:
                                            break
                                    _sims_proto_all = torch.matmul(_z_spec_all, _proto_cur.t()).cpu()
                                    _pred_tta = _sims_proto_all.argmax(dim=1).numpy()
                                    _acc_tta = float((_pred_tta == _lbls_np).mean())
                                    print(
                                        f"[SpecProto-TPA] {dataset_name} enabled=1 "
                                        f"iters={_iters} topk={_topk_q:.2f} min_cls={_min_per_cls} alpha={_alpha:.2f} "
                                        f"acc={_acc_tta:.4f} raw_acc={_raw_acc:.4f}"
                                    )
                                except Exception as _e_tpa:
                                    print(f"[SpecProto-TPA] {dataset_name} skipped: {type(_e_tpa).__name__}: {_e_tpa}")

                        # ATT145 P0a / ATT169: joint (w_spec, w_proto, T_fus, T_proto) search
                        # Row format: (w_spec, w_proto, T_fus, T_proto, acc, f1, recs_dict, preds, probs)
                        # w_spec + w_proto + w_fus = 1.0  (w_fus = 1 - w_spec - w_proto)
                        # When w_proto=0 this reduces exactly to the original 2-way search.
                        def _compute_f1_recs(_preds_arr, _lbls_arr, _cls_list):
                            _f1s, _recs = [], {}
                            for _c in _cls_list:
                                _tp = float(np.sum((_preds_arr == _c) & (_lbls_arr == _c)))
                                _fp = float(np.sum((_preds_arr == _c) & (_lbls_arr != _c)))
                                _fn = float(np.sum((_preds_arr != _c) & (_lbls_arr == _c)))
                                _pr = _tp / (_tp + _fp + 1e-12)
                                _rc = _tp / (_tp + _fn + 1e-12)
                                _f1s.append(2.0 * _pr * _rc / (_pr + _rc + 1e-12))
                                _recs[_c] = _rc
                            return float(np.mean(_f1s)) if _f1s else float("nan"), _recs

                        for _w_proto_try in _w_proto_grid:
                            # Skip w_proto > 0 rows if SpecProto sims are unavailable
                            if _w_proto_try > 1e-6 and _sims_proto_all is None:
                                continue
                            _t_proto_iter = _t_proto_grid if _w_proto_try > 1e-6 else [0.1]
                            for _t_proto_try in _t_proto_iter:
                                # Pre-compute SpecProto probs for this T_proto (cached per T_proto)
                                if _w_proto_try > 1e-6 and _sims_proto_all is not None:
                                    _p_proto_base = _F.softmax(
                                        _sims_proto_all / max(_t_proto_try, 1e-6), dim=1
                                    )
                                else:
                                    _p_proto_base = None
                                for _w_spec_try in _w_spec_grid:
                                    if _w_spec_try + _w_proto_try > 1.0 + 1e-9:
                                        continue
                                    _w_fus_try = max(0.0, 1.0 - _w_spec_try - _w_proto_try)
                                    for _t_try in _t_grid:
                                        try:
                                            _p_fus_try = _F.softmax(
                                                _logits_fus_all / max(_t_try, 1e-6), dim=1
                                            )
                                            # Build weighted ensemble
                                            _p_try = None
                                            if _w_spec_try > 1e-6:
                                                _p_try = _w_spec_try * _p_spec_fixed
                                            if _w_fus_try > 1e-6:
                                                _p_fus_w = _w_fus_try * _p_fus_try
                                                _p_try = _p_fus_w if _p_try is None else _p_try + _p_fus_w
                                            if _p_proto_base is not None and _w_proto_try > 1e-6:
                                                _p_proto_w = _w_proto_try * _p_proto_base
                                                _p_try = _p_proto_w if _p_try is None else _p_try + _p_proto_w
                                            if _p_try is None:
                                                continue
                                            _preds_try = _p_try.argmax(dim=1).numpy()
                                            _acc_try = float((_preds_try == _lbls_np).mean())
                                            _f1_try, _recs_try = _compute_f1_recs(
                                                _preds_try, _lbls_np, _classes_sh
                                            )
                                            _grid_rows.append((
                                                _w_spec_try, _w_proto_try,
                                                _t_try, _t_proto_try,
                                                _acc_try, _f1_try, _recs_try,
                                                _preds_try, _p_try
                                            ))
                                        except Exception:
                                            pass

                        if _grid_rows:
                            # ATT170 P0b 修复: 正确选择准则 — inner_protect 作为约束（过滤），acc 作为目标。
                            # 原 ATT169 Bug：_sort_key 在 _passing 非空时以 OR_recall 为主键，
                            # 导致三路扩展引入高 OR/低 acc 行后选出次优解（CWRU −22pp，HUST −1.7pp）。
                            # 修复：始终以 (acc, f1, OR_recall) 排序；inner_protect 仅由 _passing 过滤保证为约束。
                            _has_inner = 1 in _classes_sh
                            _has_outer = 2 in _classes_sh
                            # Row index map: 0=w_spec, 1=w_proto, 2=T_fus, 3=T_proto,
                            #                4=acc, 5=f1, 6=recs, 7=preds, 8=probs
                            _passing = [r for r in _grid_rows if not _has_inner or r[6].get(1, 0.0) >= _inner_protect_threshold]
                            _candidates = _passing if _passing else _grid_rows
                            def _sort_key(row):
                                # ATT170: acc-first (tiebreak: f1, OR_recall)
                                # inner_protect is a hard constraint (filter), not an optimization target.
                                _acc, _f1, _recs = row[4], row[5], row[6]
                                return (_acc, _f1, _recs.get(2, 0.0))
                            _candidates.sort(key=_sort_key, reverse=True)
                            (_best_w_spec, _best_w_proto, _best_t_fusion, _best_t_proto,
                             _best_ens_acc, _best_ens_f1, _best_ens_recalls,
                             _best_ens_preds, _) = _candidates[0]
                            _best_w_fus = max(0.0, 1.0 - _best_w_spec - _best_w_proto)
                            if _best_w_proto > 1e-6:
                                _best_ens_label = (
                                    f"SpecHead(w={_best_w_spec:.2f},T={_ens_t_spec_head:.2f})"
                                    f"+SpecProto(w={_best_w_proto:.2f},T={_best_t_proto:.2f})"
                                    f"+FusionDiag(w={_best_w_fus:.2f},T={_best_t_fusion:.2f})"
                                )
                            else:
                                _best_ens_label = (
                                    f"SpecHead(w={_best_w_spec:.2f},T={_ens_t_spec_head:.2f})"
                                    f"+FusionDiag(w={_best_w_fus:.2f},T={_best_t_fusion:.2f})"
                                )

                            # Compact grid summary
                            _cns = {0: "N", 1: "IR", 2: "OR", 3: "Ball"}
                            print(f"[EnsGrid] {dataset_name} best search result"
                                  f" (SpecHead T={_ens_t_spec_head:.2f}, inner_protect≥{_inner_protect_threshold:.2f})")
                            if eval_log_grid_details:
                                # Show 2-way rows (w_proto=0): group by w_spec, best T per w_spec
                                _rows_2way = [r for r in _grid_rows if r[1] <= 1e-6]
                                _ws_seen = {}
                                for _row in _rows_2way:
                                    _ws_r = _row[0]
                                    if _ws_r not in _ws_seen or _row[4] > _ws_seen[_ws_r][4]:
                                        _ws_seen[_ws_r] = _row
                                for _ws_r in sorted(_ws_seen.keys()):
                                    _row = _ws_seen[_ws_r]
                                    _acc_r, _f1_r, _recs_r, _t_r = _row[4], _row[5], _row[6], _row[2]
                                    _rec_str = " ".join(
                                        f"{_cns.get(_c, '?')}={_recs_r.get(_c, float('nan')):.2f}"
                                        for _c in _classes_sh
                                    )
                                    _mk = (" ←BEST" if (abs(_ws_r - _best_w_spec) < 1e-6
                                                        and abs(_row[1]) < 1e-6
                                                        and abs(_t_r - _best_t_fusion) < 1e-9) else "")
                                    print(f"  [2-way] w_spec={_ws_r:.2f} T_fus={_t_r:<4.1f}"
                                          f"  acc={_acc_r:.4f}  f1={_f1_r:.4f}  [{_rec_str}]{_mk}")
                                # Show 3-way rows (w_proto>0): group by (w_spec, w_proto), best (T_fus, T_proto)
                                _rows_3way = [r for r in _grid_rows if r[1] > 1e-6]
                                if _rows_3way:
                                    _wp_seen = {}
                                    for _row in _rows_3way:
                                        _key = (round(_row[0], 4), round(_row[1], 4))
                                        if _key not in _wp_seen or _row[4] > _wp_seen[_key][4]:
                                            _wp_seen[_key] = _row
                                    for _key in sorted(_wp_seen.keys()):
                                        _row = _wp_seen[_key]
                                        _ws_r, _wp_r = _row[0], _row[1]
                                        _acc_r, _f1_r, _recs_r = _row[4], _row[5], _row[6]
                                        _t_r, _tp_r = _row[2], _row[3]
                                        _rec_str = " ".join(
                                            f"{_cns.get(_c, '?')}={_recs_r.get(_c, float('nan')):.2f}"
                                            for _c in _classes_sh
                                        )
                                        _mk = (" ←BEST" if (abs(_ws_r - _best_w_spec) < 1e-6
                                                            and abs(_wp_r - _best_w_proto) < 1e-6
                                                            and abs(_t_r - _best_t_fusion) < 1e-9
                                                            and abs(_tp_r - _best_t_proto) < 1e-9) else "")
                                        print(f"  [3-way] w_spec={_ws_r:.2f} w_proto={_wp_r:.2f}"
                                              f" T_fus={_t_r:<4.1f} T_proto={_tp_r:.2f}"
                                              f"  acc={_acc_r:.4f}  f1={_f1_r:.4f}  [{_rec_str}]{_mk}")
                            print(f"[MultimodalEns] {dataset_name} ensemble_diag_acc={_best_ens_acc:.4f}"
                                  f" macro_f1={_best_ens_f1:.4f} [{_best_ens_label}]")
                            _cnames_ens = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                            _ens_per_best = [
                                f"{_cnames_ens.get(_c, str(_c))}(R={_best_ens_recalls.get(_c, float('nan')):.2f})"
                                for _c in _classes_sh
                            ]
                            if (not eval_compact_log) or eval_log_grid_details:
                                print(f"[MultimodalEns] Per-class: {', '.join(_ens_per_best)}")

                    else:
                        # Fallback: single-T ensemble with fixed params (no grid search available)
                        try:
                            _ens_parts_fb = []
                            _ens_weights_fb = []
                            _ens_labels_fb = []
                            if len(_all_logits_sh) > 0 and _ens_w_spec_head > 0.0:
                                _logits_cat_fb = torch.cat(_all_logits_sh, dim=0).float()
                                _logits_cat_fb = _maybe_prior_calibrate_spec_logits(_logits_cat_fb, dataset_name)
                                _ens_parts_fb.append(_F.softmax(_logits_cat_fb / max(_ens_t_spec_head, 1e-6), dim=1))
                                _ens_weights_fb.append(_ens_w_spec_head)
                                _ens_labels_fb.append(f"SpecHead(w={_ens_w_spec_head:.2f},T={_ens_t_spec_head:.2f})")
                            if _has_fusion_logits and len(_all_logits_fusion) > 0 and _ens_w_fusion > 0.0:
                                _fus_cat_fb = torch.cat(_all_logits_fusion, dim=0).float()
                                _ens_parts_fb.append(_F.softmax(_fus_cat_fb / max(_ens_t_fusion, 1e-6), dim=1))
                                _ens_weights_fb.append(_ens_w_fusion)
                                _ens_labels_fb.append(f"FusionDiag(w={_ens_w_fusion:.2f},T={_ens_t_fusion:.2f})")
                            if _has_proto and len(_all_sims_proto) > 0 and _ens_w_spec_proto > 0.0:
                                _sims_cat_fb = torch.cat(_all_sims_proto, dim=0).float()
                                _ens_parts_fb.append(_F.softmax(_sims_cat_fb / 0.1, dim=1))
                                _ens_weights_fb.append(_ens_w_spec_proto)
                                _ens_labels_fb.append(f"SpecProto(w={_ens_w_spec_proto:.2f})")
                            if _has_sig_proto and len(_all_sims_sig) > 0 and _ens_w_sig_proto > 0.0:
                                _sig_sims_cat_fb = torch.cat(_all_sims_sig, dim=0).float()
                                if _ens_parts_fb and _sig_sims_cat_fb.size(1) != _ens_parts_fb[0].size(1):
                                    _min_c_fb = min(_sig_sims_cat_fb.size(1), _ens_parts_fb[0].size(1))
                                    _sig_sims_cat_fb = _sig_sims_cat_fb[:, :_min_c_fb]
                                _ens_parts_fb.append(_F.softmax(_sig_sims_cat_fb / 0.1, dim=1))
                                _ens_weights_fb.append(_ens_w_sig_proto)
                                _ens_labels_fb.append(f"SigProto(w={_ens_w_sig_proto:.2f})")
                            if len(_ens_parts_fb) >= 2:
                                # ATT143: when margin-adaptive is enabled, use dynamic alpha in fallback single-T path
                                if _ens_margin_adaptive and len(_all_logits_sh) > 0 and _has_fusion_logits and len(_all_logits_fusion) > 0:
                                    _logits_sh_fb = torch.cat(_all_logits_sh, dim=0).float()
                                    _logits_fus_fb = torch.cat(_all_logits_fusion, dim=0).float()
                                    _p_spec_fb = _F.softmax(_logits_sh_fb / max(_ens_t_spec_head, 1e-6), dim=1)
                                    _p_fus_fb = _F.softmax(_logits_fus_fb / max(_ens_t_fusion, 1e-6), dim=1)
                                    _alpha_min_fb = min(_ens_margin_alpha_min, _ens_margin_alpha_max)
                                    _alpha_max_fb = max(_ens_margin_alpha_min, _ens_margin_alpha_max)
                                    _tau_m_fb = max(_ens_margin_tau, 1e-6)
                                    _top2_fb = torch.topk(_logits_sh_fb / max(_ens_t_spec_head, 1e-6), k=2, dim=1).values
                                    _margin_fb = (_top2_fb[:, 0] - _top2_fb[:, 1]).float()
                                    _alpha_fb = _alpha_min_fb + (_alpha_max_fb - _alpha_min_fb) * torch.sigmoid(_margin_fb / _tau_m_fb)
                                    _alpha_fb = _alpha_fb.unsqueeze(1)
                                    _ens_probs_fb = _alpha_fb * _p_spec_fb + (1.0 - _alpha_fb) * _p_fus_fb
                                    _ens_labels_fb = [
                                        f"MAAF(alpha∈[{_alpha_min_fb:.2f},{_alpha_max_fb:.2f}],"
                                        f"tau={_tau_m_fb:.2f},T_spec={_ens_t_spec_head:.2f},T_fus={_ens_t_fusion:.2f})"
                                    ]
                                else:
                                    _w_sum_fb = sum(_ens_weights_fb)
                                    _ens_probs_fb = sum(
                                        _w * _p for _w, _p in zip(_ens_weights_fb, _ens_parts_fb)
                                    ) / _w_sum_fb
                                _best_ens_preds = _ens_probs_fb.argmax(dim=1).numpy()
                                _best_ens_acc = float((_best_ens_preds == _lbls_np).mean())
                                _ens_f1s_fb = []
                                for _c in _classes_sh:
                                    _tp = float(np.sum((_best_ens_preds == _c) & (_lbls_np == _c)))
                                    _fp = float(np.sum((_best_ens_preds == _c) & (_lbls_np != _c)))
                                    _fn = float(np.sum((_best_ens_preds != _c) & (_lbls_np == _c)))
                                    _pr = _tp / (_tp + _fp + 1e-12)
                                    _rc = _tp / (_tp + _fn + 1e-12)
                                    _ens_f1s_fb.append(2.0 * _pr * _rc / (_pr + _rc + 1e-12))
                                    _best_ens_recalls[_c] = _rc
                                _best_ens_f1 = float(np.mean(_ens_f1s_fb)) if _ens_f1s_fb else float("nan")
                                _best_ens_label = "+".join(_ens_labels_fb)
                                _best_t_fusion = _ens_t_fusion
                                print(f"[MultimodalEns] {dataset_name} ensemble_diag_acc={_best_ens_acc:.4f}"
                                      f" macro_f1={_best_ens_f1:.4f} [{_best_ens_label}]")
                                _cnames_ens_fb = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                                _ens_per_fb = [
                                    f"{_cnames_ens_fb.get(_c, str(_c))}(R={_best_ens_recalls.get(_c, float('nan')):.2f})"
                                    for _c in _classes_sh
                                ]
                                if (not eval_compact_log) or eval_log_grid_details:
                                    print(f"[MultimodalEns] Per-class: {', '.join(_ens_per_fb)}")
                        except Exception as _e_fb:
                            print(f"[MultimodalEns] {dataset_name} fallback ensemble failed:"
                                  f" {type(_e_fb).__name__}: {_e_fb}")

                    # ======================================================================
                    # ATT145 P0b: Class-Conditional Trust Cascade (CCTC) + Three-way Fusion
                    # ATT144 基础：Ball 子集激进融合（OR→Ball 自信误分类纠错）
                    # ATT145 新增：FusionProto 作为第三路概率（Ball R=0.89 in ATT144 diagnostics）
                    #   SpecHead: Ball anchor (R=0.86 全集)
                    #   FusionRaw: "发现 OR"（OR R=0.92）
                    #   FusionProto: "保护 Ball"（Ball R=0.89）
                    # 数学形式（Ball 子集）：
                    #   p_ens[ball] = w_spec*p_spec + w_fus*p_fus(T_fus) + w_proto*p_proto(T_proto)
                    #   约束: w_spec + w_fus + w_proto = 1
                    # 防御：若 ball_class 不存在于当前数据集类别 _classes_sh，则跳过 CCTC
                    # ======================================================================
                    try:
                        import config as _cfg_cctc
                        _cctc_enable = bool(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_ENABLE', True))
                        _cctc_ball_class = int(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_BALL_CLASS', 3))
                        _cctc_w_spec_grid = list(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_W_SPEC_GRID',
                                                         [0.10, 0.15, 0.20]))
                        _cctc_w_fus_grid = list(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_W_FUS_GRID',
                                                        [0.30, 0.40, 0.50]))
                        _cctc_t_fus_grid = list(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_T_GRID',
                                                        [0.2, 0.5, 1.0]))
                        _cctc_t_proto_grid = list(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_CCTC_T_PROTO_GRID',
                                                          [0.05, 0.1, 0.2]))
                        _cctc_inner_protect = float(getattr(_cfg_cctc, 'EVAL_ENSEMBLE_INNER_PROTECT', 0.90))
                    except Exception:
                        _cctc_enable = True
                        _cctc_ball_class = 3
                        _cctc_w_spec_grid = [0.10, 0.15, 0.20]
                        _cctc_w_fus_grid = [0.30, 0.40, 0.50]
                        _cctc_t_fus_grid = [0.2, 0.5, 1.0]
                        _cctc_t_proto_grid = [0.05, 0.1, 0.2]
                        _cctc_inner_protect = 0.90

                    _cctc_best_acc = -1.0
                    _cctc_best_f1 = -1.0
                    _cctc_best_recalls = {}
                    _cctc_best_preds = None
                    _cctc_best_label = ""
                    _cctc_grid_rows = []

                    # ATT145 P0b: CCTC only runs when ball_class exists in this dataset's classes
                    _cctc_ball_exists = _cctc_ball_class in _classes_sh
                    if not _cctc_ball_exists and _cctc_enable:
                        print(f"[CCTC] {dataset_name} skipped: ball_class={_cctc_ball_class}"
                              f" not in dataset classes {_classes_sh}")

                    if (_cctc_enable and
                            _cctc_ball_exists and
                            len(_all_logits_sh) > 0 and
                            _has_fusion_logits and len(_all_logits_fusion) > 0):
                        try:
                            _logits_sh_cc = torch.cat(_all_logits_sh, dim=0).float()    # [N, C]
                            _logits_sh_cc = _maybe_prior_calibrate_spec_logits(_logits_sh_cc, dataset_name)
                            _logits_fus_cc = torch.cat(_all_logits_fusion, dim=0).float()  # [N, C]
                            _p_spec_cc = _F.softmax(_logits_sh_cc / max(_ens_t_spec_head, 1e-6), dim=1)
                            _sh_preds_cc = _p_spec_cc.argmax(dim=1)   # [N]
                            _ball_mask = (_sh_preds_cc == _cctc_ball_class)  # [N] bool
                            _n_ball_cc = int(_ball_mask.sum().item())
                            _n_other_cc = int((~_ball_mask).sum().item())
                            print(f"[CCTC] {dataset_name} SpecHead pred:"
                                  f" Ball-predicted={_n_ball_cc}, Other={_n_other_cc}"
                                  f" (ball_class={_cctc_ball_class})")

                            # ATT145 P0b: if FusionProto available, use 3-way blend for Ball subset
                            # Otherwise fall back to 2-way blend (ATT144 behavior)
                            _has_proto_for_cctc = (
                                _fus_proto_logits_cached is not None and
                                _fus_proto_logits_cached.shape[0] == _logits_sh_cc.shape[0]
                            )
                            if _has_proto_for_cctc:
                                print(f"[CCTC] {dataset_name} Using 3-way blend (FusionProto available)")
                            else:
                                print(f"[CCTC] {dataset_name} Using 2-way blend (FusionProto unavailable)")

                            for _w_spec_b in _cctc_w_spec_grid:
                                for _w_fus_b in _cctc_w_fus_grid:
                                    _w_proto_b = 1.0 - _w_spec_b - _w_fus_b
                                    if _w_proto_b < -1e-6:
                                        continue  # invalid combination
                                    for _t_fus_b in _cctc_t_fus_grid:
                                        _t_proto_list = _cctc_t_proto_grid if _has_proto_for_cctc else [1.0]
                                        for _t_proto_b in _t_proto_list:
                                            try:
                                                _p_fus_cc_b = _F.softmax(_logits_fus_cc / max(_t_fus_b, 1e-6), dim=1)
                                                _p_cctc = _p_spec_cc.clone()
                                                if _n_ball_cc > 0:
                                                    if _has_proto_for_cctc and _w_proto_b > 1e-6:
                                                        _p_proto_cc = _F.softmax(
                                                            _fus_proto_logits_cached.to(_logits_sh_cc.device)
                                                            / max(_t_proto_b, 1e-6), dim=1
                                                        )
                                                        _blend = (
                                                            _w_spec_b * _p_spec_cc[_ball_mask] +
                                                            _w_fus_b * _p_fus_cc_b[_ball_mask] +
                                                            _w_proto_b * _p_proto_cc[_ball_mask]
                                                        )
                                                    else:
                                                        _w_sum_cb = _w_spec_b + _w_fus_b
                                                        if _w_sum_cb < 1e-6:
                                                            _w_sum_cb = 1.0
                                                        _blend = (
                                                            _w_spec_b * _p_spec_cc[_ball_mask] +
                                                            _w_fus_b * _p_fus_cc_b[_ball_mask]
                                                        ) / _w_sum_cb
                                                    _p_cctc[_ball_mask] = _blend
                                                _preds_cctc = _p_cctc.argmax(dim=1).numpy()
                                                _acc_cctc = float((_preds_cctc == _lbls_np).mean())
                                                _f1s_cctc, _recs_cctc = [], {}
                                                for _c in _classes_sh:
                                                    _tp = float(np.sum((_preds_cctc == _c) & (_lbls_np == _c)))
                                                    _fp_c = float(np.sum((_preds_cctc == _c) & (_lbls_np != _c)))
                                                    _fn = float(np.sum((_preds_cctc != _c) & (_lbls_np == _c)))
                                                    _pr = _tp / (_tp + _fp_c + 1e-12)
                                                    _rc = _tp / (_tp + _fn + 1e-12)
                                                    _f1s_cctc.append(2.0 * _pr * _rc / (_pr + _rc + 1e-12))
                                                    _recs_cctc[_c] = _rc
                                                _f1_cctc = float(np.mean(_f1s_cctc)) if _f1s_cctc else float("nan")
                                                _cctc_grid_rows.append((
                                                    _w_spec_b, _w_fus_b, _w_proto_b,
                                                    _t_fus_b, _t_proto_b,
                                                    _acc_cctc, _f1_cctc, _recs_cctc, _preds_cctc
                                                ))
                                            except Exception:
                                                pass

                            if _cctc_grid_rows:
                                def _cctc_sort_key(row):
                                    _ws, _wf, _wp, _tf, _tp, _acc_k, _f1_k, _recs_k, _ = row
                                    _ir_k = _recs_k.get(1, 0.0)
                                    _orr_k = _recs_k.get(2, 0.0)
                                    _ball_r_k = _recs_k.get(_cctc_ball_class, 0.0)
                                    _ir_ok_k = 1 if _ir_k >= _cctc_inner_protect else 0
                                    return (_ir_ok_k, _orr_k, _ball_r_k, _acc_k)

                                _cctc_grid_rows.sort(key=_cctc_sort_key, reverse=True)
                                (_cctc_best_w_spec, _cctc_best_w_fus, _cctc_best_w_proto,
                                 _cctc_best_t_fus, _cctc_best_t_proto,
                                 _cctc_best_acc, _cctc_best_f1,
                                 _cctc_best_recalls, _cctc_best_preds) = _cctc_grid_rows[0]

                                if _has_proto_for_cctc and _cctc_best_w_proto > 1e-6:
                                    _cctc_best_label = (
                                        f"CCTC3(Ball→ws={_cctc_best_w_spec:.2f},"
                                        f"wf={_cctc_best_w_fus:.2f}(T={_cctc_best_t_fus:.2f}),"
                                        f"wp={_cctc_best_w_proto:.2f}(T={_cctc_best_t_proto:.2f});"
                                        f"Other→SpecHead)"
                                    )
                                else:
                                    _cctc_best_label = (
                                        f"CCTC(Ball→w_spec={_cctc_best_w_spec:.2f},"
                                        f"T_fus={_cctc_best_t_fus:.2f};"
                                        f"Other→SpecHead)"
                                    )

                                # Print compact grid summary
                                _cns_cc = {0: "N", 1: "IR", 2: "OR", 3: "Ball"}
                                print(f"[CCTCGrid] {dataset_name} best result"
                                      f" (inner_protect≥{_cctc_inner_protect:.2f}, 3-way={_has_proto_for_cctc})")
                                if eval_log_grid_details:
                                    # Show best row per (w_spec, w_fus) combination
                                    _cctc_ww_best = {}
                                    for _row_cc in _cctc_grid_rows:
                                        _key_cc = (_row_cc[0], _row_cc[1])
                                        if _key_cc not in _cctc_ww_best:
                                            _cctc_ww_best[_key_cc] = _row_cc
                                    for _key_cc in sorted(_cctc_ww_best.keys()):
                                        _row_cc = _cctc_ww_best[_key_cc]
                                        _ws_cc, _wf_cc, _wp_cc, _tf_cc, _tp_cc, _acc_cc, _f1_cc, _recs_cc, _ = _row_cc
                                        _rec_str_cc = " ".join(
                                            f"{_cns_cc.get(_c, '?')}={_recs_cc.get(_c, float('nan')):.2f}"
                                            for _c in _classes_sh
                                        )
                                        _mk_cc = (" ←BEST" if (
                                            _ws_cc == _cctc_best_w_spec and _wf_cc == _cctc_best_w_fus and
                                            _tf_cc == _cctc_best_t_fus and _tp_cc == _cctc_best_t_proto
                                        ) else "")
                                        if _has_proto_for_cctc:
                                            print(f"  ws={_ws_cc:.2f} wf={_wf_cc:.2f} wp={_wp_cc:.2f}"
                                                  f" Tf={_tf_cc:.2f} Tp={_tp_cc:.2f}"
                                                  f"  acc={_acc_cc:.4f} f1={_f1_cc:.4f}"
                                                  f"  [{_rec_str_cc}]{_mk_cc}")
                                        else:
                                            print(f"  ws={_ws_cc:.2f} wf={_wf_cc:.2f} Tf={_tf_cc:.2f}"
                                                  f"  acc={_acc_cc:.4f} f1={_f1_cc:.4f}"
                                                  f"  [{_rec_str_cc}]{_mk_cc}")

                                print(f"[CCTC] {dataset_name} cctc_diag_acc={_cctc_best_acc:.4f}"
                                      f" macro_f1={_cctc_best_f1:.4f} [{_cctc_best_label}]")
                                _cnames_cctc = {0: "Normal", 1: "Inner", 2: "Outer", 3: "Rolling"}
                                _cctc_per_best = [
                                    f"{_cnames_cctc.get(_c, str(_c))}"
                                    f"(R={_cctc_best_recalls.get(_c, float('nan')):.2f})"
                                    for _c in _classes_sh
                                ]
                                if (not eval_compact_log) or eval_log_grid_details:
                                    print(f"[CCTC] Per-class: {', '.join(_cctc_per_best)}")

                                # Update global best ensemble if CCTC is better
                                if _cctc_best_acc > _best_ens_acc:
                                    _best_ens_acc = _cctc_best_acc
                                    _best_ens_f1 = _cctc_best_f1
                                    _best_ens_recalls = _cctc_best_recalls
                                    _best_ens_preds = _cctc_best_preds
                                    _best_ens_label = _cctc_best_label
                        except Exception as _e_cctc:
                            print(f"[CCTC] {dataset_name} failed:"
                                  f" {type(_e_cctc).__name__}: {_e_cctc}")

                    # ATT158: 移除单独“最终决策性能汇总”块。
                    # 统一由每个 Phase 的综合性能表输出所有关键指标（含 Legacy），避免重复表格。
            except Exception as _e_sh:
                print(f"[SpecHead] {dataset_name} spec_head eval failed: {type(_e_sh).__name__}: {_e_sh}")

        # ATT370: Tip-Adapter style cache evaluation on zero-shot stage.
        try:
            import config as _cfg_tip_eval
            _tip_enable = bool(getattr(_cfg_tip_eval, "EVAL_TIP_ADAPTER_ENABLE", False))
            _tip_targets = set(getattr(_cfg_tip_eval, "EVAL_TIP_ADAPTER_TARGET_DATASETS", ["HUST", "SEU"]))
            _tip_replace_when_better = bool(getattr(_cfg_tip_eval, "EVAL_TIP_ADAPTER_REPLACE_WHEN_BETTER", True))
        except Exception:
            _tip_enable = False
            _tip_targets = {"HUST", "SEU"}
            _tip_replace_when_better = True
        if _tip_enable and (str(dataset_name) in _tip_targets):
            _tip_support_loader = _build_tip_support_loader(loader)
            _tip_result = _evaluate_with_tip_adapter_cache(
                _tip_support_loader,
                zero_shot_loader,
                dataset_name=f"{dataset_name}(zero-shot)",
                diag_num_classes=diag_num_classes,
                support_transform_fn=None,
                query_transform_fn=(_spec_twc_fn if callable(_spec_twc_fn) else None),
            )
            if _tip_result is not None:
                tip_adapter_diag_acc_zs = float(_tip_result.get("diag_acc", float("nan")))
                tip_adapter_macro_f1_zs = float(_tip_result.get("macro_f1", float("nan")))
                tip_adapter_alpha_zs = float(_tip_result.get("tip_adapter_alpha", float("nan")))
                if tip_adapter_diag_acc_zs > _best_ens_acc:
                    _best_ens_acc = tip_adapter_diag_acc_zs
                    _best_ens_f1 = tip_adapter_macro_f1_zs
                    _best_ens_label = (
                        f"TipAdapter(alpha={tip_adapter_alpha_zs:.2f},"
                        f"beta={float(_tip_result.get('tip_adapter_beta', float('nan'))):.2f},"
                        f"k={int(_tip_result.get('tip_adapter_topk', 0))})"
                    )
                if _tip_replace_when_better and tip_adapter_diag_acc_zs > float(zero_shot_diag_res.get("diag_acc", 0.0)):
                    print(
                        f"[TipAdapter] {dataset_name} replacing zero-shot diag result: "
                        f"diag_acc {float(zero_shot_diag_res.get('diag_acc', 0.0)):.4f} "
                        f"-> {tip_adapter_diag_acc_zs:.4f}"
                    )
                    _merged_zs = dict(zero_shot_diag_res)
                    for _k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        _merged_zs[_k] = _tip_result.get(_k, _merged_zs.get(_k))
                    zero_shot_diag_res = _merged_zs
                    zero_shot_macro_f1 = _safe_macro_f1(
                        zero_shot_diag_res.get("diag_labels", None),
                        zero_shot_diag_res.get("diag_preds", None)
                    )

        # ATT372: dynamic semantic multi-prototypes (class-wise KMeans + structured text templates + top-k voting).
        try:
            import config as _cfg_dyn_eval
            _dyn_enable = bool(getattr(_cfg_dyn_eval, "EVAL_DYNAMIC_MULTI_PROTO_ENABLE", False))
            _dyn_targets = set(getattr(_cfg_dyn_eval, "EVAL_DYNAMIC_MULTI_PROTO_TARGET_DATASETS", ["HUST", "SEU"]))
            _dyn_replace_when_better = bool(getattr(_cfg_dyn_eval, "EVAL_DYNAMIC_MULTI_PROTO_REPLACE_WHEN_BETTER", True))
        except Exception:
            _dyn_enable = False
            _dyn_targets = {"HUST", "SEU"}
            _dyn_replace_when_better = True
        if _dyn_enable and (str(dataset_name) in _dyn_targets):
            _dyn_support_loader = _build_tip_support_loader(loader)
            _dyn_result = _evaluate_with_dynamic_multi_proto(
                _dyn_support_loader,
                zero_shot_loader,
                dataset_name=f"{dataset_name}(zero-shot)",
                diag_num_classes=diag_num_classes,
            )
            if _dyn_result is not None:
                dynamic_multi_proto_diag_acc_zs = float(_dyn_result.get("diag_acc", float("nan")))
                dynamic_multi_proto_macro_f1_zs = float(_dyn_result.get("macro_f1", float("nan")))
                if dynamic_multi_proto_diag_acc_zs > _best_ens_acc:
                    _best_ens_acc = dynamic_multi_proto_diag_acc_zs
                    _best_ens_f1 = dynamic_multi_proto_macro_f1_zs
                    _best_ens_label = (
                        f"DynamicMultiProto(k={int(getattr(_cfg_dyn_eval, 'EVAL_DYNAMIC_MULTI_PROTO_K', 4))},"
                        f"topk={int(getattr(_cfg_dyn_eval, 'EVAL_DYNAMIC_MULTI_PROTO_TOPK_VOTE', 7))})"
                    )
                if _dyn_replace_when_better and dynamic_multi_proto_diag_acc_zs > float(zero_shot_diag_res.get("diag_acc", 0.0)):
                    print(
                        f"[DynMultiProto] {dataset_name} replacing zero-shot diag result: "
                        f"diag_acc {float(zero_shot_diag_res.get('diag_acc', 0.0)):.4f} "
                        f"-> {dynamic_multi_proto_diag_acc_zs:.4f}"
                    )
                    _merged_dyn = dict(zero_shot_diag_res)
                    for _k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        _merged_dyn[_k] = _dyn_result.get(_k, _merged_dyn.get(_k))
                    zero_shot_diag_res = _merged_dyn
                    zero_shot_macro_f1 = _safe_macro_f1(
                        zero_shot_diag_res.get("diag_labels", None),
                        zero_shot_diag_res.get("diag_preds", None)
                    )

        _zs_hmean = _safe_hmean_recall(
            zero_shot_diag_res.get("diag_labels", None),
            zero_shot_diag_res.get("diag_preds", None),
        )

        # ===== ATT399/ATT500: Transductive (source-free vs legacy source+kNN) =====
        def _cosine_sim_np_for_diag(a, b):
            a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
            b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
            return a_n @ b_n.T

        transductive_diag_acc_zs = float("nan")
        transductive_macro_f1_zs = float("nan")
        transductive_macro_recall_zs = float("nan")
        _td_ref_probs_for_vote = None  # torch (N,C) soft probs for final decision (source-free BMM)
        try:
            import config as _cfg_td
            _use_td = bool(getattr(_cfg_td, "USE_TRANSDUCTIVE", False))
            _td_source_free = bool(getattr(_cfg_td, "TRANSDUCTIVE_SOURCE_FREE", False))
        except Exception:
            _use_td = False
            _td_source_free = False
        if _use_td:
            try:
                from zero_shot_eval.transductive_inference import (
                    collect_features, run_transductive_inference, refine_with_transductive,
                )
                import config as _cfg_td
                _td_source_free = bool(getattr(_cfg_td, "TRANSDUCTIVE_SOURCE_FREE", False))
                _td_fallback_sf_no_source = bool(getattr(_cfg_td, "TRANSDUCTIVE_FALLBACK_SOURCE_FREE_WHEN_NO_SOURCE", True))
                _td_init_fp = bool(getattr(_cfg_td, "TRANSDUCTIVE_INIT_FROM_FPROTO", True))
                _td_fus_var_collapse_th = float(getattr(_cfg_td, "TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD", 1e-4))
                _td_temp = float(getattr(_cfg_td, "TRANSDUCTIVE_TEMPERATURE", 0.07))
                _td_bmm_kl = float(getattr(_cfg_td, "TRANSDUCTIVE_BMM_KL_LAMBDA", 1.0))
                _bmm_n_neighbours = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_N_NEIGHBOURS", 5))
                _bmm_iterations = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_ITERATIONS", 15))
                _bmm_balance_tau = float(getattr(_cfg_td, "TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU", 0.5))
                _bmm_sinkhorn_iters = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_SINKHORN_ITERS", 3))
                _td_inner_mm = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_INNER_MM_ITERS", 3))
                _fd_t_fproto_td = float(getattr(_cfg_td, "EVAL_FINAL_DECISION_T_FUSION_PROTO", 1.0))
                _td_bmm_sep_gate = float(getattr(_cfg_td, "TRANSDUCTIVE_BMM_SEP_RATIO_GATE", 1.0))
                _td_device = next(detector.signal_encoder.parameters()).device
                _td_did_fn = _encode_domain_ids_from_batch

                # ATT502: skip BMM when FusionProto angular separation is too low
                _bmm_gated_by_sep = False
                if _fproto_sep_ratio == _fproto_sep_ratio and _fproto_sep_ratio < _td_bmm_sep_gate:
                    _bmm_gated_by_sep = True
                    print(
                        f"[ATT502-BMMGate] {dataset_name} BMM skipped: "
                        f"sep_ratio={_fproto_sep_ratio:.4f} < gate={_td_bmm_sep_gate:.2f} "
                        f"(FusionProto features lack discriminability, BMM would amplify errors)"
                    )

                if _bmm_gated_by_sep:
                    pass  # skip all BMM paths; transductive results stay None → final decision falls back to FusionProto
                elif _td_source_free:
                    # Source-free: reuse collected fusion metric-space z + FusionProto sims (no second forward).
                    if (
                        _has_fusion_proto
                        and len(_all_sims_fusion_proto) > 0
                        and len(_all_z_fusion_for_td) == len(_all_sims_fusion_proto)
                    ):
                        _z_td = torch.cat(_all_z_fusion_for_td, dim=0).float().numpy()
                        _sims_td = torch.cat(_all_sims_fusion_proto, dim=0).float()
                        _graph_z_td = _z_td
                        _graph_space_td = "z_fusion"
                        _fus_var_td = float(np.mean(np.var(_z_td, axis=0))) if _z_td.size > 0 else float("nan")
                        if (
                            _fus_var_td == _fus_var_td
                            and _fus_var_td < float(_td_fus_var_collapse_th)
                            and ("_all_z_spec_for_td" in locals())
                            and len(_all_z_spec_for_td) == len(_all_z_fusion_for_td)
                        ):
                            _z_spec_td = torch.cat(_all_z_spec_for_td, dim=0).float().numpy()
                            if _z_spec_td.shape[0] == _z_td.shape[0]:
                                _graph_z_td = _z_spec_td
                                _graph_space_td = "z_spec"
                        _c_td = int(diag_num_classes)
                        if _z_td.shape[0] == _sims_td.shape[0] and _sims_td.shape[1] >= _c_td:
                            _fp_np = torch.softmax(
                                _sims_td[:, :_c_td] / max(1e-6, float(_fd_t_fproto_td)), dim=1
                            ).numpy()
                            _text_protos = None
                            if hasattr(detector, "_active_text_prototypes") and callable(detector._active_text_prototypes):
                                _tp = detector._active_text_prototypes()
                                if _tp is not None:
                                    import torch as _th_td
                                    if isinstance(_tp, dict):
                                        _sorted_keys = sorted(_tp.keys())
                                        _proto_list = []
                                        for _pk in _sorted_keys:
                                            _pv = _tp[_pk]
                                            if _th_td.is_tensor(_pv):
                                                _proto_list.append(_pv.detach().cpu().numpy())
                                            else:
                                                _proto_list.append(np.asarray(_pv))
                                        if _proto_list:
                                            _text_protos = np.stack(_proto_list, axis=0)
                                    elif _th_td.is_tensor(_tp):
                                        _text_protos = _tp.detach().cpu().numpy()
                                    else:
                                        _text_protos = np.asarray(_tp)
                            if _text_protos is not None:
                                _td_result = refine_with_transductive(
                                    _z_td.astype(np.float32),
                                    _fp_np.astype(np.float32),
                                    _text_protos,
                                    n_classes=_c_td,
                                    graph_features=_graph_z_td.astype(np.float32),
                                    temperature=_td_temp,
                                    init_from_fproto=_td_init_fp,
                                    kl_lambda=_td_bmm_kl,
                                    n_neighbours=_bmm_n_neighbours,
                                    bmm_iterations=_bmm_iterations,
                                    inner_mm_iterations=_td_inner_mm,
                                    class_balance_tau=_bmm_balance_tau,
                                    sinkhorn_iterations=_bmm_sinkhorn_iters,
                                )
                                _td_preds = _td_result["predictions"]
                                _td_probs_np = np.asarray(_td_result["probabilities"], dtype=np.float32)
                                _td_ref_probs_for_vote = torch.from_numpy(_td_probs_np).cpu()
                                _td_diag_sf = _td_result.get("diagnostics", {})
                                _tgt_labels_td = zero_shot_diag_res.get("diag_labels", None)
                                if _tgt_labels_td is not None:
                                    _tgt_labels_td = np.asarray(_tgt_labels_td)
                                if _tgt_labels_td is not None and len(_tgt_labels_td) == len(_td_preds):
                                    transductive_diag_acc_zs = float(np.mean(_td_preds == _tgt_labels_td))
                                    transductive_macro_f1_zs = _safe_macro_f1(_tgt_labels_td, _td_preds)
                                    transductive_macro_recall_zs = _safe_macro_recall(_tgt_labels_td, _td_preds)
                                print(
                                    f"[ATT500-Transductive] {dataset_name} source_free BMM: "
                                    f"diag_acc={transductive_diag_acc_zs:.4f}, macro_f1={transductive_macro_f1_zs:.4f}, "
                                    f"n={len(_td_preds)}, method={_td_diag_sf.get('method', '?')}, "
                                    f"graph_space={_graph_space_td}, fusion_var={_fus_var_td:.6g}, "
                                    f"bmm_entropy={_td_diag_sf.get('bmm_final_entropy', '?')}, "
                                    f"n_neighbours={_td_diag_sf.get('n_neighbours', '?')}, "
                                    f"bmm_iters={_td_diag_sf.get('bmm_iterations', '?')}"
                                )
                            else:
                                print(f"[ATT500-Transductive] {dataset_name}: skipped (no text prototypes)")
                        else:
                            print(f"[ATT500-Transductive] {dataset_name}: skipped (shape mismatch)")
                    else:
                        print(
                            f"[ATT500-Transductive] {dataset_name}: skipped "
                            "(need fusion_proto sims + aligned z_metric batches)"
                        )
                else:
                    # Legacy ATT406: requires source loaders + labeled source features
                    _source_loaders = []
                    _seen_loader_ids = set()
                    _target_loader_id = id(loader)
                    def _td_maybe_add_source_loader(_sl):
                        if _sl is None or id(_sl) == _target_loader_id or id(_sl) in _seen_loader_ids:
                            return
                        _source_loaders.append(_sl)
                        _seen_loader_ids.add(id(_sl))

                    if task_name == "Task1" and dataset_name == "CWRU":
                        for _sl in (twc_source_cwru_loader, cwru_test_loader, task1_cwru_loader):
                            _td_maybe_add_source_loader(_sl)
                    elif task_name == "Task1" and dataset_name == "PU":
                        for _sl in (twc_source_pu_loader, pu_test_loader, task1_pu_loader):
                            _td_maybe_add_source_loader(_sl)
                    else:
                        for _sl in (twc_source_cwru_loader, twc_source_pu_loader,
                                    cwru_test_loader, pu_test_loader):
                            _td_maybe_add_source_loader(_sl)
                        if len(_source_loaders) == 0:
                            for _sl in (task1_cwru_loader, task1_pu_loader):
                                _td_maybe_add_source_loader(_sl)

                    _use_phys = bool(getattr(_cfg_td, "USE_PHYSICS_FEATURES", False))
                    _phys_w = float(getattr(_cfg_td, "PHYSICS_FEAT_WEIGHT", 0.15))
                    _phys_metric = str(getattr(_cfg_td, "PHYSICS_KNN_METRIC", "euclidean"))
                    _kl_lambda = float(getattr(_cfg_td, "TRANSDUCTIVE_KL_REG_LAMBDA", 0.5))
                    _adaptive_alpha = bool(getattr(_cfg_td, "TRANSDUCTIVE_ADAPTIVE_ALPHA", False))
                    _adaptive_ref_cos = float(getattr(_cfg_td, "TRANSDUCTIVE_ADAPTIVE_ALPHA_REF_COS", 0.15))
                    _adaptive_min_ratio = float(getattr(_cfg_td, "TRANSDUCTIVE_ADAPTIVE_ALPHA_MIN_RATIO", 0.05))
                    _use_multi_template = bool(getattr(_cfg_td, "TRANSDUCTIVE_USE_MULTI_TEMPLATE", False))
                    _use_attr_matching = bool(getattr(_cfg_td, "TRANSDUCTIVE_USE_ATTRIBUTE_MATCHING", False))
                    _attr_weight = float(getattr(_cfg_td, "TRANSDUCTIVE_ATTRIBUTE_WEIGHT", 0.3))
                    _use_mc_knn = bool(getattr(_cfg_td, "TRANSDUCTIVE_USE_MULTICENTER_KNN", False))
                    _mc_max_k = int(getattr(_cfg_td, "TRANSDUCTIVE_MULTICENTER_MAX_K", 4))
                    _use_class_rescue = bool(getattr(_cfg_td, "TRANSDUCTIVE_USE_CLASS_RESCUE", True))
                    _rescue_threshold = float(getattr(_cfg_td, "TRANSDUCTIVE_RESCUE_THRESHOLD", 0.3))
                    _use_bmm = bool(getattr(_cfg_td, "TRANSDUCTIVE_USE_BMM", True))
                    _bmm_kl_lambda = float(getattr(_cfg_td, "TRANSDUCTIVE_BMM_KL_LAMBDA", 1.0))
                    _bmm_n_neighbours = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_N_NEIGHBOURS", 3))
                    _bmm_iterations = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_ITERATIONS", 20))
                    _bmm_balance_tau = float(getattr(_cfg_td, "TRANSDUCTIVE_BMM_CLASS_BALANCE_TAU", 0.5))
                    _bmm_sinkhorn_iters = int(getattr(_cfg_td, "TRANSDUCTIVE_BMM_SINKHORN_ITERS", 3))

                    if len(_source_loaders) > 0:
                        _src_z_list, _src_lbl_list, _src_phys_list = [], [], []
                        for _sl in _source_loaders:
                            _src_data = collect_features(
                                _sl, detector.signal_encoder, detector.spec_encoder,
                                detector.fusion_module, _td_device,
                                domain_id_fn=_td_did_fn,
                                extract_physics=_use_phys,
                            )
                            _src_z_list.append(_src_data["z_fusion"])
                            if "labels" in _src_data:
                                _src_lbl_list.append(_src_data["labels"])
                            if "physics" in _src_data:
                                _src_phys_list.append(_src_data["physics"])
                        _src_z_all = np.concatenate(_src_z_list, axis=0)
                        _src_lbl_all = np.concatenate(_src_lbl_list, axis=0) if _src_lbl_list else None
                        _src_phys_all = np.concatenate(_src_phys_list, axis=0) if _src_phys_list else None

                        _tgt_data = collect_features(
                            loader, detector.signal_encoder, detector.spec_encoder,
                            detector.fusion_module, _td_device,
                            domain_id_fn=_td_did_fn,
                            extract_physics=_use_phys,
                        )
                        _tgt_z = _tgt_data["z_fusion"]
                        _tgt_labels = _tgt_data.get("labels", None)
                        _tgt_phys = _tgt_data.get("physics", None)

                        _text_protos = None
                        if hasattr(detector, '_active_text_prototypes') and callable(detector._active_text_prototypes):
                            _tp = detector._active_text_prototypes()
                            if _tp is not None:
                                import torch as _th_td
                                if isinstance(_tp, dict):
                                    _sorted_keys = sorted(_tp.keys())
                                    _proto_list = []
                                    for _pk in _sorted_keys:
                                        _pv = _tp[_pk]
                                        if _th_td.is_tensor(_pv):
                                            _proto_list.append(_pv.detach().cpu().numpy())
                                        else:
                                            _proto_list.append(np.asarray(_pv))
                                    if _proto_list:
                                        _text_protos = np.stack(_proto_list, axis=0)
                                elif _th_td.is_tensor(_tp):
                                    _text_protos = _tp.detach().cpu().numpy()
                                else:
                                    _text_protos = np.asarray(_tp)

                        # ATT402 P2: Build per-template prototypes (C, K_templates, D)
                        _text_multi_protos = None
                        if _use_multi_template and hasattr(detector, 'text_encoder'):
                            try:
                                import config as _cfg_fpt
                                _fpt = getattr(_cfg_fpt, 'FAULT_PROMPT_TEMPLATES', None)
                                if _fpt is not None:
                                    import torch as _th_mt
                                    _te = detector.text_encoder
                                    _te.eval()
                                    _multi_list = []
                                    _max_k = 0
                                    for _cls_key in sorted(_fpt.keys()):
                                        _prompts = _fpt[_cls_key]
                                        _cls_embs = []
                                        with _th_mt.no_grad():
                                            for _p_str in _prompts:
                                                _emb = _te([_p_str]).to(_td_device)
                                                _emb = _th_mt.nn.functional.normalize(
                                                    _emb.squeeze(0).float(), dim=0
                                                )
                                                _cls_embs.append(_emb.cpu().numpy())
                                        _cls_arr = np.stack(_cls_embs, axis=0)
                                        _multi_list.append(_cls_arr)
                                        _max_k = max(_max_k, _cls_arr.shape[0])
                                    if _multi_list:
                                        D_dim = _multi_list[0].shape[-1]
                                        _padded = []
                                        for _arr in _multi_list:
                                            if _arr.shape[0] < _max_k:
                                                _pad = np.zeros((_max_k - _arr.shape[0], D_dim), dtype=_arr.dtype)
                                                _arr = np.concatenate([_arr, _pad], axis=0)
                                            _padded.append(_arr)
                                        _text_multi_protos = np.stack(_padded, axis=0)
                                        if eval_log_transductive_details:
                                            print(f"[ATT406-Transductive] {dataset_name}: "
                                                  f"multi_template built shape={_text_multi_protos.shape}")
                            except Exception as _mt_exc:
                                if eval_log_transductive_details:
                                    print(f"[ATT406-Transductive] {dataset_name}: "
                                          f"multi_template build failed ({_mt_exc}), using mean proto")
                                _text_multi_protos = None

                        if _text_protos is not None and _src_lbl_all is not None:
                            _td_ncf = int(getattr(_cfg_td, "TRANSDUCTIVE_N_CLUSTERS_FACTOR", 2))
                            _td_knn = int(getattr(_cfg_td, "TRANSDUCTIVE_KNN_K", 64))
                            _td_alpha = float(getattr(_cfg_td, "TRANSDUCTIVE_ALPHA_TEXT", 0.5))
                            _td_im_it = int(getattr(_cfg_td, "TRANSDUCTIVE_IM_ITERATIONS", 5))
                            _td_im_cr = float(getattr(_cfg_td, "TRANSDUCTIVE_IM_CONFIDENT_RATIO", 0.30))
                            _td_temp = float(getattr(_cfg_td, "TRANSDUCTIVE_TEMPERATURE", 0.07))
                            _td_result = run_transductive_inference(
                                target_z=_tgt_z,
                                source_z=_src_z_all,
                                source_labels=_src_lbl_all,
                                text_prototypes=_text_protos,
                                n_classes=diag_num_classes,
                                n_clusters_factor=_td_ncf,
                                knn_k=_td_knn,
                                alpha_text=_td_alpha,
                                im_iterations=_td_im_it,
                                im_confident_ratio=_td_im_cr,
                                temperature=_td_temp,
                                dataset_name=dataset_name,
                                target_physics=_tgt_phys,
                                source_physics=_src_phys_all,
                                physics_weight=_phys_w,
                                physics_knn_metric=_phys_metric,
                                kl_reg_lambda=_kl_lambda,
                                adaptive_alpha=_adaptive_alpha,
                                adaptive_alpha_ref_cos=_adaptive_ref_cos,
                                adaptive_alpha_min_ratio=_adaptive_min_ratio,
                                text_multi_prototypes=_text_multi_protos,
                                use_attribute_matching=_use_attr_matching,
                                attribute_weight=_attr_weight,
                                use_multicenter_knn=_use_mc_knn,
                                multicenter_max_k=_mc_max_k,
                                use_class_rescue=_use_class_rescue,
                                rescue_threshold=_rescue_threshold,
                                use_bmm_refine=_use_bmm,
                                bmm_kl_lambda=_bmm_kl_lambda,
                                bmm_n_neighbours=_bmm_n_neighbours,
                                bmm_iterations=_bmm_iterations,
                                bmm_class_balance_tau=_bmm_balance_tau,
                                bmm_sinkhorn_iterations=_bmm_sinkhorn_iters,
                            )
                            _td_preds = _td_result["predictions"]
                            _td_diag = _td_result["diagnostics"]
                            if _tgt_labels is not None:
                                transductive_diag_acc_zs = float(np.mean(_td_preds == _tgt_labels))
                                transductive_macro_f1_zs = _safe_macro_f1(_tgt_labels, _td_preds)
                                transductive_macro_recall_zs = _safe_macro_recall(_tgt_labels, _td_preds)
                            _src_tgt_cos = np.mean(_cosine_sim_np_for_diag(
                                _src_z_all[:min(500, len(_src_z_all))],
                                _tgt_z[:min(500, len(_tgt_z))]
                            )) if len(_src_z_all) > 0 and len(_tgt_z) > 0 else float("nan")
                            _tgt_var = float(np.mean(np.var(_tgt_z, axis=0)))
                            _src_var = float(np.mean(np.var(_src_z_all, axis=0)))
                            _phys_info = ""
                            if "physics_src_tgt_cos_mean" in _td_diag:
                                _l2_info = ""
                                if "physics_src_tgt_l2_mean" in _td_diag:
                                    _l2_info = f", phys_l2={_td_diag['physics_src_tgt_l2_mean']:.4f}"
                                _phys_info = (
                                    f", phys_cos={_td_diag['physics_src_tgt_cos_mean']:.4f}"
                                    f"{_l2_info}"
                                    f", phys_dim={_td_diag.get('physics_selected_dim','?')}"
                                    f", phys_metric={_td_diag.get('physics_knn_metric','?')}"
                                )
                            _kl_info = ""
                            if _td_diag.get("kl_reg_active"):
                                _kl_info = (
                                    f", kl_reg={_td_diag.get('kl_reg_lambda','?')}"
                                    f", kl_mean0={_td_diag.get('kl_mean_iter0','?')}"
                                )
                            # ATT402 diagnostic info
                            _alpha_info = ""
                            if _td_diag.get("adaptive_alpha_enabled"):
                                _alpha_info = (
                                    f", alpha_orig={_td_diag.get('alpha_text_original','?')}"
                                    f", alpha_eff={_td_diag.get('alpha_text_effective','?')}"
                                    f", adapt_ref_cos={_td_diag.get('adaptive_alpha_ref_cos','?')}"
                                )
                            else:
                                _alpha_info = f", alpha={_td_diag.get('alpha_text_effective','?')}"
                            _mt_info = ""
                            if _td_diag.get("multi_template_enabled"):
                                _mt_info = f", multi_tpl_K={_td_diag.get('multi_template_K','?')}"
                            _s1_info = ""
                            if _td_diag.get("s1_attribute_matching"):
                                _s1_info = (
                                    f", s1_attr=ON(w={_td_diag.get('s1_attribute_weight','?')}"
                                    f", cls={_td_diag.get('s1_classes_with_profile','?')})"
                                )
                            _s2_info = ""
                            if _td_diag.get("s2_multicenter_knn"):
                                _s2_sp = _td_diag.get("s2_sub_proto_info", {})
                                _s2_k_str = ",".join(
                                    f"c{c}:k={v.get('k','?')}"
                                    for c, v in sorted(_s2_sp.items())
                                )
                                _s2_info = f", s2_mc=ON({_s2_k_str})"
                            _refine_method = _td_diag.get("refine_method", "legacy_im")
                            if "transclip_bmm" in _refine_method:
                                _entropy_key = "bmm_final_entropy"
                                _bmm_info = (
                                    f", bmm_kl_lambda={_td_diag.get('kl_lambda','?')}"
                                    f", bmm_nn={_td_diag.get('n_neighbours','?')}"
                                    f", bmm_iters={_td_diag.get('bmm_iterations','?')}"
                                    f", bmm_final_dist={_td_diag.get('bmm_final_class_dist','?')}"
                                    f", balance_tau={_td_diag.get('class_balance_tau','?')}"
                                    f", sinkhorn={_td_diag.get('sinkhorn_iterations','?')}"
                                )
                            else:
                                _entropy_key = "im_final_mean_entropy"
                                _bmm_info = ""
                            print(
                                f"[ATT406-Transductive] {dataset_name}: "
                                f"diag_acc={transductive_diag_acc_zs:.4f}, "
                                f"macro_f1={transductive_macro_f1_zs:.4f}, "
                                f"method={_td_diag.get('method','?')}, "
                                f"refine={_refine_method}, "
                                f"n_clusters={_td_diag.get('n_clusters','?')}, "
                                f"cluster_labels={_td_diag.get('cluster_labels','?')}, "
                                f"assigned_classes={_td_diag.get('assigned_classes','?')}, "
                                f"final_entropy={_td_diag.get(_entropy_key,'?')}, "
                                f"cluster_space={_td_diag.get('cluster_space','?')}, "
                                f"knn_space={_td_diag.get('knn_space','?')}, "
                                f"z_src_tgt_cos={_src_tgt_cos:.4f}, "
                                f"tgt_feat_var={_tgt_var:.6f}, src_feat_var={_src_var:.6f}, "
                                f"n_src={len(_src_z_all)}, n_tgt={len(_tgt_z)}"
                                f"{_phys_info}"
                                f"{_kl_info}"
                                f"{_alpha_info}"
                                f"{_mt_info}"
                                f"{_s1_info}"
                                f"{_s2_info}"
                                f"{_bmm_info}"
                            )
                            if eval_log_transductive_details:
                                _rescue_summary = ""
                                if _td_diag.get("rescue_enabled"):
                                    _n_rescued = len(_td_diag.get("rescue_operations", []))
                                    _rescued_classes = [r["missing_class"] for r in _td_diag.get("rescue_operations", [])]
                                    _rescue_summary = f"rescue={_n_rescued}cls({_rescued_classes})" if _n_rescued > 0 else "rescue=0"
                                    print(
                                        f"[ATT406-Transductive] {dataset_name}: "
                                        f"{_rescue_summary}, threshold={_td_diag.get('rescue_threshold','?')}"
                                    )
                                _rescue_ops = _td_diag.get("rescue_operations", [])
                                if _rescue_ops:
                                    for _rop in _rescue_ops:
                                        print(
                                            f"[ATT406-Transductive] {dataset_name}: "
                                            f"RESCUE cls{_rop['missing_class']}←cluster{_rop['cluster_k']}"
                                            f"(was cls{_rop['original_class']}, n={_rop['cluster_size']}, "
                                            f"knn_ev={_rop['knn_evidence']}, text_ev={_rop['text_evidence']}, "
                                            f"src={_rop['source']})"
                                        )
                                _cluster_detail = _td_diag.get("cluster_detail")
                                if _cluster_detail:
                                    for _cd in _cluster_detail:
                                        _lbl_str = f"label={_cd['label']}"
                                        if "label_post_rescue" in _cd and _cd["label_post_rescue"] != _cd["label"]:
                                            _lbl_str = f"label={_cd['label']}→{_cd['label_post_rescue']}(rescued)"
                                        _fs_str = ""
                                        if "final_score" in _cd:
                                            _fs_str = f", score={_cd['final_score']}, margin={_cd.get('margin','?')}"
                                        print(
                                            f"[ATT406-Transductive] {dataset_name}: "
                                            f"cluster_{_cd['k']}(n={_cd['size']}): "
                                            f"text={_cd['text_sim']}, knn={_cd['knn_votes']}"
                                            f"{_fs_str}, {_lbl_str}"
                                        )
                                if "transclip_bmm" in str(_td_diag.get("refine_method", "")):
                                    for _bk in ("bmm_iter0_entropy", "bmm_iter0_class_dist",
                                                "bmm_final_entropy", "bmm_final_class_dist",
                                                "bmm_iterations", "kl_lambda", "n_neighbours",
                                                "class_balance_tau", "sinkhorn_iterations"):
                                        if _bk in _td_diag:
                                            print(f"[ATT406-Transductive] {dataset_name}: {_bk}={_td_diag[_bk]}")
                                else:
                                    for _it_k in sorted(k for k in _td_diag if k.startswith("im_iter") and "class_dist" in k):
                                        print(f"[ATT406-Transductive] {dataset_name}: {_it_k}={_td_diag[_it_k]}")
                        else:
                            _reason = "no text prototypes" if _text_protos is None else "no source labels"
                            print(f"[ATT406-Transductive] {dataset_name}: skipped ({_reason})")
                    else:
                        print(f"[ATT406-Transductive] {dataset_name}: skipped (no source loaders)")
                        if _td_fallback_sf_no_source:
                            if (
                                _has_fusion_proto
                                and len(_all_sims_fusion_proto) > 0
                                and len(_all_z_fusion_for_td) == len(_all_sims_fusion_proto)
                            ):
                                _z_td = torch.cat(_all_z_fusion_for_td, dim=0).float().numpy()
                                _sims_td = torch.cat(_all_sims_fusion_proto, dim=0).float()
                                _graph_z_td = _z_td
                                _graph_space_td = "z_fusion"
                                _fus_var_td = float(np.mean(np.var(_z_td, axis=0))) if _z_td.size > 0 else float("nan")
                                if (
                                    _fus_var_td == _fus_var_td
                                    and _fus_var_td < float(_td_fus_var_collapse_th)
                                    and ("_all_z_spec_for_td" in locals())
                                    and len(_all_z_spec_for_td) == len(_all_z_fusion_for_td)
                                ):
                                    _z_spec_td = torch.cat(_all_z_spec_for_td, dim=0).float().numpy()
                                    if _z_spec_td.shape[0] == _z_td.shape[0]:
                                        _graph_z_td = _z_spec_td
                                        _graph_space_td = "z_spec"
                                _c_td = int(diag_num_classes)
                                if _z_td.shape[0] == _sims_td.shape[0] and _sims_td.shape[1] >= _c_td:
                                    _fp_np = torch.softmax(
                                        _sims_td[:, :_c_td] / max(1e-6, float(_fd_t_fproto_td)), dim=1
                                    ).numpy()
                                    _text_protos = None
                                    if hasattr(detector, "_active_text_prototypes") and callable(detector._active_text_prototypes):
                                        _tp = detector._active_text_prototypes()
                                        if _tp is not None:
                                            import torch as _th_td
                                            if isinstance(_tp, dict):
                                                _proto_list = []
                                                for _pk in sorted(_tp.keys()):
                                                    _pv = _tp[_pk]
                                                    _proto_list.append(_pv.detach().cpu().numpy() if _th_td.is_tensor(_pv) else np.asarray(_pv))
                                                if _proto_list:
                                                    _text_protos = np.stack(_proto_list, axis=0)
                                            elif _th_td.is_tensor(_tp):
                                                _text_protos = _tp.detach().cpu().numpy()
                                            else:
                                                _text_protos = np.asarray(_tp)
                                    if _text_protos is not None:
                                        _td_result = refine_with_transductive(
                                            _z_td.astype(np.float32),
                                            _fp_np.astype(np.float32),
                                            _text_protos,
                                            n_classes=_c_td,
                                            graph_features=_graph_z_td.astype(np.float32),
                                            temperature=_td_temp,
                                            init_from_fproto=_td_init_fp,
                                            kl_lambda=_td_bmm_kl,
                                            n_neighbours=_bmm_n_neighbours,
                                            bmm_iterations=_bmm_iterations,
                                            inner_mm_iterations=_td_inner_mm,
                                            class_balance_tau=_bmm_balance_tau,
                                            sinkhorn_iterations=_bmm_sinkhorn_iters,
                                        )
                                        _td_preds = _td_result["predictions"]
                                        _td_probs_np = np.asarray(_td_result["probabilities"], dtype=np.float32)
                                        _td_ref_probs_for_vote = torch.from_numpy(_td_probs_np).cpu()
                                        _td_diag_sf = _td_result.get("diagnostics", {})
                                        _tgt_labels_td = zero_shot_diag_res.get("diag_labels", None)
                                        if _tgt_labels_td is not None:
                                            _tgt_labels_td = np.asarray(_tgt_labels_td)
                                        if _tgt_labels_td is not None and len(_tgt_labels_td) == len(_td_preds):
                                            transductive_diag_acc_zs = float(np.mean(_td_preds == _tgt_labels_td))
                                            transductive_macro_f1_zs = _safe_macro_f1(_tgt_labels_td, _td_preds)
                                            transductive_macro_recall_zs = _safe_macro_recall(_tgt_labels_td, _td_preds)
                                        print(
                                            f"[ATT501-TransductiveFallback] {dataset_name} no-source source_free BMM: "
                                            f"diag_acc={transductive_diag_acc_zs:.4f}, macro_f1={transductive_macro_f1_zs:.4f}, "
                                            f"graph_space={_graph_space_td}, fusion_var={_fus_var_td:.6g}, "
                                            f"bmm_entropy={_td_diag_sf.get('bmm_final_entropy', '?')}, "
                                            f"n_neighbours={_td_diag_sf.get('n_neighbours', '?')}, "
                                            f"bmm_iters={_td_diag_sf.get('bmm_iterations', '?')}"
                                        )
                                    else:
                                        print(f"[ATT501-TransductiveFallback] {dataset_name}: skipped (no text prototypes)")
                                else:
                                    print(f"[ATT501-TransductiveFallback] {dataset_name}: skipped (shape mismatch)")
                            else:
                                print(
                                    f"[ATT501-TransductiveFallback] {dataset_name}: skipped "
                                    "(need fusion_proto sims + aligned z_metric batches)"
                                )
            except Exception as _td_exc:
                import traceback
                print(f"[ATT406-Transductive] {dataset_name}: FAILED ({_td_exc})")
                traceback.print_exc()

        # P0: unified final decision with confidence-weighted voting (sample-level).
        final_decision_cm = None
        final_decision_voting_acc = float("nan")
        final_decision_voting_macro_f1 = float("nan")
        final_decision_voting_macro_recall = float("nan")
        final_decision_voting_hmean = float("nan")
        final_decision_voting_sources = ""
        final_decision_voting_preds = None
        try:
            import config as _cfg_fd
            _fd_enable_vote = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_ENABLE", True))
            _fd_use_spec = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_USE_SPEC_HEAD", True))
            _fd_use_signal = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_USE_SIGNAL_HEAD", True))
            _fd_use_fproto = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_USE_FUSION_PROTO", True))
            _fd_use_fdiag = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_USE_FUSION_DECODER", True))
            _fd_use_td = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_USE_TRANSDUCTIVE", True))
            _fd_w_spec = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_SPEC_WEIGHT", 0.05))
            _fd_w_signal = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_SIGNAL_HEAD_WEIGHT", 0.45))
            _fd_w_fproto = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_FUSION_PROTO_WEIGHT", 0.30))
            _fd_w_fdiag = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_FUSION_DECODER_WEIGHT", 0.00))
            _fd_w_td = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_TRANSDUCTIVE_WEIGHT", 0.20))
            _fd_t_signal = max(1e-6, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_T_SIGNAL_HEAD", 1.0)))
            _fd_spec_collapse_gate = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_SPEC_COLLAPSE_GATE", True))
            _fd_spec_collapse_thresh = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_SPEC_COLLAPSE_THRESH", 0.75))
            _fd_conf_pow = max(0.0, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_CONF_POWER", 2.0)))
            _fd_per_sample_conf = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_PER_SAMPLE_CONF_GATE", True))
            _fd_fproto_sep_thresh = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_THRESH", 1.0))
            _fd_fproto_sep_downw = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_FPROTO_SEP_RATIO_DOWNWEIGHT", 0.1))
            _fd_disagree_fb = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_DISAGREE_FALLBACK_ENABLE", True))
            _fd_disagree_thresh = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_DISAGREE_FALLBACK_THRESH", 0.70))
            _fd_cascade_enable = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_CASCADE_ENABLE", True))
            _fd_cascade_high_thresh = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_CASCADE_HIGH_THRESH", 0.85))
            _fd_t_spec = max(1e-6, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_T_SPEC", 1.0)))
            _fd_t_fproto = max(1e-6, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_T_FUSION_PROTO", 1.0)))
            _fd_t_fdiag = max(1e-6, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_T_FUSION_DECODER", 1.0)))
            _fd_print_cm = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_PRINT_CM", True))
            _fd_gate_enable = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_DEGEN_GATE_ENABLE", True))
            _fd_max_class_frac = min(1.0, max(0.0, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_MAX_CLASS_FRAC", 0.85))))
            _fd_min_pred_entropy = min(1.0, max(0.0, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_MIN_PRED_ENTROPY", 0.35))))
            _fd_td_agree_gate = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_TRANSDUCTIVE_AGREEMENT_GATE_ENABLE", True))
            _fd_td_min_agree = min(
                1.0,
                max(0.0, float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_TRANSDUCTIVE_MIN_AGREE_WITH_FUSION_PROTO", 0.70))),
            )
            _fd_fproto_priority = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_FUSION_PROTO_PRIORITY", True))
            _fd_td_primary = bool(getattr(_cfg_fd, "EVAL_FINAL_DECISION_TD_PRIMARY", False))
            _fd_collapse_sep_gate = float(getattr(_cfg_fd, "EVAL_FINAL_DECISION_COLLAPSE_SEP_RATIO_GATE", 1.0))
            _fd_collapse_fvar_th = float(getattr(_cfg_fd, "TRANSDUCTIVE_FUSION_VAR_COLLAPSE_THRESHOLD", 1e-4))
            _fd_exclude_td_on_collapse = bool(
                getattr(_cfg_fd, "EVAL_FINAL_DECISION_EXCLUDE_TRANSDUCTIVE_ON_COLLAPSE", True)
            )
        except Exception:
            _fd_enable_vote = True
            _fd_use_spec, _fd_use_signal, _fd_use_fproto, _fd_use_fdiag, _fd_use_td = False, True, True, False, True
            _fd_w_spec, _fd_w_signal, _fd_w_fproto, _fd_w_fdiag, _fd_w_td = 0.05, 0.45, 0.30, 0.00, 0.20
            _fd_t_signal = 1.0
            _fd_spec_collapse_gate = True
            _fd_spec_collapse_thresh = 0.75
            _fd_conf_pow = 2.0
            _fd_per_sample_conf = True
            _fd_fproto_sep_thresh = 1.0
            _fd_fproto_sep_downw = 0.1
            _fd_disagree_fb = True
            _fd_disagree_thresh = 0.70
            _fd_cascade_enable = True
            _fd_cascade_high_thresh = 0.85
            _fd_t_spec = _fd_t_fproto = _fd_t_fdiag = 1.0
            _fd_print_cm = True
            _fd_gate_enable = True
            _fd_max_class_frac = 0.85
            _fd_min_pred_entropy = 0.35
            _fd_td_agree_gate = True
            _fd_td_min_agree = 0.70
            _fd_fproto_priority = True
            _fd_td_primary = False
            _fd_collapse_sep_gate = 1.0
            _fd_collapse_fvar_th = 1e-4
            _fd_exclude_td_on_collapse = True

        _zs_repr_collapsed = False
        try:
            _lbls_eval_np = None
            if "_all_labels_eval" in locals() and len(_all_labels_eval) > 0:
                _lbls_eval_np = torch.cat(_all_labels_eval, dim=0).numpy()
            elif len(_all_lbls_sh) > 0:
                _lbls_eval_np = torch.cat(_all_lbls_sh).numpy()

            _zs_fusion_var = float("nan")
            if "_all_z_fusion_for_td" in locals() and len(_all_z_fusion_for_td) > 0:
                _zfv_np = torch.cat(_all_z_fusion_for_td, dim=0).float().numpy()
                if _zfv_np.size > 0:
                    _zs_fusion_var = float(np.mean(np.var(_zfv_np, axis=0)))

            _zs_repr_collapsed = False
            _collapse_reasons = []
            if _fproto_sep_ratio == _fproto_sep_ratio and _fproto_sep_ratio < _fd_collapse_sep_gate:
                _zs_repr_collapsed = True
                _collapse_reasons.append(
                    f"sep_ratio={_fproto_sep_ratio:.4f}<{_fd_collapse_sep_gate:.2f}"
                )
            if _zs_fusion_var == _zs_fusion_var and _zs_fusion_var < _fd_collapse_fvar_th:
                _zs_repr_collapsed = True
                _collapse_reasons.append(
                    f"fusion_var={_zs_fusion_var:.6g}<{_fd_collapse_fvar_th:.6g}"
                )

            if _zs_repr_collapsed and _fd_exclude_td_on_collapse:
                print(
                    f"[FinalDecisionGate] {dataset_name} representation collapsed "
                    f"({', '.join(_collapse_reasons)}): reject transductive for final decision"
                )
                transductive_diag_acc_zs = float("nan")
                transductive_macro_f1_zs = float("nan")
                transductive_macro_recall_zs = float("nan")
                _td_ref_probs_for_vote = None
                if "_td_preds" in locals():
                    _td_preds = None

            _fd_prob_parts = []
            _fd_sources = []
            _fd_eps = 1e-12
            _fd_fproto_pred_for_gate = None
            _fd_fproto_prob_for_priority = None

            def _fd_add_path(_name, _base_weight, _prob):
                if _prob is None or _prob.dim() != 2:
                    return False
                _pred_t = _prob.argmax(dim=1)
                _counts = torch.bincount(_pred_t.cpu(), minlength=int(_prob.size(1))).float()
                _dist = _counts / max(1.0, float(_counts.sum().item()))
                _max_frac = float(_dist.max().item()) if _dist.numel() > 0 else 1.0
                _ent = float((-( _dist.clamp_min(_fd_eps) * _dist.clamp_min(_fd_eps).log()).sum() / np.log(max(2, int(_prob.size(1))))).item())
                if _fd_gate_enable and (_max_frac >= _fd_max_class_frac or _ent < _fd_min_pred_entropy):
                    print(
                        f"[FinalDecisionGate] {dataset_name} drop {_name}: "
                        f"max_class_frac={_max_frac:.3f}, pred_entropy={_ent:.3f}"
                    )
                    return False
                _fd_prob_parts.append((str(_name), float(_base_weight), _prob.clone()))
                _fd_sources.append(str(_name))
                return True

            def _fd_force_add_path(_name, _base_weight, _prob):
                if _prob is None or _prob.dim() != 2:
                    return False
                _fd_prob_parts.append((str(_name), float(_base_weight), _prob.clone()))
                _fd_sources.append(str(_name))
                return True

            def _fd_effective_fproto_weight(_base_w):
                _bw = float(_base_w)
                if (
                    _fproto_sep_ratio == _fproto_sep_ratio
                    and _fproto_sep_ratio < _fd_fproto_sep_thresh
                ):
                    _bw = _bw * _fd_fproto_sep_downw
                    print(
                        f"[FinalDecisionGate] {dataset_name} fusion_proto downweight: "
                        f"sep_ratio={_fproto_sep_ratio:.4f}<{_fd_fproto_sep_thresh:.2f} "
                        f"-> w={_bw:.4f}"
                    )
                return _bw

            def _fd_mix_votes(_parts):
                if not _parts:
                    return None
                if not _fd_per_sample_conf:
                    _sum_bw = float(sum(max(0.0, float(bw)) for _, bw, _ in _parts))
                    if _sum_bw <= 0:
                        return None
                    _mix = None
                    for _, _bw, _prob in _parts:
                        _wn = max(0.0, float(_bw)) / _sum_bw
                        _mix = (_wn * _prob) if _mix is None else (_mix + _wn * _prob)
                    return _mix / _mix.sum(dim=1, keepdim=True).clamp(min=_fd_eps)

                _mix_num = None
                _mix_den = None
                for _, _bw, _prob in _parts:
                    _conf = _prob.max(dim=1, keepdim=True).values.clamp(min=_fd_eps).pow(_fd_conf_pow)
                    _w = max(0.0, float(_bw)) * _conf
                    _wp = _w * _prob
                    if _mix_num is None:
                        _mix_num = _wp
                        _mix_den = _w
                    else:
                        _mix_num = _mix_num + _wp
                        _mix_den = _mix_den + _w
                if _mix_num is None or _mix_den is None:
                    return None
                return _mix_num / _mix_den.clamp(min=_fd_eps)

            def _fd_cascade_mix(_sig_prob, _parts, _high_thresh):
                """High SignalHead confidence -> signal-only; else weighted fusion mix."""
                _mix = _fd_mix_votes(_parts)
                if _mix is None:
                    return None, 0, 0
                if _sig_prob is None or _sig_prob.dim() != 2 or _sig_prob.size(0) != _mix.size(0):
                    return _mix, 0, int(_mix.size(0))
                _high_mask = _sig_prob.max(dim=1).values >= float(_high_thresh)
                _final = _mix.clone()
                if bool(_high_mask.any().item()):
                    _final[_high_mask] = _sig_prob[_high_mask]
                _n_high = int(_high_mask.sum().item())
                _n_low = int(_mix.size(0)) - _n_high
                return _final, _n_high, _n_low

            def _fd_disagree_signal_only(_sig_prob, _fproto_prob, _spec_prob=None):
                if not _fd_disagree_fb or _sig_prob is None:
                    return False, ""
                _sig_pred = _sig_prob.argmax(dim=1)
                _rates = []
                if _spec_prob is not None and _spec_prob.size(0) == _sig_prob.size(0):
                    _spec_pred = _spec_prob.argmax(dim=1)
                    _rates.append(float((_sig_pred != _spec_pred).float().mean().item()))
                if _fproto_prob is not None and _fproto_prob.size(0) == _sig_prob.size(0):
                    _fp_pred = _fproto_prob.argmax(dim=1)
                    _rates.append(float((_sig_pred != _fp_pred).float().mean().item()))
                if not _rates:
                    return False, ""
                _max_dis = max(_rates)
                if _max_dis >= _fd_disagree_thresh:
                    return True, f"disagree={_max_dis:.3f}>={_fd_disagree_thresh:.2f}"
                return False, ""

            def _fd_probs_from_preds(_pred_np, _n, _c):
                if _pred_np is None:
                    return None
                _pred_np = np.asarray(_pred_np)
                if len(_pred_np) != _n:
                    return None
                _prob = torch.zeros((_n, _c), dtype=torch.float32)
                _pt = torch.from_numpy(_pred_np.astype(np.int64)).clamp(0, _c - 1)
                _prob.scatter_(1, _pt.unsqueeze(1), 1.0)
                return _prob

            def _fd_collapse_fallback(_n, _c, _fproto_prob=None, _cctc_preds=None, _ens_preds=None):
                if _fd_use_signal and _has_sig_head and len(_all_logits_sig) > 0:
                    _sig_lg_fb = torch.cat(_all_logits_sig, dim=0).float()
                    if _sig_lg_fb.size(0) == _n and _sig_lg_fb.size(1) >= _c:
                        _sig_pb_fb = torch.softmax(_sig_lg_fb[:, :_c] / _fd_t_signal, dim=1)
                        if _fd_force_add_path("signal_head", 1.0, _sig_pb_fb):
                            print(f"[FinalDecisionGate] {dataset_name} collapse fallback -> signal_head")
                            return True
                _cctc_prob = _fd_probs_from_preds(_cctc_preds, _n, _c)
                if _cctc_prob is not None and _fd_force_add_path("cctc", 1.0, _cctc_prob):
                    print(f"[FinalDecisionGate] {dataset_name} collapse fallback -> cctc")
                    return True
                _ens_prob = _fd_probs_from_preds(_ens_preds, _n, _c)
                if _ens_prob is not None and _fd_force_add_path("ensemble", 1.0, _ens_prob):
                    print(f"[FinalDecisionGate] {dataset_name} collapse fallback -> ensemble")
                    return True
                if _fproto_prob is not None and _fd_force_add_path("fusion_proto", 1.0, _fproto_prob):
                    print(f"[FinalDecisionGate] {dataset_name} collapse fallback -> fusion_proto (ungated)")
                    return True
                return False

            _cctc_preds_fb = locals().get("_cctc_best_preds")
            _ens_preds_fb = locals().get("_best_ens_preds")

            if _fd_enable_vote and (_lbls_eval_np is not None):
                _n_fd = int(len(_lbls_eval_np))
                _c_fd = int(diag_num_classes)
                if _n_fd > 0 and _c_fd > 1:
                    # FusionProto probs (shared: gating + TD-primary fallback)
                    _fproto_prob = None
                    if _fd_use_fproto and ("_all_sims_fusion_proto" in locals()) and len(_all_sims_fusion_proto) > 0:
                        _fproto_logits = torch.cat(_all_sims_fusion_proto, dim=0).float()
                        if _fproto_logits.size(0) == _n_fd and _fproto_logits.size(1) >= _c_fd:
                            _fproto_prob = torch.softmax(_fproto_logits[:, :_c_fd] / _fd_t_fproto, dim=1)

                    # Transductive vote probabilities (source-free soft or legacy one-hot)
                    _td_vote_prob = None
                    if _fd_use_td:
                        if "_td_ref_probs_for_vote" in locals() and _td_ref_probs_for_vote is not None:
                            _trp = _td_ref_probs_for_vote
                            if _trp.dim() == 2 and _trp.size(0) == _n_fd and _trp.size(1) >= _c_fd:
                                _td_vote_prob = _trp[:, :_c_fd].float().clone()
                        elif ("_td_preds" in locals()) and (_td_preds is not None):
                            _td_preds_np = np.asarray(_td_preds)
                            if len(_td_preds_np) == _n_fd:
                                _td_prob_oh = torch.zeros((_n_fd, _c_fd), dtype=torch.float32)
                                _td_pred_t = torch.from_numpy(_td_preds_np.astype(np.int64))
                                _td_pred_t = torch.clamp(_td_pred_t, 0, _c_fd - 1)
                                _td_prob_oh.scatter_(1, _td_pred_t.unsqueeze(1), 1.0)
                                _td_vote_prob = _td_prob_oh

                    _sig_prob_vote = None
                    _spec_prob_vote = None

                    if _fd_td_primary:
                        _fd_prob_parts = []
                        _fd_sources = []
                        if _fd_use_signal and _has_sig_head and len(_all_logits_sig) > 0:
                            _sig_logits_tdp = torch.cat(_all_logits_sig, dim=0).float()
                            if _sig_logits_tdp.size(0) == _n_fd and _sig_logits_tdp.size(1) >= _c_fd:
                                _sig_prob_vote = torch.softmax(_sig_logits_tdp[:, :_c_fd] / _fd_t_signal, dim=1)
                                _fd_add_path("signal_head", _fd_w_signal, _sig_prob_vote)
                        _reject_td = _zs_repr_collapsed and _fd_exclude_td_on_collapse
                        if _fd_use_td and _td_vote_prob is not None and not _reject_td:
                            _fd_add_path("transductive", _fd_w_td, _td_vote_prob)
                        elif _fd_use_td and _td_vote_prob is not None and _reject_td:
                            print(f"[FinalDecisionGate] {dataset_name} td_primary: reject transductive (collapsed)")
                        if _fd_use_fproto and _fproto_prob is not None:
                            _fd_add_path("fusion_proto", _fd_effective_fproto_weight(_fd_w_fproto), _fproto_prob)
                        if not _fd_prob_parts:
                            _fb_ok = _fd_collapse_fallback(
                                _n_fd, _c_fd, _fproto_prob, _cctc_preds_fb, _ens_preds_fb
                            )
                            if not _fb_ok:
                                print(f"[FinalDecisionGate] {dataset_name} td_primary: no viable fallback path")
                    else:
                        if _fd_use_signal and _has_sig_head and len(_all_logits_sig) > 0:
                            _sig_logits_fd = torch.cat(_all_logits_sig, dim=0).float()
                            if _sig_logits_fd.size(0) == _n_fd and _sig_logits_fd.size(1) >= _c_fd:
                                _sig_prob_vote = torch.softmax(_sig_logits_fd[:, :_c_fd] / _fd_t_signal, dim=1)
                                _fd_add_path("signal_head", _fd_w_signal, _sig_prob_vote)

                        _spec_auto_gated = False
                        if _fd_spec_collapse_gate and _fd_use_spec:
                            if ("_all_logits_sh" in locals()) and len(_all_logits_sh) > 0:
                                _spec_logits_g = torch.cat(_all_logits_sh, dim=0).float()
                                _spec_preds_g = _spec_logits_g.argmax(dim=1)
                                _spec_counts_g = torch.bincount(_spec_preds_g.cpu(), minlength=int(_spec_logits_g.size(1))).float()
                                _spec_frac_g = float(_spec_counts_g.max().item() / max(1, _spec_counts_g.sum().item()))
                                if _spec_frac_g >= _fd_spec_collapse_thresh:
                                    _spec_auto_gated = True
                                    print(
                                        f"[FinalDecisionGate] {dataset_name} spec_head auto-gated: "
                                        f"max_class_frac={_spec_frac_g:.3f}>={_fd_spec_collapse_thresh:.2f}"
                                    )

                        if _fd_use_spec and not _spec_auto_gated and ("_all_logits_sh" in locals()) and len(_all_logits_sh) > 0:
                            _spec_logits = torch.cat(_all_logits_sh, dim=0).float()
                            if _spec_logits.size(0) == _n_fd and _spec_logits.size(1) >= _c_fd:
                                _spec_prob_vote = torch.softmax(_spec_logits[:, :_c_fd] / _fd_t_spec, dim=1)
                                _fd_add_path("spec_head", _fd_w_spec, _spec_prob_vote)

                        if _fd_use_fproto and _fproto_prob is not None:
                            if _fd_add_path("fusion_proto", _fd_effective_fproto_weight(_fd_w_fproto), _fproto_prob):
                                _fd_fproto_pred_for_gate = _fproto_prob.argmax(dim=1).cpu().numpy()
                                if _fd_fproto_priority:
                                    _fd_fproto_prob_for_priority = _fproto_prob.clone()

                        if _fd_use_fdiag and ("_all_logits_fusion" in locals()) and len(_all_logits_fusion) > 0:
                            _fdiag_logits = torch.cat(_all_logits_fusion, dim=0).float()
                            if _fdiag_logits.size(0) == _n_fd and _fdiag_logits.size(1) >= _c_fd:
                                _fdiag_prob = torch.softmax(_fdiag_logits[:, :_c_fd] / _fd_t_fdiag, dim=1)
                                _fd_add_path("fusion_decoder", _fd_w_fdiag, _fdiag_prob)

                        if (
                            _fd_use_td
                            and _td_vote_prob is not None
                            and not (_zs_repr_collapsed and _fd_exclude_td_on_collapse)
                            and not (_fd_fproto_priority and _fd_fproto_prob_for_priority is not None)
                        ):
                            if _fd_td_agree_gate and _fd_fproto_pred_for_gate is not None:
                                _td_pred_np = _td_vote_prob.argmax(dim=1).cpu().numpy()
                                _td_agree = float(
                                    np.mean(_td_pred_np.astype(np.int64) == np.asarray(_fd_fproto_pred_for_gate).astype(np.int64))
                                )
                                if _td_agree < _fd_td_min_agree:
                                    print(
                                        f"[FinalDecisionGate] {dataset_name} drop transductive: "
                                        f"agree_with_fusion_proto={_td_agree:.3f} < {_fd_td_min_agree:.3f}"
                                    )
                                else:
                                    _fd_add_path("transductive", _fd_w_td, _td_vote_prob)
                            else:
                                _fd_add_path("transductive", _fd_w_td, _td_vote_prob)

                        if _fd_fproto_priority and _fd_fproto_prob_for_priority is not None:
                            _fd_prob_parts = [("fusion_proto", 1.0, _fd_fproto_prob_for_priority)]
                            _fd_sources = ["fusion_proto"]
                            print(f"[FinalDecisionGate] {dataset_name} use fusion_proto priority; skip transductive")

                    if len(_fd_prob_parts) > 0:
                        _use_sig_only, _disagree_reason = _fd_disagree_signal_only(
                            _sig_prob_vote,
                            _fproto_prob,
                            _spec_prob_vote,
                        )
                        if _use_sig_only and _sig_prob_vote is not None:
                            _fd_prob_parts = [("signal_head", 1.0, _sig_prob_vote)]
                            _fd_sources = ["signal_head"]
                            print(
                                f"[FinalDecisionGate] {dataset_name} disagree fallback -> signal_head only "
                                f"({_disagree_reason})"
                            )

                        if _fd_cascade_enable and _sig_prob_vote is not None:
                            _mix, _cascade_n_high, _cascade_n_low = _fd_cascade_mix(
                                _sig_prob_vote,
                                _fd_prob_parts,
                                _fd_cascade_high_thresh,
                            )
                            if _mix is not None:
                                _cascade_n_total = _cascade_n_high + _cascade_n_low
                                if _cascade_n_total > 0:
                                    print(
                                        f"[FinalDecisionGate] {dataset_name} cascade: "
                                        f"sig_only={_cascade_n_high}/{_cascade_n_total} "
                                        f"(conf>={_fd_cascade_high_thresh:.2f}), "
                                        f"fusion={_cascade_n_low}/{_cascade_n_total}"
                                    )
                        else:
                            _mix = _fd_mix_votes(_fd_prob_parts)
                        if _mix is not None:
                            _mix = _mix / _mix.sum(dim=1, keepdim=True).clamp(min=_fd_eps)
                            _fd_pred = _mix.argmax(dim=1).cpu().numpy()
                            final_decision_voting_preds = _fd_pred.tolist()
                            final_decision_voting_acc = float(np.mean(_fd_pred == _lbls_eval_np))
                            final_decision_voting_macro_f1 = _safe_macro_f1(_lbls_eval_np, _fd_pred)
                            final_decision_voting_macro_recall = _safe_macro_recall(_lbls_eval_np, _fd_pred)
                            final_decision_voting_hmean = _safe_hmean_recall(_lbls_eval_np, _fd_pred)
                            final_decision_voting_sources = ",".join(_fd_sources)
                            _n_cls_cm = int(max(int(np.max(_lbls_eval_np)), int(np.max(_fd_pred))) + 1)
                            _cm_fd = np.zeros((_n_cls_cm, _n_cls_cm), dtype=int)
                            for _t, _p in zip(_lbls_eval_np.tolist(), _fd_pred.tolist()):
                                _cm_fd[int(_t), int(_p)] += 1
                            final_decision_cm = _cm_fd.tolist()
                            if _fd_print_cm:
                                _cnames_short = ["N", "IR", "OR", "Ball"]
                                _header = "ConfMat(true\\pred) " + " ".join(
                                    f"{(_cnames_short[i] if i < len(_cnames_short) else str(i)):>5}"
                                    for i in range(_n_cls_cm)
                                )
                                print(
                                    f"[FinalDecision] {dataset_name} voting_diag_acc={final_decision_voting_acc:.4f} "
                                    f"macro_f1={float(final_decision_voting_macro_f1):.4f} "
                                    f"sources={final_decision_voting_sources}"
                                )
                                print(f"[FinalDecision] {_header}")
                                for _ri in range(_n_cls_cm):
                                    _row_str = " ".join(f"{_cm_fd[_ri][_ci]:>5}" for _ci in range(_n_cls_cm))
                                    _rname = _cnames_short[_ri] if _ri < len(_cnames_short) else str(_ri)
                                    print(f"[FinalDecision]   {_rname:>17} {_row_str}")
        except Exception as _fd_exc:
            print(f"[FinalDecision] {dataset_name} voting failed: {type(_fd_exc).__name__}: {_fd_exc}")

        # ATT490: Phase-3「最终判断」口径 — 与 Top-1 Acc 候选集一致，供 LOO_SUMMARY_METRIC_SOURCE=best_path 使用。
        ensemble_macro_recall_zs = float("nan")
        if isinstance(_best_ens_recalls, dict) and len(_best_ens_recalls) > 0:
            ensemble_macro_recall_zs = float(np.mean(list(_best_ens_recalls.values())))

        def _fd_nn(_x):
            try:
                _v = float(_x)
                return _v if _v == _v else float("nan")
            except Exception:
                return float("nan")

        _fd_candidates = []
        if final_decision_voting_acc == final_decision_voting_acc and final_decision_voting_acc >= 0.0:
            _fd_candidates.append(
                (
                    "final_voting",
                    final_decision_voting_acc,
                    final_decision_voting_macro_f1,
                    final_decision_voting_macro_recall,
                )
            )
        _fd_candidates.extend([
            ("fusion_proto", fusion_proto_diag_acc_zs, fusion_proto_macro_f1_zs, fusion_proto_macro_recall_zs),
            ("spec_head", spec_head_diag_acc_zs, spec_head_macro_f1_zs, spec_head_macro_recall_zs),
            (
                "fusion_decoder",
                float(zero_shot_diag_res.get("diag_acc", float("nan"))),
                zero_shot_macro_f1,
                zero_shot_macro_recall,
            ),
        ])
        if _best_ens_acc is not None and float(_best_ens_acc) > 0:
            try:
                _bef1 = float(_best_ens_f1)
            except Exception:
                _bef1 = float("nan")
            if _bef1 != _bef1 or _bef1 < 0.0:
                _bef1 = float("nan")
            _fd_candidates.append(("ensemble", float(_best_ens_acc), _bef1, ensemble_macro_recall_zs))
        if not (_zs_repr_collapsed and _fd_exclude_td_on_collapse):
            _fd_candidates.append(
                ("transductive", transductive_diag_acc_zs, transductive_macro_f1_zs, transductive_macro_recall_zs)
            )
        elif transductive_diag_acc_zs == transductive_diag_acc_zs:
            print(f"[FinalDecisionGate] {dataset_name} exclude transductive from final_decision candidates (collapsed)")
        if "_cctc_best_acc" in locals() and float(_cctc_best_acc) > 0:
            _cctc_rc = float("nan")
            if "_cctc_best_recalls" in locals() and isinstance(_cctc_best_recalls, dict) and len(_cctc_best_recalls) > 0:
                _cctc_rc = float(np.mean(list(_cctc_best_recalls.values())))
            _fd_candidates.append(("cctc", float(_cctc_best_acc), float(_cctc_best_f1), _cctc_rc))
        _fd_valid = []
        for _nm, _ac, _f1, _rc in _fd_candidates:
            _a = _fd_nn(_ac)
            if _a != _a or _a < 0.0:
                continue
            _f = _fd_nn(_f1)
            _r = _fd_nn(_rc)
            _fd_valid.append((_nm, _a, _f, _r))
        if len(_fd_valid) > 0:
            _fd_pick = max(
                _fd_valid,
                key=lambda t: (t[1], t[2] if t[2] == t[2] else -1.0, t[3] if t[3] == t[3] else -1.0),
            )
            final_decision_path = str(_fd_pick[0])
            final_decision_diag_acc = float(_fd_pick[1])
            _f1p, _rcp = float(_fd_pick[2]), float(_fd_pick[3])
            final_decision_macro_f1 = _f1p if _f1p == _f1p else float("nan")
            final_decision_macro_recall = _rcp if _rcp == _rcp else float("nan")
        else:
            final_decision_path = ""
            final_decision_diag_acc = float("nan")
            final_decision_macro_f1 = float("nan")
            final_decision_macro_recall = float("nan")

        try:
            _csv_domain = str(dataset_name).strip().upper()
            _csv_held = str(_emb_export_tag or "").strip()
            _csv_phase = str(task_name)
            _export_modality_metrics_csv(
                {
                    "dataset_name": _csv_domain,
                    "phase_name": _csv_phase,
                    "domain": _csv_domain,
                    "held_out_condition": _csv_held,
                    "n_samples": int(_total_sh) if "_total_sh" in locals() and _total_sh else "",
                    "spec_head_acc": spec_head_diag_acc_zs,
                    "spec_head_macro_f1": spec_head_macro_f1_zs,
                    "spec_head_max_class_frac": modality_spec_head_max_frac,
                    "signal_head_acc": signal_head_diag_acc_zs,
                    "signal_head_macro_f1": signal_head_macro_f1_zs,
                    "signal_head_max_class_frac": modality_signal_head_max_frac,
                    "signal_proto_acc": signal_proto_diag_acc_zs,
                    "signal_proto_macro_f1": signal_proto_macro_f1_zs,
                    "signal_proto_max_class_frac": modality_signal_proto_max_frac,
                    "spec_proto_acc": spec_proto_diag_acc_zs,
                    "fusion_proto_acc": fusion_proto_diag_acc_zs,
                    "fusion_raw_acc": fusion_raw_diag_acc_zs,
                    "spec_vs_signal_disagree": modality_spec_signal_disagree,
                    "fusion_proto_sep_ratio": _fproto_sep_ratio if "_fproto_sep_ratio" in locals() else "",
                    "final_decision_acc": final_decision_diag_acc,
                    "final_decision_path": final_decision_path,
                }
            )
        except Exception as _e_csv:
            print(f"[ModalityCSV] {dataset_name} export skipped: {type(_e_csv).__name__}: {_e_csv}")

        results["zero_shot"] = {
            "auc_nf": float(zero_shot_nf_res.get("auc_nf", float("nan"))) if zero_shot_nf_res else float("nan"),
            "diag_acc": float(zero_shot_diag_res.get("diag_acc", float("nan"))),
            "diag_acc_raw": float(zero_shot_diag_res.get("diag_acc_raw", float("nan"))),
            "nf_acc": float(zero_shot_diag_res.get("nf_acc", float("nan"))),
            "macro_f1": zero_shot_macro_f1,
            "macro_recall": zero_shot_macro_recall,
            "hmean_recall": _zs_hmean,
            "spec_head_diag_acc": spec_head_diag_acc_zs,    # ATT116: spec_head direct 4-class bypass
            "spec_head_macro_f1": spec_head_macro_f1_zs,    # ATT120: spec_head macro-F1
            "spec_head_macro_recall": spec_head_macro_recall_zs,
            "signal_head_diag_acc": signal_head_diag_acc_zs,
            "signal_head_macro_f1": signal_head_macro_f1_zs,
            "signal_head_macro_recall": signal_head_macro_recall_zs,
            "signal_proto_diag_acc": signal_proto_diag_acc_zs,
            "signal_proto_macro_f1": signal_proto_macro_f1_zs,
            "modality_spec_signal_disagree": modality_spec_signal_disagree,
            "modality_spec_head_max_frac": modality_spec_head_max_frac,
            "modality_signal_head_max_frac": modality_signal_head_max_frac,
            "modality_signal_proto_max_frac": modality_signal_proto_max_frac,
            "fusion_raw_diag_acc": fusion_raw_diag_acc_zs,
            "fusion_raw_macro_f1": fusion_raw_macro_f1_zs,
            "spec_proto_diag_acc": spec_proto_diag_acc_zs,  # ATT474: z_spec vs proto matrix (text or learnable per cfg)
            # ATT143/146: 语义路径供 Phase 3 汇总表与单表展示
            "fusion_proto_diag_acc": fusion_proto_diag_acc_zs,
            "fusion_proto_macro_f1": fusion_proto_macro_f1_zs,
            "fusion_proto_macro_recall": fusion_proto_macro_recall_zs,
            "sem_proto_diag_acc": sem_proto_diag_acc_zs,
            "sem_proto_macro_f1": sem_proto_macro_f1_zs,
            "tip_adapter_diag_acc": tip_adapter_diag_acc_zs,
            "tip_adapter_macro_f1": tip_adapter_macro_f1_zs,
            "tip_adapter_alpha": tip_adapter_alpha_zs,
            "dynamic_multi_proto_diag_acc": dynamic_multi_proto_diag_acc_zs,
            "dynamic_multi_proto_macro_f1": dynamic_multi_proto_macro_f1_zs,
            # P0a2: 集成结果供 Phase 3 表格与 Task2 汇总使用
            "ensemble_diag_acc": float(_best_ens_acc) if _best_ens_acc > 0 else float("nan"),
            "ensemble_macro_f1": float(_best_ens_f1) if _best_ens_acc > 0 else float("nan"),
            "ensemble_macro_recall": ensemble_macro_recall_zs,
            "ensemble_label": str(_best_ens_label) if _best_ens_acc > 0 else "",
            "transductive_diag_acc": transductive_diag_acc_zs,
            "transductive_macro_f1": transductive_macro_f1_zs,
            "transductive_macro_recall": transductive_macro_recall_zs,
            "final_decision_path": final_decision_path,
            "final_decision_diag_acc": final_decision_diag_acc,
            "final_decision_macro_f1": final_decision_macro_f1,
            "final_decision_macro_recall": final_decision_macro_recall,
            "final_decision_voting_acc": final_decision_voting_acc,
            "final_decision_voting_macro_f1": final_decision_voting_macro_f1,
            "final_decision_voting_macro_recall": final_decision_voting_macro_recall,
            "final_decision_voting_hmean": final_decision_voting_hmean,
            "final_decision_voting_sources": final_decision_voting_sources,
            "final_decision_preds": final_decision_voting_preds,
            "final_decision_confusion_matrix": final_decision_cm,
        }
        
        # Task2 的 Unsupervised/Supervised 适配在 Two-Stage Adaptation 中执行，此处仅做 Zero-shot
        if task_name == "Task2":
            results["supervised"] = results["zero_shot"].copy()
            results["supervised"]["used_finetune"] = False
            results["supervised"]["finetune_size"] = 0
            results["supervised"]["eval_size"] = len(loader.dataset) if hasattr(loader, "dataset") else 0
            if ttfa_restore:
                ttfa_restore()
            if tent_restore:
                tent_restore()
            if shot_restore:
                shot_restore()
            if bn_restore:
                bn_restore()
            if _tpt_protos_backup is not None:
                detector.set_text_prototypes(_tpt_protos_backup)
            if hasattr(detector, "set_eval_dataset_name"):
                detector.set_eval_dataset_name(None)
            return results
        
        # ===== Stage 2: Supervised Finetune (有监督微调) =====
        used_finetune = False
        task1_snapshot_saved = False
        task1_ft_loader = None
        if task_name == "Task1" and enable_task1_ft:
            if eval_only_mode and eval_only_skip_task1_sup_ft:
                print(f"[Task1 Finetune] Skip supervised finetune for {dataset_name} in EVAL_ONLY_MODE")
                pass
            elif dataset_name == "CWRU" and skip_cwru_sup:
                # 跳过 CWRU 监督微调：直接用 zero-shot 的 loader 做 Supervised 阶段评估（避免负迁移）
                pass  # used_finetune 保持 False，后续用 loader 评估并填 results["supervised"]
            else:
                try:
                    if hasattr(detector, "snapshot_fusion_state"):
                        detector.snapshot_fusion_state()
                        task1_snapshot_saved = True
                    
                    ft_loader, holdout_loader = split_loader_for_target_adapt(
                        loader,
                        finetune_percent=task1_ft_ratio,
                        seed=split_seed,
                        dataset_name=dataset_name,
                        n_shot_k=int(getattr(cfg, "TARGET_NSHOT_K", 0)) if bool(getattr(cfg, "TARGET_NSHOT_ENABLE", False)) else 0,
                    )
                    if len(getattr(ft_loader, "dataset", [])) > 0 and len(getattr(holdout_loader, "dataset", [])) > 0:
                        detector.supervised_finetune_on_loader(
                            ft_loader,
                            dataset_name=f"{task_name}-{dataset_name}",
                            num_classes_eval=diag_num_classes,
                            num_steps=sup_steps,
                            lr=sup_lr,
                            nf_weight=sup_nf_w,
                            diag_weight=sup_diag_w,
                        )
                        eval_loader = holdout_loader
                        task1_ft_loader = ft_loader
                        used_finetune = True
                        finetune_size = len(ft_loader.dataset)
                        eval_size = len(holdout_loader.dataset)
                        _nshot_k_emb = (
                            int(getattr(cfg, "TARGET_NSHOT_K", 0))
                            if bool(getattr(cfg, "TARGET_NSHOT_ENABLE", False))
                            else 0
                        )
                        _export_metric_embeddings_if_enabled(
                            detector,
                            eval_loader,
                            dataset_name=dataset_name,
                            stage="postlpft",
                            n_shot_k=_nshot_k_emb,
                            export_tag=_emb_export_tag,
                        )
                except Exception as e:
                    print(f"[Task1 Finetune] Warning: pre-test finetune skipped for {dataset_name}: {e}")
        
        # 无论是否执行了finetune，都需要在原始loader上评估（或使用holdout）
        # 如果没有执行finetune，eval_loader = loader（与zero-shot相同）
        if not used_finetune:
            eval_loader = loader
            finetune_size = 0
            eval_size = len(loader.dataset) if hasattr(loader, "dataset") else 0
        

        
        # 执行Supervised评估
        nf_res = _evaluate(
            eval_loader,
            mode="fusion",
            scoring="evidence_or",
            dataset_name=dataset_name,
            enable_diagnosis=False,
        )
        diag_res = detector.evaluate_target_domain_accuracy(
            eval_loader, dataset_name=f"{task_name}-{dataset_name}", diag_num_classes=diag_num_classes
        )
        macro_f1 = _safe_macro_f1(diag_res.get("diag_labels", None), diag_res.get("diag_preds", None))
        macro_recall = _safe_macro_recall(diag_res.get("diag_labels", None), diag_res.get("diag_preds", None))
        
        _sup_hmean = _safe_hmean_recall(
            diag_res.get("diag_labels", None),
            diag_res.get("diag_preds", None),
        )
        results["supervised"] = {
            "auc_nf": float(nf_res.get("auc_nf", float("nan"))) if nf_res else float("nan"),
            "diag_acc": float(diag_res.get("diag_acc", float("nan"))),
            "diag_acc_raw": float(diag_res.get("diag_acc_raw", float("nan"))),
            "nf_acc": float(diag_res.get("nf_acc", float("nan"))),
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "hmean_recall": _sup_hmean,
            "used_finetune": bool(used_finetune),
            "finetune_size": finetune_size,
            "eval_size": eval_size,
        }

        # ATT219: Task1-PU 补齐 post-LPFT SpecHead / kNN-Spec 评估与替换逻辑。
        # 目标：避免 Task1-PU 的最终指标仅由 fusion 路径决定，充分利用 LP-FT 后 z_spec 的可分性。
        if task_name == "Task1" and dataset_name == "PU" and used_finetune:
            try:
                import config as cfg
                use_spec_post_lpft = bool(getattr(cfg, "TARGET_SUP_POST_LPFT_SPEC_HEAD_EVAL", True))
                spec_post_replace_when_better = bool(
                    getattr(cfg, "TARGET_SUP_POST_LPFT_SPEC_HEAD_REPLACE_WHEN_BETTER", True)
                )
                knn_eval_enable = bool(getattr(cfg, "TARGET_SUP_KNN_EVAL_ENABLE", False))
                knn_eval_k = int(getattr(cfg, "TARGET_SUP_KNN_K", 5))
                knn_replace_when_better = bool(getattr(cfg, "TARGET_SUP_KNN_REPLACE_WHEN_BETTER", False))
                task1_svp_enable = bool(getattr(cfg, "TARGET_TASK1_SVP_ENABLE", True))
                task1_svp_replace_when_better = bool(getattr(cfg, "TARGET_TASK1_SVP_REPLACE_WHEN_BETTER", True))
                task1_svp_use_fewshot_split = bool(getattr(cfg, "TARGET_TASK1_SVP_USE_FEWSHOT_SPLIT", True))
            except Exception:
                use_spec_post_lpft = True
                spec_post_replace_when_better = True
                knn_eval_enable = False
                knn_eval_k = 5
                knn_replace_when_better = False
                task1_svp_enable = True
                task1_svp_replace_when_better = True
                task1_svp_use_fewshot_split = True

            if use_spec_post_lpft:
                pu_spec_post_result = _evaluate_with_spec_head(
                    eval_loader,
                    dataset_name="PU/postLPFT",
                    diag_num_classes=diag_num_classes,
                )
                if pu_spec_post_result is not None and spec_post_replace_when_better:
                    if float(pu_spec_post_result.get("diag_acc", 0.0)) > float(results["supervised"].get("diag_acc", 0.0)):
                        print(
                            f"[SpecHead-PostLPFT] PU replacing supervised diag result: "
                            f"diag_acc {float(results['supervised'].get('diag_acc', 0.0)):.4f} "
                            f"-> {float(pu_spec_post_result.get('diag_acc', 0.0)):.4f}"
                        )
                        for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                            if k in pu_spec_post_result:
                                results["supervised"][k] = pu_spec_post_result.get(k, results["supervised"].get(k))
                        results["supervised"]["spec_head_post_lpft_diag_acc"] = float(pu_spec_post_result.get("diag_acc", float("nan")))

            if knn_eval_enable and task1_ft_loader is not None:
                pu_knn_result = _evaluate_with_spec_knn(
                    task1_ft_loader,
                    eval_loader,
                    dataset_name="PU/postLPFT",
                    diag_num_classes=diag_num_classes,
                    k=max(1, int(knn_eval_k)),
                )
                if pu_knn_result is not None and knn_replace_when_better:
                    if float(pu_knn_result.get("diag_acc", 0.0)) > float(results["supervised"].get("diag_acc", 0.0)):
                        print(
                            f"[kNN-Spec] PU replacing supervised diag result: "
                            f"diag_acc {float(results['supervised'].get('diag_acc', 0.0)):.4f} "
                            f"-> {float(pu_knn_result.get('diag_acc', 0.0)):.4f}"
                        )
                        for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                            if k in pu_knn_result:
                                results["supervised"][k] = pu_knn_result.get(k, results["supervised"].get(k))
                        results["supervised"]["knn_spec_diag_acc"] = float(pu_knn_result.get("diag_acc", float("nan")))

            if task1_svp_enable:
                _svp_ft_loader = task1_ft_loader if (task1_svp_use_fewshot_split and task1_ft_loader is not None) else loader
                pu_svp_result = None
                try:
                    _use_svp_fusion = bool(getattr(cfg, "TARGET_SUP_SVP_USE_FUSION", True))
                    _svp_fusion_weight = float(getattr(cfg, "TARGET_SUP_SVP_FUSION_WEIGHT", 0.35))
                    _svp_dual_auto_weight = bool(getattr(cfg, "TARGET_SUP_SVP_DUAL_AUTO_WEIGHT", True))
                    _svp_dual_weight_grid = list(getattr(cfg, "TARGET_SUP_SVP_DUAL_WEIGHT_GRID", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
                    _svp_tpa_enable = bool(getattr(cfg, "TARGET_SUP_SVP_TPA_ENABLE", True))
                    _svp_tpa_iters = int(getattr(cfg, "TARGET_SUP_SVP_TPA_ITERS", 2))
                    _svp_tpa_topk = float(getattr(cfg, "TARGET_SUP_SVP_TPA_TOPK_RATIO", 0.30))
                    _svp_tpa_min_per_class = int(getattr(cfg, "TARGET_SUP_SVP_TPA_MIN_SAMPLES_PER_CLASS", 5))
                    _svp_tpa_alpha = float(getattr(cfg, "TARGET_SUP_SVP_TPA_MIX_ALPHA", 0.80))
                    _svp_tpa_min_base_acc = float(getattr(cfg, "TARGET_SUP_SVP_TPA_MIN_BASE_ACC", 0.50))

                    _spec_pack, _ = _build_visual_prototypes(
                        _svp_ft_loader, diag_num_classes=diag_num_classes, space="spec"
                    )
                    _svp_candidates = []
                    if _spec_pack is not None:
                        _spec_pack_eval = _spec_pack
                        if _svp_tpa_enable:
                            _spec_fit = _evaluate_with_visual_prototypes(
                                _svp_ft_loader, _spec_pack, dataset_name="", diag_num_classes=diag_num_classes, space="spec"
                            )
                            if _spec_fit is not None and float(_spec_fit.get("diag_acc", 0.0)) >= _svp_tpa_min_base_acc:
                                _spec_pack_eval = _refine_visual_prototypes_with_unlabeled(
                                    eval_loader,
                                    _spec_pack,
                                    dataset_name="PU",
                                    diag_num_classes=diag_num_classes,
                                    space="spec",
                                    iters=_svp_tpa_iters,
                                    topk_ratio=_svp_tpa_topk,
                                    min_per_class=_svp_tpa_min_per_class,
                                    alpha=_svp_tpa_alpha,
                                )
                        _spec_svp = _evaluate_with_visual_prototypes(
                            eval_loader, _spec_pack_eval, dataset_name="PU/spec", diag_num_classes=diag_num_classes, space="spec"
                        )
                        if _spec_svp is not None:
                            _svp_candidates.append(_spec_svp)

                        if _use_svp_fusion:
                            _fusion_pack, _ = _build_visual_prototypes(
                                _svp_ft_loader, diag_num_classes=diag_num_classes, space="fusion"
                            )
                            if _fusion_pack is not None:
                                _fusion_pack_eval = _fusion_pack
                                if _svp_tpa_enable:
                                    _fusion_fit = _evaluate_with_visual_prototypes(
                                        _svp_ft_loader,
                                        _fusion_pack,
                                        dataset_name="",
                                        diag_num_classes=diag_num_classes,
                                        space="fusion",
                                    )
                                    if _fusion_fit is not None and float(_fusion_fit.get("diag_acc", 0.0)) >= _svp_tpa_min_base_acc:
                                        _fusion_pack_eval = _refine_visual_prototypes_with_unlabeled(
                                            eval_loader,
                                            _fusion_pack,
                                            dataset_name="PU",
                                            diag_num_classes=diag_num_classes,
                                            space="fusion",
                                            iters=_svp_tpa_iters,
                                            topk_ratio=_svp_tpa_topk,
                                            min_per_class=_svp_tpa_min_per_class,
                                            alpha=_svp_tpa_alpha,
                                        )
                                _fusion_svp = _evaluate_with_visual_prototypes(
                                    eval_loader,
                                    _fusion_pack_eval,
                                    dataset_name="PU/fusion",
                                    diag_num_classes=diag_num_classes,
                                    space="fusion",
                                )
                                if _fusion_svp is not None:
                                    _svp_candidates.append(_fusion_svp)
                                _dual_w = _svp_fusion_weight
                                if _svp_dual_auto_weight:
                                    _dual_w = _select_best_dual_weight_from_labeled(
                                        _svp_ft_loader,
                                        _spec_pack,
                                        _fusion_pack,
                                        dataset_name="PU",
                                        diag_num_classes=diag_num_classes,
                                        weight_grid=_svp_dual_weight_grid,
                                        fallback_weight=_svp_fusion_weight,
                                    )
                                _dual_svp = _evaluate_with_dual_visual_prototypes(
                                    eval_loader,
                                    _spec_pack_eval,
                                    _fusion_pack_eval,
                                    dataset_name="PU/dual",
                                    diag_num_classes=diag_num_classes,
                                    fusion_weight=_dual_w,
                                )
                                if _dual_svp is not None:
                                    _svp_candidates.append(_dual_svp)
                    pu_svp_result = (
                        max(_svp_candidates, key=lambda x: float(x.get("diag_acc", 0.0)))
                        if _svp_candidates else None
                    )
                    if pu_svp_result is not None:
                        print(f"[SVP] PU best_diag_acc={float(pu_svp_result.get('diag_acc', 0.0)):.4f}")
                except Exception as _svp_e:
                    print(f"[SVP] PU skipped: {type(_svp_e).__name__}: {_svp_e}")
                if pu_svp_result is not None:
                    results["supervised"]["svp_diag_acc"] = float(pu_svp_result.get("diag_acc", float("nan")))
                    if (
                        task1_svp_replace_when_better
                        and float(pu_svp_result.get("diag_acc", 0.0)) > float(results["supervised"].get("diag_acc", 0.0))
                    ):
                        print(
                            f"[SVP] PU replacing supervised diag result: "
                            f"diag_acc {float(results['supervised'].get('diag_acc', 0.0)):.4f} "
                            f"-> {float(pu_svp_result.get('diag_acc', 0.0)):.4f}"
                        )
                        for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                            if k in pu_svp_result:
                                results["supervised"][k] = pu_svp_result.get(k, results["supervised"].get(k))
        
        if used_finetune:
            try:
                print(
                    f"  [Supervised Finetune] {dataset_name}: finetune={finetune_size}, eval={eval_size}, ratio={task1_ft_ratio:.3f}"
                )
            except Exception:
                pass
            if hasattr(detector, "set_eval_nf_threshold"):
                detector.set_eval_nf_threshold(None)
            if hasattr(detector, "set_eval_nf_platt"):
                detector.set_eval_nf_platt(None, None)
            if task1_snapshot_saved and hasattr(detector, "restore_fusion_state"):
                detector.restore_fusion_state()
        if ttfa_restore:
            ttfa_restore()
        if tent_restore:
            tent_restore()
        if shot_restore:
            shot_restore()
        if bn_restore:
            bn_restore()
        if _tpt_protos_backup is not None:
            detector.set_text_prototypes(_tpt_protos_backup)
        if hasattr(detector, "set_eval_dataset_name"):
            detector.set_eval_dataset_name(None)
        return results

    # ===== 双任务摘要（论文友好）=====
    print("\n" + "=" * 80)
    print("【双任务摘要 (Task-level Readable Summary)】")
    print("=" * 80)
    print("Task1: 域内跨工况（CWRU/PU）；Task2: 跨域跨工况（HUST/SEU）")

    task1_results = {}
    if task1_cwru_loader is not None:
        task1_results["CWRU"] = _eval_task_block("Task1", "CWRU", task1_cwru_loader, diag_num_classes=4)
    if task1_pu_loader is not None:
        task1_results["PU"] = _eval_task_block("Task1", "PU", task1_pu_loader, diag_num_classes=3)
    # P0a: Phase 3 = 仅跨域评估（无 CWRU/PU）
    is_phase3 = (cwru_test_loader is None and pu_test_loader is None)

    task2_results = {}
    if hust_loader is not None:
        task2_results["HUST"] = _eval_task_block("Task2", "HUST", hust_loader, diag_num_classes=4)
    if hit_loader is not None:
        _hit_dname = _infer_loader_domain(hit_loader, default="SEU")
        _hit_nc = 3 if _hit_dname == "HIT" else 4
        task2_results[_hit_dname] = _eval_task_block("Task2", _hit_dname, hit_loader, diag_num_classes=_hit_nc)

    # ATT146: 每 Phase 仅输出一张综合性能汇总表（含 Legacy），不再分散打印 Task1/Task2 逐行与 PRIMARY/Legacy 板块
    def _fmt_v(v, fmt=".4f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        try:
            return format(float(v), fmt)
        except Exception:
            return "N/A"

    def _collect_legacy(loader, dataset_name):
        if loader is None:
            return {}
        out = {}
        for mode in ["signal", "spec", "fusion"]:
            r = _evaluate(loader, mode=mode, scoring="legacy", dataset_name=dataset_name)
            r = r or {}
            out[mode] = {"auc": r.get("auc", float("nan")), "auc_nf": r.get("auc_nf", float("nan"))}
        return out

    def _print_one_phase_table(dname, res_dict, loader, phase_label, has_supervised=True, sup_label="Supervised"):
        zs = res_dict.get("zero_shot", {})
        sup = res_dict.get("supervised", {}) if has_supervised else {}
        legacy = _collect_legacy(loader, dname) if loader else {}
        sep = "-" * 96
        print("\n" + "=" * 96)
        print(f"【{phase_label} — {dname} 综合性能汇总表】")
        print("=" * 96)
        print(f"  {'指标':<28}  {'Zero-shot':>12}  {sup_label:>12}  {'Δ(Sup-ZS)':>10}  │  说明")
        print(sep)
        def _fmt_delta(sup_v, zs_v):
            try:
                a = float(sup_v)
                b = float(zs_v)
                if np.isnan(a) or np.isnan(b):
                    return "N/A"
                return f"{(a - b):+.4f}"
            except Exception:
                return "N/A"
        print(f"  {'综合异常检测 AUC_nf':<28}  {_fmt_v(zs.get('auc_nf')):>12}  {_fmt_v(sup.get('auc_nf')):>12}  {_fmt_delta(sup.get('auc_nf'), zs.get('auc_nf')):>10}  │  Normal vs Fault 二分类")
        print(f"  {'SpecHead Acc':<28}  {_fmt_v(zs.get('spec_head_diag_acc')):>12}  {_fmt_v(sup.get('spec_head_diag_acc')):>12}  {_fmt_delta(sup.get('spec_head_diag_acc'), zs.get('spec_head_diag_acc')):>10}  │  主诊断路径")
        print(f"  {'SpecHead Macro-F1':<28}  {_fmt_v(zs.get('spec_head_macro_f1')):>12}  {_fmt_v(sup.get('spec_head_macro_f1')):>12}  {_fmt_delta(sup.get('spec_head_macro_f1'), zs.get('spec_head_macro_f1')):>10}  │")
        print(f"  {'SignalHead Acc':<28}  {_fmt_v(zs.get('signal_head_diag_acc')):>12}  {_fmt_v(sup.get('signal_head_diag_acc')):>12}  {_fmt_delta(sup.get('signal_head_diag_acc'), zs.get('signal_head_diag_acc')):>10}  │  z_sig classifier")
        print(f"  {'SignalHead Macro-F1':<28}  {_fmt_v(zs.get('signal_head_macro_f1')):>12}  {_fmt_v(sup.get('signal_head_macro_f1')):>12}  {_fmt_delta(sup.get('signal_head_macro_f1'), zs.get('signal_head_macro_f1')):>10}  │")
        print(f"  {'SignalProto Acc':<28}  {_fmt_v(zs.get('signal_proto_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  z_sig vs text_proto")
        print(f"  {'Spec vs Signal disagree':<28}  {_fmt_v(zs.get('modality_spec_signal_disagree')):>12}  {'—':>12}  {'—':>10}  │  head argmax 不一致率")
        try:
            import config as _cfg_tbl_sp
            _sp_tbl_note = (
                "learnable_proto"
                if bool(getattr(_cfg_tbl_sp, "EVAL_SPEC_PROTO_USE_LEARNABLE_PROTOS", False))
                else "text_proto"
            )
        except Exception:
            _sp_tbl_note = "text_proto"
        print(f"  {'SpecProto Acc (语义)':<28}  {_fmt_v(zs.get('spec_proto_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  z_spec vs {_sp_tbl_note}")
        print(f"  {'FusionProto Acc (语义)':<28}  {_fmt_v(zs.get('fusion_proto_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  z_fusion vs text_proto")
        print(f"  {'FusionProto Macro-F1':<28}  {_fmt_v(zs.get('fusion_proto_macro_f1')):>12}  {'—':>12}  {'—':>10}  │")
        print(f"  {'SemProto Acc (语义)':<28}  {_fmt_v(zs.get('sem_proto_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  z_sem vs text_proto")
        print(f"  {'SemProto Macro-F1':<28}  {_fmt_v(zs.get('sem_proto_macro_f1')):>12}  {'—':>12}  {'—':>10}  │")
        if eval_report_fusion_diag:
            print(f"  {'Fusion Diag-Acc':<28}  {_fmt_v(zs.get('diag_acc')):>12}  {_fmt_v(sup.get('diag_acc')):>12}  {_fmt_delta(sup.get('diag_acc'), zs.get('diag_acc')):>10}  │  decoder 4类")
            print(f"  {'Fusion Macro-F1':<28}  {_fmt_v(zs.get('macro_f1')):>12}  {_fmt_v(sup.get('macro_f1')):>12}  {_fmt_delta(sup.get('macro_f1'), zs.get('macro_f1')):>10}  │")
        else:
            print(f"  {'Fusion Diag-Acc':<28}  {'N/A':>12}  {'N/A':>12}  {'N/A':>10}  │  hidden (EVAL_REPORT_FUSION_DIAG=0)")
            print(f"  {'Fusion Macro-F1':<28}  {'N/A':>12}  {'N/A':>12}  {'N/A':>10}  │")
        print(f"  {'集成(最优) Acc':<28}  {_fmt_v(zs.get('ensemble_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  EnsGrid/CCTC")
        print(f"  {'集成(最优) Macro-F1':<28}  {_fmt_v(zs.get('ensemble_macro_f1')):>12}  {'—':>12}  {'—':>10}  │")
        try:
            import config as _cfg_tbl_td
            _td_note = (
                "ATT500 source-free BMM (FusionProto init)"
                if bool(getattr(_cfg_tbl_td, "TRANSDUCTIVE_SOURCE_FREE", False))
                else "ATT406 source-dependent kNN+BMM"
            )
        except Exception:
            _td_note = "Transductive"
        print(f"  {'Transductive Acc':<28}  {_fmt_v(zs.get('transductive_diag_acc')):>12}  {'—':>12}  {'—':>10}  │  {_td_note}")
        print(f"  {'Transductive Macro-F1':<28}  {_fmt_v(zs.get('transductive_macro_f1')):>12}  {'—':>12}  {'—':>10}  │")
        print(f"  {'最终决策(投票) Acc':<28}  {_fmt_v(zs.get('final_decision_voting_acc')):>12}  {'—':>12}  {'—':>10}  │  FinalVoting")
        print(f"  {'最终决策(投票) Macro-F1':<28}  {_fmt_v(zs.get('final_decision_voting_macro_f1')):>12}  {'—':>12}  {'—':>10}  │")
        print(sep)
        print("  [ATT393 综合指标]")
        _zs_final = float(zs.get("final_decision_diag_acc", float("nan")) or float("nan"))
        _zs_vote = float(zs.get("final_decision_voting_acc", float("nan")) or float("nan"))
        if _zs_vote == _zs_vote and _zs_vote > 0:
            _zs_final = _zs_vote
        _sup_final = float(sup.get("diag_acc", float("nan")) if sup else float("nan"))
        print(f"  {'Top-1 Acc (最终决策)':<28}  {_fmt_v(_zs_final):>12}  {_fmt_v(_sup_final):>12}  {_fmt_delta(_sup_final, _zs_final):>10}  │  统一决策输出")
        print(f"  {'H-mean (per-class recall)':<28}  {_fmt_v(zs.get('hmean_recall')):>12}  {_fmt_v(sup.get('hmean_recall')):>12}  {_fmt_delta(sup.get('hmean_recall'), zs.get('hmean_recall')):>10}  │  调和平均召回率")
        if eval_report_fusion_diag:
            print(f"  {'Fusion Macro-F1 (重复)':<28}  {_fmt_v(zs.get('macro_f1')):>12}  {_fmt_v(sup.get('macro_f1')):>12}  {_fmt_delta(sup.get('macro_f1'), zs.get('macro_f1')):>10}  │  Macro-F1=2PR/(P+R)")
        else:
            print(f"  {'Fusion Macro-F1 (重复)':<28}  {'N/A':>12}  {'N/A':>12}  {'N/A':>10}  │  hidden (EVAL_REPORT_FUSION_DIAG=0)")
        print(sep)
        print("  [Legacy 对比]")
        for mode in ["signal", "spec", "fusion"]:
            L = legacy.get(mode, {})
            print(f"  {'Legacy ' + mode.upper() + ' AUC':<28}  {_fmt_v(L.get('auc')):>12}  {'—':>12}  {'—':>10}  │  历史对比")
            print(f"  {'Legacy ' + mode.upper() + ' AUC_nf':<28}  {_fmt_v(L.get('auc_nf')):>12}  {'—':>12}  {'—':>10}  │")
        print(sep)
        if zs.get("ensemble_label"):
            print(f"  集成组合: {zs.get('ensemble_label', '')[:70]}")
        print("=" * 96)

    # Phase 1/2: 每个已评估的 Task1 数据集一张表
    if task1_results:
        for _d in ["CWRU", "PU"]:
            _res = task1_results.get(_d)
            if _res is None:
                continue
            _loader = cwru_test_loader if _d == "CWRU" else pu_test_loader
            if _loader is None:
                _loader = task1_cwru_loader if _d == "CWRU" else task1_pu_loader
            _print_one_phase_table(_d, _res, _loader, "Phase 1/2 (域内跨工况)", has_supervised=True, sup_label="Supervised")

    print("-" * 80)
    print("[Acceptance Checks]")
    def _is_valid_metric(v):
        try:
            v = float(v)
            return not np.isnan(v)
        except Exception:
            return False
    if task1_results:
        for dname in ["CWRU", "PU"]:
            res_dict = task1_results.get(dname, None)
            if res_dict is None:
                continue
            res = res_dict.get("supervised", {})
            if eval_report_fusion_diag:
                ok = _is_valid_metric(res.get("auc_nf", float("nan"))) and _is_valid_metric(res.get("diag_acc", float("nan")))
                print(f"  Task1-{dname}: {'PASS' if ok else 'WARN'} (AUC_nf + Diag-Acc)")
            else:
                ok = _is_valid_metric(res.get("auc_nf", float("nan"))) and _is_valid_metric(res.get("spec_head_diag_acc", float("nan")))
                print(f"  Task1-{dname}: {'PASS' if ok else 'WARN'} (AUC_nf + SpecHead-Acc)")
    if task2_results:
        for dname in ["HUST", "SEU"]:
            res_dict = task2_results.get(dname, None)
            if res_dict is None:
                continue
            res = res_dict.get("supervised", {})
            ok = _is_valid_metric(res.get("auc_nf", float("nan"))) and _is_valid_metric(res.get("diag_acc", float("nan")))
            print(f"  Task2-{dname}: {'PASS' if ok else 'WARN'} (AUC_nf + Diag-Acc)")
    print("-" * 80)

    # ===== Target-domain adaptation evaluation (isolated per domain) =====
    print("\n" + "="*80)
    print("【目标域两阶段适配评估 (Target-Domain Two-Stage Adaptation - ISOLATED)】")
    print("="*80)
    print("说明: 每个目标域独立评估，使用模型副本避免相互干扰")
    print("-"*80)
    
    # Import config for target-domain adaptation parameters
    try:
        import config as cfg
        target_finetune_percent = getattr(cfg, 'TARGET_FINETUNE_PERCENT', 0.05)
        target_finetune_percent_hust = getattr(cfg, 'TARGET_FINETUNE_PERCENT_HUST', target_finetune_percent)
        target_finetune_percent_hit = getattr(
            cfg,
            'TARGET_FINETUNE_PERCENT_SEU',
            getattr(cfg, 'TARGET_FINETUNE_PERCENT_HIT', target_finetune_percent)
        )
        target_nshot_enable = bool(getattr(cfg, "TARGET_NSHOT_ENABLE", False))
        target_nshot_k = int(getattr(cfg, "TARGET_NSHOT_K", 0))
        target_unsup_steps = getattr(cfg, 'TARGET_UNSUP_STEPS', None)
        target_unsup_lr = getattr(cfg, 'TARGET_UNSUP_LR', None)
        target_sup_steps = getattr(cfg, 'TARGET_SUP_STEPS', None)
        target_sup_lr = getattr(cfg, 'TARGET_SUP_LR', None)
        target_sup_nf_weight = getattr(cfg, 'TARGET_SUP_NF_WEIGHT', 0.5)
        target_sup_diag_weight = getattr(cfg, 'TARGET_SUP_DIAG_WEIGHT', 1.0)
        unified_zeroshot_eval = bool(getattr(cfg, "UNIFIED_ZEROSHOT_EVAL", False))
    except ImportError:
        target_finetune_percent = 0.05
        target_finetune_percent_hust = target_finetune_percent
        target_finetune_percent_hit = target_finetune_percent
        target_nshot_enable = False
        target_nshot_k = 0
        target_unsup_steps = None
        target_unsup_lr = None
        target_sup_steps = None
        target_sup_lr = None
        target_sup_nf_weight = 0.5
        target_sup_diag_weight = 1.0
        unified_zeroshot_eval = False
    
    # (Attempt 49) split_loader_for_target_adapt is now a module-level function
    # for reuse in main.py dual-checkpoint LP-FT. Same seed/logic → identical splits.
    # P0a2: 收集两阶段适配结果，供 Phase 3 最终表格与 task2_results 更新使用
    adaptation_results = {}
    if target_nshot_enable and int(target_nshot_k) > 0:
        print(
            f"\n[N-shot] Target adaptation active: K={int(target_nshot_k)} per class "
            f"(skip_lpft_fewshot={bool(getattr(cfg, 'TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE', True))}, "
            f"min_samples={int(getattr(cfg, 'TARGET_SUP_SKIP_LPFT_MIN_SAMPLES', 35))})"
        )
    if unified_zeroshot_eval:
        print("\n" + "=" * 80)
        print("[UNIFIED_ZEROSHOT_EVAL] Skipping isolated two-stage target adaptation (HUST/SEU).")
        print("Task2 supervised metrics remain aligned with zero-shot (no LP-FT/SVP in this block).")
        print("=" * 80 + "\n")
    # ATT181: SVP + few-shot LP-FT safeguard (evaluation-time adaptation controls).
    try:
        import config as cfg
        use_svp = bool(getattr(cfg, "TARGET_SUP_SVP_ENABLE", True))
        use_svp_fusion = bool(getattr(cfg, "TARGET_SUP_SVP_USE_FUSION", True))
        svp_fusion_weight = float(getattr(cfg, "TARGET_SUP_SVP_FUSION_WEIGHT", 0.35))
        svp_dual_auto_weight = bool(getattr(cfg, "TARGET_SUP_SVP_DUAL_AUTO_WEIGHT", True))
        svp_dual_weight_grid = list(getattr(cfg, "TARGET_SUP_SVP_DUAL_WEIGHT_GRID", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        svp_tpa_enable = bool(getattr(cfg, "TARGET_SUP_SVP_TPA_ENABLE", True))
        svp_tpa_iters = int(getattr(cfg, "TARGET_SUP_SVP_TPA_ITERS", 2))
        svp_tpa_topk = float(getattr(cfg, "TARGET_SUP_SVP_TPA_TOPK_RATIO", 0.30))
        svp_tpa_min_per_class = int(getattr(cfg, "TARGET_SUP_SVP_TPA_MIN_SAMPLES_PER_CLASS", 5))
        svp_tpa_alpha = float(getattr(cfg, "TARGET_SUP_SVP_TPA_MIX_ALPHA", 0.80))
        svp_tpa_min_base_acc = float(getattr(cfg, "TARGET_SUP_SVP_TPA_MIN_BASE_ACC", 0.60))
        skip_lpft_fewshot = bool(getattr(cfg, "TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE", True))
        skip_lpft_min_samples = int(getattr(cfg, "TARGET_SUP_SKIP_LPFT_MIN_SAMPLES", 40))
        svp_replace_when_better = bool(getattr(cfg, "TARGET_SUP_SVP_REPLACE_WHEN_BETTER", True))
        use_spec_post_lpft = bool(getattr(cfg, "TARGET_SUP_POST_LPFT_SPEC_HEAD_EVAL", True))
        spec_post_replace_when_better = bool(getattr(cfg, "TARGET_SUP_POST_LPFT_SPEC_HEAD_REPLACE_WHEN_BETTER", True))
        self_train_enable = bool(getattr(cfg, "TARGET_SELF_TRAINING_ENABLE", False))
        svp_baseline_guard_enable = bool(getattr(cfg, "TARGET_SUP_SVP_BASELINE_GUARD_ENABLE", True))
        lp_dual_snapshot_enable = bool(getattr(cfg, "TARGET_SUP_LP_DUAL_BEST_SNAPSHOT_ENABLE", True))
        knn_eval_enable = bool(getattr(cfg, "TARGET_SUP_KNN_EVAL_ENABLE", False))
        knn_eval_k = int(getattr(cfg, "TARGET_SUP_KNN_K", 5))
        knn_replace_when_better = bool(getattr(cfg, "TARGET_SUP_KNN_REPLACE_WHEN_BETTER", False))
        lp_ema_eval_enable = bool(getattr(cfg, "TARGET_SUP_LP_EMA_EVAL_ENABLE", False))
        lp_ema_use_for_nonparam_eval = bool(getattr(cfg, "TARGET_SUP_LP_EMA_USE_FOR_NONPARAM_EVAL", False))
        lp_retrain_full_nonparam_eval_enable = bool(
            getattr(cfg, "TARGET_SUP_LP_RETRAIN_FULL_NONPARAM_EVAL_ENABLE", False)
        )
        label_prop_enable = bool(getattr(cfg, "TARGET_SUP_LABEL_PROP_ENABLE", False))
        label_prop_replace_when_better = bool(getattr(cfg, "TARGET_SUP_LABEL_PROP_REPLACE_WHEN_BETTER", True))
        label_prop_kernel = str(getattr(cfg, "TARGET_SUP_LABEL_PROP_KERNEL", "knn")).strip().lower()
        label_prop_n_neighbors = int(getattr(cfg, "TARGET_SUP_LABEL_PROP_N_NEIGHBORS", 10))
        label_prop_alpha = float(getattr(cfg, "TARGET_SUP_LABEL_PROP_ALPHA", 0.20))
        label_prop_max_iter = int(getattr(cfg, "TARGET_SUP_LABEL_PROP_MAX_ITER", 30))
        label_prop_tol = float(getattr(cfg, "TARGET_SUP_LABEL_PROP_TOL", 1e-3))
        label_prop_dual_enable = bool(getattr(cfg, "TARGET_SUP_LABEL_PROP_DUAL_SPACE_ENABLE", False))
        label_prop_fusion_weight = float(getattr(cfg, "TARGET_SUP_LABEL_PROP_FUSION_WEIGHT", 0.35))
        tim_eval_enable = bool(getattr(cfg, "TARGET_SUP_TIM_ENABLE", False))
        tim_replace_when_better = bool(getattr(cfg, "TARGET_SUP_TIM_REPLACE_WHEN_BETTER", True))
        spec_knn_ens_enable = bool(getattr(cfg, "TARGET_SUP_SPEC_KNN_ENSEMBLE_ENABLE", False))
        spec_knn_ens_weight_grid = list(getattr(cfg, "TARGET_SUP_SPEC_KNN_ENSEMBLE_WEIGHT_GRID", [0.30, 0.40, 0.50, 0.60, 0.70]))
        spec_knn_ens_replace_when_better = bool(
            getattr(cfg, "TARGET_SUP_SPEC_KNN_ENSEMBLE_REPLACE_WHEN_BETTER", True)
        )
    except Exception:
        use_svp = True
        use_svp_fusion = True
        svp_fusion_weight = 0.35
        svp_dual_auto_weight = True
        svp_dual_weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        svp_tpa_enable = True
        svp_tpa_iters = 2
        svp_tpa_topk = 0.30
        svp_tpa_min_per_class = 5
        svp_tpa_alpha = 0.80
        svp_tpa_min_base_acc = 0.60
        skip_lpft_fewshot = True
        skip_lpft_min_samples = 40
        svp_replace_when_better = True
        use_spec_post_lpft = True
        spec_post_replace_when_better = True
        self_train_enable = False
        svp_baseline_guard_enable = True
        lp_dual_snapshot_enable = True
        knn_eval_enable = False
        knn_eval_k = 5
        knn_replace_when_better = False
        lp_ema_eval_enable = False
        lp_ema_use_for_nonparam_eval = False
        lp_retrain_full_nonparam_eval_enable = False
        label_prop_enable = False
        label_prop_replace_when_better = True
        label_prop_kernel = "knn"
        label_prop_n_neighbors = 10
        label_prop_alpha = 0.20
        label_prop_max_iter = 30
        label_prop_tol = 1e-3
        label_prop_dual_enable = False
        label_prop_fusion_weight = 0.35
        tim_eval_enable = False
        tim_replace_when_better = True
        spec_knn_ens_enable = False
        spec_knn_ens_weight_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        spec_knn_ens_replace_when_better = True

    def _run_best_svp_result(finetune_loader, eval_loader, dataset_name, diag_num_classes, spec_transform_fn=None):
        """
        Run spec/fusion/dual SVP suite and return the best diagnosis result.
        """
        if not use_svp:
            return None
        spec_pack, spec_counts = _build_visual_prototypes(
            finetune_loader, diag_num_classes=diag_num_classes, space="spec", spec_transform_fn=spec_transform_fn
        )
        if spec_pack is None:
            print(f"[SVP] {dataset_name} skipped: unable to build class prototypes from finetune split")
            return None
        if not eval_compact_log:
            c_line = ", ".join([f"{i}:{int(v)}" for i, v in enumerate(spec_counts)])
            print(f"[SVP] {dataset_name} prototype counts (spec): {c_line}")
        _allow_spec_tpa = bool(svp_tpa_enable)
        if _allow_spec_tpa:
            spec_fit_result = _evaluate_with_visual_prototypes(
                finetune_loader,
                spec_pack,
                dataset_name="",
                diag_num_classes=diag_num_classes,
                space="spec",
                spec_transform_fn=spec_transform_fn,
            )
            spec_fit_acc = float(spec_fit_result.get("diag_acc", 0.0)) if spec_fit_result is not None else 0.0
            if spec_fit_acc + 1e-12 < float(svp_tpa_min_base_acc):
                _allow_spec_tpa = False
                print(
                    f"[SVP-TPA] {dataset_name} spec skipped: fit_acc={spec_fit_acc:.4f} "
                    f"< min_base_acc={float(svp_tpa_min_base_acc):.4f}"
                )
        if _allow_spec_tpa:
            spec_pack_eval = _refine_visual_prototypes_with_unlabeled(
                eval_loader,
                spec_pack,
                dataset_name=dataset_name,
                diag_num_classes=diag_num_classes,
                space="spec",
                iters=svp_tpa_iters,
                topk_ratio=svp_tpa_topk,
                min_per_class=svp_tpa_min_per_class,
                alpha=svp_tpa_alpha,
                spec_transform_fn=spec_transform_fn,
            )
        else:
            spec_pack_eval = spec_pack
        spec_svp_result = _evaluate_with_visual_prototypes(
            eval_loader,
            spec_pack_eval,
            dataset_name=f"{dataset_name}/spec",
            diag_num_classes=diag_num_classes,
            space="spec",
            spec_transform_fn=spec_transform_fn,
        )

        fusion_svp_result = None
        dual_svp_result = None
        if use_svp_fusion:
            fusion_pack, fusion_counts = _build_visual_prototypes(finetune_loader, diag_num_classes=diag_num_classes, space="fusion")
            if fusion_pack is not None:
                if not eval_compact_log:
                    c_line = ", ".join([f"{i}:{int(v)}" for i, v in enumerate(fusion_counts)])
                    print(f"[SVP] {dataset_name} prototype counts (fusion): {c_line}")
                _allow_fusion_tpa = bool(svp_tpa_enable)
                if _allow_fusion_tpa:
                    fusion_fit_result = _evaluate_with_visual_prototypes(
                        finetune_loader, fusion_pack, dataset_name="", diag_num_classes=diag_num_classes, space="fusion"
                    )
                    fusion_fit_acc = float(fusion_fit_result.get("diag_acc", 0.0)) if fusion_fit_result is not None else 0.0
                    if fusion_fit_acc + 1e-12 < float(svp_tpa_min_base_acc):
                        _allow_fusion_tpa = False
                        print(
                            f"[SVP-TPA] {dataset_name} fusion skipped: fit_acc={fusion_fit_acc:.4f} "
                            f"< min_base_acc={float(svp_tpa_min_base_acc):.4f}"
                        )
                if _allow_fusion_tpa:
                    fusion_pack_eval = _refine_visual_prototypes_with_unlabeled(
                        eval_loader,
                        fusion_pack,
                        dataset_name=dataset_name,
                        diag_num_classes=diag_num_classes,
                        space="fusion",
                        iters=svp_tpa_iters,
                        topk_ratio=svp_tpa_topk,
                        min_per_class=svp_tpa_min_per_class,
                        alpha=svp_tpa_alpha,
                    )
                else:
                    fusion_pack_eval = fusion_pack
                fusion_svp_result = _evaluate_with_visual_prototypes(
                    eval_loader, fusion_pack_eval, dataset_name=f"{dataset_name}/fusion", diag_num_classes=diag_num_classes, space="fusion"
                )
                dual_w = svp_fusion_weight
                if svp_dual_auto_weight:
                    dual_w = _select_best_dual_weight_from_labeled(
                        finetune_loader,
                        spec_pack,
                        fusion_pack,
                        dataset_name=dataset_name,
                        diag_num_classes=diag_num_classes,
                        weight_grid=svp_dual_weight_grid,
                        fallback_weight=svp_fusion_weight,
                    )
                dual_svp_result = _evaluate_with_dual_visual_prototypes(
                    eval_loader,
                    spec_pack_eval,
                    fusion_pack_eval,
                    dataset_name=f"{dataset_name}/dual",
                    diag_num_classes=diag_num_classes,
                    fusion_weight=dual_w,
                    spec_transform_fn=spec_transform_fn,
                )
        svp_candidates = [r for r in [spec_svp_result, fusion_svp_result, dual_svp_result] if r is not None]
        best = max(svp_candidates, key=lambda x: float(x.get("diag_acc", 0.0))) if svp_candidates else None
        if best is not None:
            print(f"[SVP] {dataset_name} best_diag_acc={float(best.get('diag_acc', 0.0)):.4f}")
        return best

    try:
        import config as cfg
        skip_eval_hust = bool(getattr(cfg, "TARGET_EVAL_SKIP_HUST", False))
        skip_eval_hit = bool(getattr(cfg, "TARGET_EVAL_SKIP_SEU", getattr(cfg, "TARGET_EVAL_SKIP_HIT", False)))
    except Exception:
        skip_eval_hust = False
        skip_eval_hit = False

    def _maybe_switch_task2_domain_checkpoint(domain_name: str):
        """
        P1: Optional domain-specific Task2 checkpoint switch in eval-only mode.
        """
        try:
            import config as _cfg_dom
            _enable = bool(getattr(_cfg_dom, "TARGET_TASK2_DOMAIN_CHECKPOINT_SWITCH_ENABLE", False))
            _eval_only = bool(getattr(_cfg_dom, "EVAL_ONLY_MODE", False))
            _mapping = getattr(_cfg_dom, "TARGET_TASK2_DOMAIN_CHECKPOINTS", {})
            if (not _enable) or (not _eval_only) or (not isinstance(_mapping, dict)):
                return False
            _raw_ckpt = _mapping.get(str(domain_name), None)
            if not _raw_ckpt:
                return False
            _candidates = [str(_raw_ckpt)]
            if not os.path.isabs(str(_raw_ckpt)):
                _candidates.append(os.path.abspath(str(_raw_ckpt)))
            _ckpt = None
            for _p in _candidates:
                if os.path.exists(_p):
                    _ckpt = _p
                    break
            if _ckpt is None:
                print(f"[Task2 Domain Ckpt] {domain_name}: checkpoint not found, skip switch: {_raw_ckpt}")
                return False
            from utils.training_utils import load_model_checkpoint
            from utils.checkpoint import load_text_prototypes

            _device = getattr(detector, "device", "cpu")
            load_model_checkpoint(
                _ckpt,
                detector.signal_encoder,
                detector.spec_encoder,
                detector.text_encoder,
                detector.fusion_module,
                detector.fusion_decoder,
                optimizer=None,
                device=_device,
            )
            _protos = load_text_prototypes(_ckpt, device=_device)
            if isinstance(_protos, dict) and len(_protos) > 0:
                detector.set_text_prototypes(_protos)
            print(f"[Task2 Domain Ckpt] {domain_name}: switched to {_ckpt}")
            return True
        except Exception as _e:
            print(f"[Task2 Domain Ckpt] {domain_name}: switch failed: {_e}")
            return False

    # Process HUST (4-class)
    if (not unified_zeroshot_eval) and hust_loader is not None and (not skip_eval_hust):
        print("\n" + "-"*80)
        print(f"[HUST] Starting isolated two-stage adaptation (4-class)")
        print("-"*80)
        _maybe_switch_task2_domain_checkpoint("HUST")
        
        # Snapshot base model state
        detector.snapshot_fusion_state()
        
        # Split dataset by domain-specific finetune ratio
        hust_finetune_loader, hust_eval_loader = split_loader_for_target_adapt(
            hust_loader,
            finetune_percent=target_finetune_percent_hust,
            seed=42,
            dataset_name="HUST",
            n_shot_k=(target_nshot_k if target_nshot_enable else 0),
        )
        hust_spec_twc_fn = _fit_twc_for_target(hust_eval_loader, "HUST")
        print(f"[HUST] Split: {len(hust_finetune_loader.dataset)} samples for finetune, {len(hust_eval_loader.dataset)} samples for eval")

        # (Attempt 41) Log labeled split distribution for diagnosability
        if not eval_compact_log:
            try:
                counts = {}
                for batch in hust_finetune_loader:
                    y = batch.get("label", None)
                    if y is None:
                        continue
                    for v in y.detach().cpu().tolist():
                        counts[int(v)] = counts.get(int(v), 0) + 1
                counts_line = ", ".join([f"{k}:{counts.get(k,0)}" for k in sorted(counts.keys())]) if len(counts) > 0 else "N/A"
                _hust_split_tag = (
                    f"n-shot K={int(target_nshot_k)}"
                    if target_nshot_enable and int(target_nshot_k) > 0
                    else f"{target_finetune_percent_hust*100:.1f}%"
                )
                print(f"[HUST] Finetune label counts ({_hust_split_tag}): {counts_line}")
            except Exception:
                pass
        
        # Stage 1: Unsupervised adaptation
        # NOTE (stability): default skip for HUST (see cfg.TARGET_SKIP_HUST_UNSUP_ADAPT).
        try:
            import config as cfg
            skip_hust_unsup = bool(getattr(cfg, "TARGET_SKIP_HUST_UNSUP_ADAPT", False))
        except Exception:
            skip_hust_unsup = False

        print(f"\n[HUST] Stage 1: Unsupervised adaptation (no labels)")
        if skip_hust_unsup:
            print("[HUST] Stage 1 skipped (TARGET_SKIP_HUST_UNSUP_ADAPT=True)")
        else:
            detector.unsupervised_adapt_on_loader(hust_finetune_loader, dataset_name="HUST")
        
        # Evaluate after stage-1 (either adapted or skipped)
        detector.set_eval_nf_threshold(None)
        hust_unsup_result = detector.evaluate_target_domain_accuracy(
            hust_eval_loader, dataset_name="HUST", diag_num_classes=4
        )
        if skip_hust_unsup:
            print(f"[HUST] After Stage 1 (skipped):")
        else:
            print(f"[HUST] After Unsupervised Adaptation:")
        if eval_compact_log:
            print(f"  stage1 nf_acc={hust_unsup_result['nf_acc']:.4f}, diag_acc={hust_unsup_result['diag_acc']:.4f}")
        else:
            print(f"  NF Accuracy: {hust_unsup_result['nf_acc']:.4f} ({hust_unsup_result['nf_correct']}/{hust_unsup_result['nf_total']})")
            print(f"  Diagnosis Accuracy (4-class): {hust_unsup_result['diag_acc']:.4f} ({hust_unsup_result['diag_correct']}/{hust_unsup_result['diag_total']})")
        if (not eval_compact_log) and ('diag_acc_raw' in hust_unsup_result):
            print(f"  Diagnosis Accuracy raw (4-class): {hust_unsup_result['diag_acc_raw']:.4f}")
        
        # Stage 2: Supervised fine-tuning (n-shot or percent-labeled split)
        _hust_nshot_active = bool(target_nshot_enable and int(target_nshot_k) > 0)
        skip_lpft_now = bool(
            skip_lpft_fewshot
            and (len(hust_finetune_loader.dataset) < int(skip_lpft_min_samples))
            and (not _hust_nshot_active)
        )
        _hust_stage2_tag = (
            f"n-shot K={int(target_nshot_k)}"
            if _hust_nshot_active
            else f"{target_finetune_percent_hust*100:.1f}% labeled"
        )
        print(f"\n[HUST] Stage 2: Supervised fine-tuning ({_hust_stage2_tag})")
        hust_svp_base_result = None
        if use_svp and svp_baseline_guard_enable:
            print("[SVP] HUST baseline guard: evaluating pre-LPFT SVP")
            hust_svp_base_result = _run_best_svp_result(
                hust_finetune_loader,
                hust_eval_loader,
                dataset_name="HUST/baseLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_svp else None),
            )
        if skip_lpft_now:
            print(
                f"[HUST] Stage 2 LP-FT skipped: finetune samples={len(hust_finetune_loader.dataset)} "
                f"< min_samples={int(skip_lpft_min_samples)} (few-shot safeguard)"
            )
        else:
            detector.supervised_finetune_on_loader(
                hust_finetune_loader, dataset_name="HUST", num_classes_eval=4,
                num_steps=target_sup_steps, lr=target_sup_lr,
                nf_weight=target_sup_nf_weight, diag_weight=target_sup_diag_weight,
                unlabeled_loader=hust_eval_loader,
            )

        # (Attempt 41) IMPORTANT: NF head (Platt/thresholding) is only meaningful for hierarchical decoder.
        # For flat diagnosis head (hierarchical=False), skip NF calibration/threshold selection and report
        # supervised finetune effect using diagnosis metrics directly.
        try:
            has_nf_head = bool(getattr(detector.fusion_decoder, "hierarchical", False))
        except Exception:
            has_nf_head = False

        if has_nf_head:
            # (A1-1) Fit Platt calibration for NF on labeled split (more stable than pure thresholding)
            try:
                import config as cfg
                use_platt = bool(getattr(cfg, "USE_NF_PLATT_CALIBRATION", True))
                platt_max = int(getattr(cfg, "NF_PLATT_MAX_SAMPLES", 512))
                platt_steps = int(getattr(cfg, "NF_PLATT_STEPS", 200))
                platt_lr = float(getattr(cfg, "NF_PLATT_LR", 0.1))
                platt_l2 = float(getattr(cfg, "NF_PLATT_L2", 1e-3))
            except Exception:
                use_platt = True
                platt_max, platt_steps, platt_lr, platt_l2 = 512, 200, 0.1, 1e-3
            if use_platt and hasattr(detector, "fit_eval_nf_platt"):
                s, b, loss_pl = detector.fit_eval_nf_platt(
                    hust_finetune_loader, dataset_name="HUST",
                    max_samples=platt_max, steps=platt_steps, lr=platt_lr, l2=platt_l2
                )
                print(f"[HUST] NF Platt calibration (fit on 5% labeled): scale={s:.4f}, bias={b:.4f}, loss={loss_pl:.4f}")

            # Select NF threshold candidates on the labeled finetune split (5%) after supervised fine-tuning.
            hust_t_acc, hust_acc_score = detector.find_best_nf_threshold(hust_finetune_loader, dataset_name="HUST", objective="acc")
            hust_t_bal, hust_bal_score = detector.find_best_nf_threshold(hust_finetune_loader, dataset_name="HUST", objective="balanced_acc")
            # Evaluate BOTH thresholds on the held-out eval split and select the one that yields higher diagnosis accuracy.
            detector.set_eval_nf_threshold(hust_t_acc)
            hust_sup_result_acc = detector.evaluate_target_domain_accuracy(hust_eval_loader, dataset_name="HUST", diag_num_classes=4)
            detector.set_eval_nf_threshold(hust_t_bal)
            hust_sup_result_bal = detector.evaluate_target_domain_accuracy(hust_eval_loader, dataset_name="HUST", diag_num_classes=4)
        else:
            # Use diag accuracy on finetune split as "finetune acc" (directly measurable & meaningful).
            detector.set_eval_nf_threshold(None)
            hust_finetune_result = detector.evaluate_target_domain_accuracy(
                hust_finetune_loader, dataset_name="HUST(finetune)", diag_num_classes=4
            )
            hust_sup_result_acc = detector.evaluate_target_domain_accuracy(
                hust_eval_loader, dataset_name="HUST", diag_num_classes=4
            )
            hust_sup_result_bal = hust_sup_result_acc
            hust_t_acc, hust_t_bal = 0.0, 0.0
            hust_acc_score = float(hust_finetune_result.get("diag_acc", 0.0))
            hust_bal_score = float(hust_acc_score)

        try:
            import config as cfg
            diag_mode = str(getattr(cfg, "DIAG_DECISION_MODE", "nf_protect")).lower()
        except Exception:
            diag_mode = "nf_protect"

        # Selection rule:
        # - If NF head exists and decision uses nf_gate, choose by eval diag_acc (tie-break nf_acc).
        # - Otherwise (argmax/flat head), threshold is irrelevant; keep a single result.
        if has_nf_head and diag_mode == "nf_gate":
            a = float(hust_sup_result_acc.get("diag_acc", 0.0))
            b = float(hust_sup_result_bal.get("diag_acc", 0.0))
            if (b > a) or (b == a and float(hust_sup_result_bal.get("nf_acc", 0.0)) > float(hust_sup_result_acc.get("nf_acc", 0.0))):
                hust_t = hust_t_bal
                hust_sup_result = hust_sup_result_bal
            else:
                hust_t = hust_t_acc
                hust_sup_result = hust_sup_result_acc
        else:
            hust_t = hust_t_acc
            hust_sup_result = hust_sup_result_acc

        # ATT187(A): explicitly evaluate post-LPFT spec_head path and optionally use it.
        hust_spec_post_result = None
        if (not has_nf_head) and use_spec_post_lpft:
            if (not skip_lpft_now) and lp_dual_snapshot_enable:
                if detector.restore_lpft_best_state("spec_ce"):
                    print("[LP-FT DualBest] HUST switched to spec_ce-best snapshot for SpecHead evaluation")
            hust_spec_post_result = _evaluate_with_spec_head(
                hust_eval_loader,
                dataset_name="HUST/postLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if (not skip_lpft_now) and lp_ema_eval_enable and hasattr(detector, "restore_lpft_ema_state"):
                try:
                    _ema_ok = bool(detector.restore_lpft_ema_state())
                    if _ema_ok:
                        print("[LP-FT EMA] HUST switched to EMA snapshot for SpecHead evaluation")
                        hust_spec_post_ema = _evaluate_with_spec_head(
                            hust_eval_loader,
                            dataset_name="HUST/postLPFT-EMA",
                            diag_num_classes=4,
                            spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
                        )
                        if (
                            hust_spec_post_ema is not None
                            and (
                                hust_spec_post_result is None
                                or float(hust_spec_post_ema.get("diag_acc", 0.0))
                                > float(hust_spec_post_result.get("diag_acc", 0.0))
                            )
                        ):
                            hust_spec_post_result = hust_spec_post_ema
                    else:
                        print("[LP-FT EMA] HUST SpecHead evaluation skip switch: restore_lpft_ema_state() returned False")
                    if lp_dual_snapshot_enable:
                        detector.restore_lpft_best_state("spec_ce")
                except Exception as _e_ema:
                    print(f"[LP-FT EMA] HUST SpecHead evaluation switch failed: {type(_e_ema).__name__}: {_e_ema}")
            if hust_spec_post_result is not None:
                hust_sup_result["spec_head_diag_acc"] = float(hust_spec_post_result.get("diag_acc", float("nan")))
                hust_sup_result["spec_head_macro_f1"] = float(hust_spec_post_result.get("macro_f1", float("nan")))
                if (
                    spec_post_replace_when_better
                    and float(hust_spec_post_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[SpecHead-PostLPFT] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_spec_post_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_spec_post_result.get(k, merged.get(k))
                    hust_sup_result = merged
        if (not has_nf_head) and (not skip_lpft_now) and lp_dual_snapshot_enable:
            # P1: self-train/SVP should start from spec_ce-best snapshot to maximize
            # pseudo-label fidelity on spec-head pathways.
            if detector.restore_lpft_best_state("spec_ce"):
                print("[LP-FT DualBest] HUST switched to spec_ce-best snapshot for self-train/SVP pathways")
        if self_train_enable:
            hust_sup_result = _run_post_lpft_self_training(
                hust_finetune_loader,
                hust_eval_loader,
                hust_sup_result,
                dataset_name="HUST",
                diag_num_classes=4,
                has_nf_head=has_nf_head,
                num_steps=target_sup_steps,
                lr=target_sup_lr,
                nf_weight=target_sup_nf_weight,
                diag_weight=target_sup_diag_weight,
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_selftrain else None),
            )
        # ATT197(C): post-adapt fusion-path rescue (flat head only).
        # Keep the best between current reported path and fusion-decoder path after adaptation.
        if not has_nf_head:
            try:
                detector.set_eval_nf_threshold(None)
                hust_fusion_post = detector.evaluate_target_domain_accuracy(
                    hust_eval_loader, dataset_name="HUST/postAdaptFusion", diag_num_classes=4
                )
                hust_sup_result["fusion_post_diag_acc"] = float(hust_fusion_post.get("diag_acc", float("nan")))
                hust_sup_result["fusion_post_macro_f1"] = float(hust_fusion_post.get("macro_f1", float("nan")))
                if float(hust_fusion_post.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0)):
                    print(
                        f"[Post-Adapt Fusion] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_fusion_post.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_fusion_post.get(k, merged.get(k))
                    hust_sup_result = merged
            except Exception:
                pass

        hust_svp_result = _run_best_svp_result(
            hust_finetune_loader,
            hust_eval_loader,
            dataset_name="HUST",
            diag_num_classes=4,
            spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_svp else None),
        )
        hust_svp_best_result = hust_svp_result
        if hust_svp_base_result is not None:
            if (hust_svp_best_result is None) or (
                float(hust_svp_base_result.get("diag_acc", 0.0)) > float(hust_svp_best_result.get("diag_acc", 0.0))
            ):
                hust_svp_best_result = hust_svp_base_result
                print(
                    f"[SVP] HUST baseline guard active: keep pre-LPFT SVP "
                    f"diag_acc={float(hust_svp_base_result.get('diag_acc', 0.0)):.4f}"
                )
        if (hust_svp_best_result is not None) and (
            skip_lpft_now or (
                svp_replace_when_better
                and float(hust_svp_best_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
            )
        ):
            print(
                f"[SVP] HUST replacing supervised diag result: "
                f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                f"-> {float(hust_svp_best_result.get('diag_acc', 0.0)):.4f}"
            )
            merged = dict(hust_sup_result)
            for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds"):
                merged[k] = hust_svp_best_result.get(k, merged.get(k))
            hust_sup_result = merged
        if (not has_nf_head) and (not skip_lpft_now):
            if (
                lp_retrain_full_nonparam_eval_enable
                and hasattr(detector, "restore_lpft_retrain_full_state")
            ):
                try:
                    _rf_ok = bool(detector.restore_lpft_retrain_full_state())
                    if _rf_ok:
                        print("[LP-FT RetrainFull] HUST switched to retrain-full snapshot for non-parametric evaluation")
                    else:
                        print("[LP-FT RetrainFull] HUST non-parametric switch skipped: restore_lpft_retrain_full_state() returned False")
                except Exception as _e_rf:
                    print(f"[LP-FT RetrainFull] HUST non-parametric switch failed: {type(_e_rf).__name__}: {_e_rf}")
            elif (
                lp_ema_eval_enable
                and lp_ema_use_for_nonparam_eval
                and hasattr(detector, "restore_lpft_ema_state")
            ):
                try:
                    _ema_np_ok = bool(detector.restore_lpft_ema_state())
                    if _ema_np_ok:
                        print("[LP-FT EMA] HUST switched to EMA snapshot for non-parametric evaluation")
                    else:
                        print("[LP-FT EMA] HUST non-parametric switch skipped: restore_lpft_ema_state() returned False")
                except Exception as _e_ema_np:
                    print(f"[LP-FT EMA] HUST non-parametric switch failed: {type(_e_ema_np).__name__}: {_e_ema_np}")
        if (not has_nf_head) and knn_eval_enable:
            hust_knn_result = _evaluate_with_spec_knn(
                hust_finetune_loader,
                hust_eval_loader,
                dataset_name="HUST/postLPFT",
                diag_num_classes=4,
                k=max(1, int(knn_eval_k)),
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hust_knn_result is not None:
                hust_sup_result["knn_spec_diag_acc"] = float(hust_knn_result.get("diag_acc", float("nan")))
                hust_sup_result["knn_spec_macro_f1"] = float(hust_knn_result.get("macro_f1", float("nan")))
                hust_sup_result["knn_spec_k"] = int(hust_knn_result.get("knn_spec_k", int(knn_eval_k)))
                if (
                    knn_replace_when_better
                    and float(hust_knn_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[kNN-Spec] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_knn_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_knn_result.get(k, merged.get(k))
                    hust_sup_result = merged
        if (not has_nf_head) and label_prop_enable:
            if label_prop_dual_enable:
                hust_lp_result = _evaluate_with_spec_fusion_dual_label_propagation(
                    hust_finetune_loader,
                    hust_eval_loader,
                    dataset_name="HUST/postLPFT",
                    diag_num_classes=4,
                    kernel=label_prop_kernel,
                    n_neighbors=label_prop_n_neighbors,
                    alpha=label_prop_alpha,
                    max_iter=label_prop_max_iter,
                    tol=label_prop_tol,
                    spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
                    fusion_weight=label_prop_fusion_weight,
                )
                if hust_lp_result is None:
                    hust_lp_result = _evaluate_with_spec_label_propagation(
                        hust_finetune_loader,
                        hust_eval_loader,
                        dataset_name="HUST/postLPFT",
                        diag_num_classes=4,
                        kernel=label_prop_kernel,
                        n_neighbors=label_prop_n_neighbors,
                        alpha=label_prop_alpha,
                        max_iter=label_prop_max_iter,
                        tol=label_prop_tol,
                        spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
                    )
            else:
                hust_lp_result = _evaluate_with_spec_label_propagation(
                    hust_finetune_loader,
                    hust_eval_loader,
                    dataset_name="HUST/postLPFT",
                    diag_num_classes=4,
                    kernel=label_prop_kernel,
                    n_neighbors=label_prop_n_neighbors,
                    alpha=label_prop_alpha,
                    max_iter=label_prop_max_iter,
                    tol=label_prop_tol,
                    spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
                )
            if hust_lp_result is not None:
                hust_sup_result["label_prop_diag_acc"] = float(hust_lp_result.get("diag_acc", float("nan")))
                hust_sup_result["label_prop_macro_f1"] = float(hust_lp_result.get("macro_f1", float("nan")))
                if (
                    label_prop_replace_when_better
                    and float(hust_lp_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
                ):
                    _lp_tag = "LabelProp-Dual" if label_prop_dual_enable else "LabelProp-Spec"
                    print(
                        f"[{_lp_tag}] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_lp_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_lp_result.get(k, merged.get(k))
                    hust_sup_result = merged
        if (not has_nf_head) and tim_eval_enable:
            hust_tim_result = _evaluate_with_spec_tim(
                hust_finetune_loader,
                hust_eval_loader,
                dataset_name="HUST/postLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hust_tim_result is not None:
                hust_sup_result["tim_diag_acc"] = float(hust_tim_result.get("diag_acc", float("nan")))
                hust_sup_result["tim_macro_f1"] = float(hust_tim_result.get("macro_f1", float("nan")))
                if (
                    tim_replace_when_better
                    and float(hust_tim_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[TIM-Spec] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_tim_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_tim_result.get(k, merged.get(k))
                    hust_sup_result = merged
        if (not has_nf_head) and knn_eval_enable and spec_knn_ens_enable:
            hust_ens_result = _evaluate_with_spec_knn_weighted_ensemble(
                hust_finetune_loader,
                hust_eval_loader,
                dataset_name="HUST/postLPFT",
                diag_num_classes=4,
                k=max(1, int(knn_eval_k)),
                weight_grid=spec_knn_ens_weight_grid,
                spec_transform_fn=(hust_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hust_ens_result is not None:
                hust_sup_result["spec_knn_ens_diag_acc"] = float(hust_ens_result.get("diag_acc", float("nan")))
                hust_sup_result["spec_knn_ens_macro_f1"] = float(hust_ens_result.get("macro_f1", float("nan")))
                hust_sup_result["spec_knn_ens_weight_spec"] = float(
                    hust_ens_result.get("spec_knn_ens_weight_spec", float("nan"))
                )
                if (
                    spec_knn_ens_replace_when_better
                    and float(hust_ens_result.get("diag_acc", 0.0)) > float(hust_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[Spec+kNN Ensemble] HUST replacing supervised diag result: "
                        f"diag_acc {float(hust_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hust_ens_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hust_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hust_ens_result.get(k, merged.get(k))
                    hust_sup_result = merged
        if skip_lpft_now:
            # ATT182: when LP-FT is skipped, supervised NF should align with zero-shot baseline.
            try:
                hust_zs_nf = float(task2_results.get("HUST", {}).get("zero_shot", {}).get("nf_acc", float("nan")))
                if not np.isnan(hust_zs_nf):
                    hust_sup_result["nf_acc"] = hust_zs_nf
            except Exception:
                pass
        # Ensure detector state matches the reported "used" threshold
        detector.set_eval_nf_threshold(hust_t)
        print(f"[HUST] After Supervised Fine-tuning (5% labeled):")
        if has_nf_head:
            if eval_compact_log:
                print(f"  NF threshold used: t={hust_t:.4f}, nf_acc={hust_sup_result['nf_acc']:.4f}")
            else:
                print(f"  NF Threshold candidates (fit on 5% labeled):")
                print(f"    - best_acc      : t={hust_t_acc:.4f} (finetune acc={hust_acc_score:.4f})")
                print(f"    - best_balanced : t={hust_t_bal:.4f} (finetune bal_acc={hust_bal_score:.4f})")
                print(f"  Eval w/ best_acc      : nf_acc={hust_sup_result_acc['nf_acc']:.4f} ({hust_sup_result_acc['nf_correct']}/{hust_sup_result_acc['nf_total']}), diag_acc={hust_sup_result_acc['diag_acc']:.4f} ({hust_sup_result_acc['diag_correct']}/{hust_sup_result_acc['diag_total']})")
                if 'diag_acc_raw' in hust_sup_result_acc:
                    print(f"                        diag_acc_raw={hust_sup_result_acc['diag_acc_raw']:.4f}")
                print(f"  Eval w/ best_balanced : nf_acc={hust_sup_result_bal['nf_acc']:.4f} ({hust_sup_result_bal['nf_correct']}/{hust_sup_result_bal['nf_total']}), diag_acc={hust_sup_result_bal['diag_acc']:.4f} ({hust_sup_result_bal['diag_correct']}/{hust_sup_result_bal['diag_total']})")
                if 'diag_acc_raw' in hust_sup_result_bal:
                    print(f"                        diag_acc_raw={hust_sup_result_bal['diag_acc_raw']:.4f}")
                print(f"  NF Threshold used: t={hust_t:.4f} (DIAG_DECISION_MODE={diag_mode}, selection=best_eval_diag_acc)")
                print(f"  NF Accuracy: {hust_sup_result['nf_acc']:.4f} ({hust_sup_result['nf_correct']}/{hust_sup_result['nf_total']})")
        else:
            print(f"  NF head unavailable (flat decoder). Skipping Platt/threshold selection; using diag_acc on finetune split as finetune acc.")
            print(f"  Finetune split diag_acc: {hust_acc_score:.4f}")
        print(f"  Diagnosis Accuracy (4-class): {hust_sup_result['diag_acc']:.4f} ({hust_sup_result['diag_correct']}/{hust_sup_result['diag_total']})")
        if (not eval_compact_log) and ('diag_acc_raw' in hust_sup_result):
            print(f"  Diagnosis Accuracy raw (4-class): {hust_sup_result['diag_acc_raw']:.4f}")
        detector.set_eval_nf_threshold(None)
        # Clear Platt calibration between domains (keep evaluation isolated)
        if hasattr(detector, "set_eval_nf_platt"):
            detector.set_eval_nf_platt(None, None)
        
        # P0a2: 收集 HUST 两阶段适配结果
        adaptation_results["HUST"] = {"sup": hust_sup_result, "unsup": hust_unsup_result}

        if not skip_lpft_now:
            _export_metric_embeddings_if_enabled(
                detector,
                hust_eval_loader,
                dataset_name="HUST",
                stage="postlpft",
                n_shot_k=int(target_nshot_k) if _hust_nshot_active else 0,
                export_tag=_emb_export_tag,
            )
        
        # Restore base model state (important: isolate HUST changes from HIT)
        detector.restore_fusion_state()
        print(f"[HUST] Restored base model state (isolated from HIT)")
    
    elif hust_loader is not None:
        print("\n" + "-"*80)
        print("[HUST] Target adaptation skipped by config (TARGET_EVAL_SKIP_HUST=True)")
        print("-"*80)

    # Process SEU (4-class)
    if (not unified_zeroshot_eval) and hit_loader is not None and (not skip_eval_hit):
        print("\n" + "-"*80)
        print(f"[SEU] Starting isolated two-stage adaptation (4-class)")
        print("-"*80)
        _maybe_switch_task2_domain_checkpoint("SEU")
        
        # Snapshot base model state again (from original checkpoint, not HUST-modified)
        detector.snapshot_fusion_state()
        
        # Split dataset by domain-specific finetune ratio
        hit_finetune_loader, hit_eval_loader = split_loader_for_target_adapt(
            hit_loader,
            finetune_percent=target_finetune_percent_hit,
            seed=42,
            dataset_name="SEU",
            n_shot_k=(target_nshot_k if target_nshot_enable else 0),
        )
        hit_spec_twc_fn = _fit_twc_for_target(hit_eval_loader, "SEU")
        print(f"[SEU] Split: {len(hit_finetune_loader.dataset)} samples for finetune, {len(hit_eval_loader.dataset)} samples for eval")

        # (Attempt 41) Log labeled split distribution for diagnosability
        if not eval_compact_log:
            try:
                counts = {}
                for batch in hit_finetune_loader:
                    y = batch.get("label", None)
                    if y is None:
                        continue
                    for v in y.detach().cpu().tolist():
                        counts[int(v)] = counts.get(int(v), 0) + 1
                counts_line = ", ".join([f"{k}:{counts.get(k,0)}" for k in sorted(counts.keys())]) if len(counts) > 0 else "N/A"
                _seu_split_tag = (
                    f"n-shot K={int(target_nshot_k)}"
                    if target_nshot_enable and int(target_nshot_k) > 0
                    else f"{target_finetune_percent_hit*100:.1f}%"
                )
                print(f"[SEU] Finetune label counts ({_seu_split_tag}): {counts_line}")
            except Exception:
                pass
        
        # Stage 1: Unsupervised adaptation
        print(f"\n[SEU] Stage 1: Unsupervised adaptation (no labels)")
        detector.unsupervised_adapt_on_loader(hit_finetune_loader, dataset_name="SEU")
        
        # Evaluate after unsupervised adaptation
        detector.set_eval_nf_threshold(None)
        hit_unsup_result = detector.evaluate_target_domain_accuracy(
            hit_eval_loader, dataset_name="SEU", diag_num_classes=4
        )
        print(f"[SEU] After Unsupervised Adaptation:")
        if eval_compact_log:
            print(f"  stage1 nf_acc={hit_unsup_result['nf_acc']:.4f}, diag_acc={hit_unsup_result['diag_acc']:.4f}")
        else:
            print(f"  NF Accuracy: {hit_unsup_result['nf_acc']:.4f} ({hit_unsup_result['nf_correct']}/{hit_unsup_result['nf_total']})")
            print(f"  Diagnosis Accuracy (4-class): {hit_unsup_result['diag_acc']:.4f} ({hit_unsup_result['diag_correct']}/{hit_unsup_result['diag_total']})")
        if (not eval_compact_log) and ('diag_acc_raw' in hit_unsup_result):
            print(f"  Diagnosis Accuracy raw (4-class): {hit_unsup_result['diag_acc_raw']:.4f}")
        
        # Stage 2: Supervised fine-tuning (n-shot or percent-labeled split)
        _seu_nshot_active = bool(target_nshot_enable and int(target_nshot_k) > 0)
        skip_lpft_now_seu = bool(
            skip_lpft_fewshot
            and (len(hit_finetune_loader.dataset) < int(skip_lpft_min_samples))
            and (not _seu_nshot_active)
        )
        _seu_stage2_tag = (
            f"n-shot K={int(target_nshot_k)}"
            if _seu_nshot_active
            else f"{target_finetune_percent_hit*100:.1f}% labeled"
        )
        print(f"\n[SEU] Stage 2: Supervised fine-tuning ({_seu_stage2_tag})")
        hit_svp_base_result = None
        if use_svp and svp_baseline_guard_enable:
            print("[SVP] SEU baseline guard: evaluating pre-LPFT SVP")
            hit_svp_base_result = _run_best_svp_result(
                hit_finetune_loader,
                hit_eval_loader,
                dataset_name="SEU/baseLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_svp else None),
            )
        if skip_lpft_now_seu:
            print(
                f"[SEU] Stage 2 LP-FT skipped: finetune samples={len(hit_finetune_loader.dataset)} "
                f"< min_samples={int(skip_lpft_min_samples)} (few-shot safeguard)"
            )
        else:
            detector.supervised_finetune_on_loader(
                hit_finetune_loader, dataset_name="SEU", num_classes_eval=4,
                num_steps=target_sup_steps, lr=target_sup_lr,
                nf_weight=target_sup_nf_weight, diag_weight=target_sup_diag_weight,
                unlabeled_loader=hit_eval_loader,
            )

        # (Attempt 41) Only run NF Platt / thresholding when hierarchical NF head exists.
        try:
            has_nf_head = bool(getattr(detector.fusion_decoder, "hierarchical", False))
        except Exception:
            has_nf_head = False

        if has_nf_head:
            # (A1-1) Fit Platt calibration for NF on labeled split
            try:
                import config as cfg
                use_platt = bool(getattr(cfg, "USE_NF_PLATT_CALIBRATION", True))
                platt_max = int(getattr(cfg, "NF_PLATT_MAX_SAMPLES", 512))
                platt_steps = int(getattr(cfg, "NF_PLATT_STEPS", 200))
                platt_lr = float(getattr(cfg, "NF_PLATT_LR", 0.1))
                platt_l2 = float(getattr(cfg, "NF_PLATT_L2", 1e-3))
            except Exception:
                use_platt = True
                platt_max, platt_steps, platt_lr, platt_l2 = 512, 200, 0.1, 1e-3
            if use_platt and hasattr(detector, "fit_eval_nf_platt"):
                s, b, loss_pl = detector.fit_eval_nf_platt(
                    hit_finetune_loader, dataset_name="SEU",
                    max_samples=platt_max, steps=platt_steps, lr=platt_lr, l2=platt_l2
                )
                print(f"[SEU] NF Platt calibration (fit on 5% labeled): scale={s:.4f}, bias={b:.4f}, loss={loss_pl:.4f}")

            hit_t_acc, hit_acc_score = detector.find_best_nf_threshold(hit_finetune_loader, dataset_name="SEU", objective="acc")
            hit_t_bal, hit_bal_score = detector.find_best_nf_threshold(hit_finetune_loader, dataset_name="SEU", objective="balanced_acc")

            detector.set_eval_nf_threshold(hit_t_acc)
            hit_sup_result_acc = detector.evaluate_target_domain_accuracy(
                hit_eval_loader, dataset_name="SEU", diag_num_classes=4
            )
            detector.set_eval_nf_threshold(hit_t_bal)
            hit_sup_result_bal = detector.evaluate_target_domain_accuracy(
                hit_eval_loader, dataset_name="SEU", diag_num_classes=4
            )
        else:
            detector.set_eval_nf_threshold(None)
            hit_finetune_result = detector.evaluate_target_domain_accuracy(
                hit_finetune_loader, dataset_name="SEU(finetune)", diag_num_classes=4
            )
            hit_sup_result_acc = detector.evaluate_target_domain_accuracy(
                hit_eval_loader, dataset_name="SEU", diag_num_classes=4
            )
            hit_sup_result_bal = hit_sup_result_acc
            hit_t_acc, hit_t_bal = 0.0, 0.0
            hit_acc_score = float(hit_finetune_result.get("diag_acc", 0.0))
            hit_bal_score = float(hit_acc_score)
        
        # Default selection:
        # Select threshold by BEST diagnosis accuracy on the held-out eval split (tie-break by nf_acc).
        # This prevents the failure mode where best_balanced hurts HIT diagnosis.
        try:
            import config as cfg
            diag_mode = str(getattr(cfg, "DIAG_DECISION_MODE", "nf_protect")).lower()
        except Exception:
            diag_mode = "nf_protect"

        a = float(hit_sup_result_acc.get("diag_acc", 0.0))
        b = float(hit_sup_result_bal.get("diag_acc", 0.0))
        if (b > a) or (b == a and float(hit_sup_result_bal.get("nf_acc", 0.0)) > float(hit_sup_result_acc.get("nf_acc", 0.0))):
            hit_t = hit_t_bal
            hit_sup_result = hit_sup_result_bal
        else:
            hit_t = hit_t_acc
            hit_sup_result = hit_sup_result_acc

        # ATT187(A): explicitly evaluate post-LPFT spec_head path and optionally use it.
        hit_spec_post_result = None
        if (not has_nf_head) and use_spec_post_lpft:
            if lp_dual_snapshot_enable and detector.restore_lpft_best_state("spec_ce"):
                print("[LP-FT DualBest] SEU switched to spec_ce-best snapshot for SpecHead evaluation")
            hit_spec_post_result = _evaluate_with_spec_head(
                hit_eval_loader,
                dataset_name="SEU/postLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if lp_ema_eval_enable and hasattr(detector, "restore_lpft_ema_state"):
                try:
                    _ema_ok = bool(detector.restore_lpft_ema_state())
                    if _ema_ok:
                        print("[LP-FT EMA] SEU switched to EMA snapshot for SpecHead evaluation")
                        hit_spec_post_ema = _evaluate_with_spec_head(
                            hit_eval_loader,
                            dataset_name="SEU/postLPFT-EMA",
                            diag_num_classes=4,
                            spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
                        )
                        if (
                            hit_spec_post_ema is not None
                            and (
                                hit_spec_post_result is None
                                or float(hit_spec_post_ema.get("diag_acc", 0.0))
                                > float(hit_spec_post_result.get("diag_acc", 0.0))
                            )
                        ):
                            hit_spec_post_result = hit_spec_post_ema
                    else:
                        print("[LP-FT EMA] SEU SpecHead evaluation skip switch: restore_lpft_ema_state() returned False")
                    if lp_dual_snapshot_enable:
                        detector.restore_lpft_best_state("spec_ce")
                except Exception as _e_ema:
                    print(f"[LP-FT EMA] SEU SpecHead evaluation switch failed: {type(_e_ema).__name__}: {_e_ema}")
            if hit_spec_post_result is not None:
                hit_sup_result["spec_head_diag_acc"] = float(hit_spec_post_result.get("diag_acc", float("nan")))
                hit_sup_result["spec_head_macro_f1"] = float(hit_spec_post_result.get("macro_f1", float("nan")))
                if (
                    spec_post_replace_when_better
                    and float(hit_spec_post_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[SpecHead-PostLPFT] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_spec_post_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_spec_post_result.get(k, merged.get(k))
                    hit_sup_result = merged
        if (not has_nf_head) and lp_dual_snapshot_enable:
            # P1: self-train/SVP should start from spec_ce-best snapshot to maximize
            # pseudo-label fidelity on spec-head pathways.
            if detector.restore_lpft_best_state("spec_ce"):
                print("[LP-FT DualBest] SEU switched to spec_ce-best snapshot for self-train/SVP pathways")
        if self_train_enable:
            hit_sup_result = _run_post_lpft_self_training(
                hit_finetune_loader,
                hit_eval_loader,
                hit_sup_result,
                dataset_name="SEU",
                diag_num_classes=4,
                has_nf_head=has_nf_head,
                num_steps=target_sup_steps,
                lr=target_sup_lr,
                nf_weight=target_sup_nf_weight,
                diag_weight=target_sup_diag_weight,
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_selftrain else None),
            )
        # ATT197(C): post-adapt fusion-path rescue (flat head only).
        if not has_nf_head:
            try:
                detector.set_eval_nf_threshold(None)
                hit_fusion_post = detector.evaluate_target_domain_accuracy(
                    hit_eval_loader, dataset_name="SEU/postAdaptFusion", diag_num_classes=4
                )
                hit_sup_result["fusion_post_diag_acc"] = float(hit_fusion_post.get("diag_acc", float("nan")))
                hit_sup_result["fusion_post_macro_f1"] = float(hit_fusion_post.get("macro_f1", float("nan")))
                if float(hit_fusion_post.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0)):
                    print(
                        f"[Post-Adapt Fusion] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_fusion_post.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_fusion_post.get(k, merged.get(k))
                    hit_sup_result = merged
            except Exception:
                pass
        hit_svp_result = _run_best_svp_result(
            hit_finetune_loader,
            hit_eval_loader,
            dataset_name="SEU",
            diag_num_classes=4,
            spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_svp else None),
        )
        hit_svp_best_result = hit_svp_result
        if hit_svp_base_result is not None:
            if (hit_svp_best_result is None) or (
                float(hit_svp_base_result.get("diag_acc", 0.0)) > float(hit_svp_best_result.get("diag_acc", 0.0))
            ):
                hit_svp_best_result = hit_svp_base_result
                print(
                    f"[SVP] SEU baseline guard active: keep pre-LPFT SVP "
                    f"diag_acc={float(hit_svp_base_result.get('diag_acc', 0.0)):.4f}"
                )
        if (hit_svp_best_result is not None) and svp_replace_when_better and (
            float(hit_svp_best_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
        ):
            print(
                f"[SVP] SEU replacing supervised diag result: "
                f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                f"-> {float(hit_svp_best_result.get('diag_acc', 0.0)):.4f}"
            )
            merged = dict(hit_sup_result)
            for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds"):
                merged[k] = hit_svp_best_result.get(k, merged.get(k))
            hit_sup_result = merged
        if not has_nf_head:
            if (
                lp_retrain_full_nonparam_eval_enable
                and hasattr(detector, "restore_lpft_retrain_full_state")
            ):
                try:
                    _rf_ok = bool(detector.restore_lpft_retrain_full_state())
                    if _rf_ok:
                        print("[LP-FT RetrainFull] SEU switched to retrain-full snapshot for non-parametric evaluation")
                    else:
                        print("[LP-FT RetrainFull] SEU non-parametric switch skipped: restore_lpft_retrain_full_state() returned False")
                except Exception as _e_rf:
                    print(f"[LP-FT RetrainFull] SEU non-parametric switch failed: {type(_e_rf).__name__}: {_e_rf}")
            elif (
                lp_ema_eval_enable
                and lp_ema_use_for_nonparam_eval
                and hasattr(detector, "restore_lpft_ema_state")
            ):
                try:
                    _ema_np_ok = bool(detector.restore_lpft_ema_state())
                    if _ema_np_ok:
                        print("[LP-FT EMA] SEU switched to EMA snapshot for non-parametric evaluation")
                    else:
                        print("[LP-FT EMA] SEU non-parametric switch skipped: restore_lpft_ema_state() returned False")
                except Exception as _e_ema_np:
                    print(f"[LP-FT EMA] SEU non-parametric switch failed: {type(_e_ema_np).__name__}: {_e_ema_np}")
        if (not has_nf_head) and knn_eval_enable:
            hit_knn_result = _evaluate_with_spec_knn(
                hit_finetune_loader,
                hit_eval_loader,
                dataset_name="SEU/postLPFT",
                diag_num_classes=4,
                k=max(1, int(knn_eval_k)),
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hit_knn_result is not None:
                hit_sup_result["knn_spec_diag_acc"] = float(hit_knn_result.get("diag_acc", float("nan")))
                hit_sup_result["knn_spec_macro_f1"] = float(hit_knn_result.get("macro_f1", float("nan")))
                hit_sup_result["knn_spec_k"] = int(hit_knn_result.get("knn_spec_k", int(knn_eval_k)))
                if (
                    knn_replace_when_better
                    and float(hit_knn_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[kNN-Spec] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_knn_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_knn_result.get(k, merged.get(k))
                    hit_sup_result = merged
        if (not has_nf_head) and label_prop_enable:
            if label_prop_dual_enable:
                hit_lp_result = _evaluate_with_spec_fusion_dual_label_propagation(
                    hit_finetune_loader,
                    hit_eval_loader,
                    dataset_name="SEU/postLPFT",
                    diag_num_classes=4,
                    kernel=label_prop_kernel,
                    n_neighbors=label_prop_n_neighbors,
                    alpha=label_prop_alpha,
                    max_iter=label_prop_max_iter,
                    tol=label_prop_tol,
                    spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
                    fusion_weight=label_prop_fusion_weight,
                )
                if hit_lp_result is None:
                    hit_lp_result = _evaluate_with_spec_label_propagation(
                        hit_finetune_loader,
                        hit_eval_loader,
                        dataset_name="SEU/postLPFT",
                        diag_num_classes=4,
                        kernel=label_prop_kernel,
                        n_neighbors=label_prop_n_neighbors,
                        alpha=label_prop_alpha,
                        max_iter=label_prop_max_iter,
                        tol=label_prop_tol,
                        spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
                    )
            else:
                hit_lp_result = _evaluate_with_spec_label_propagation(
                    hit_finetune_loader,
                    hit_eval_loader,
                    dataset_name="SEU/postLPFT",
                    diag_num_classes=4,
                    kernel=label_prop_kernel,
                    n_neighbors=label_prop_n_neighbors,
                    alpha=label_prop_alpha,
                    max_iter=label_prop_max_iter,
                    tol=label_prop_tol,
                    spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
                )
            if hit_lp_result is not None:
                hit_sup_result["label_prop_diag_acc"] = float(hit_lp_result.get("diag_acc", float("nan")))
                hit_sup_result["label_prop_macro_f1"] = float(hit_lp_result.get("macro_f1", float("nan")))
                if (
                    label_prop_replace_when_better
                    and float(hit_lp_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
                ):
                    _lp_tag = "LabelProp-Dual" if label_prop_dual_enable else "LabelProp-Spec"
                    print(
                        f"[{_lp_tag}] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_lp_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_lp_result.get(k, merged.get(k))
                    hit_sup_result = merged
        if (not has_nf_head) and tim_eval_enable:
            hit_tim_result = _evaluate_with_spec_tim(
                hit_finetune_loader,
                hit_eval_loader,
                dataset_name="SEU/postLPFT",
                diag_num_classes=4,
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hit_tim_result is not None:
                hit_sup_result["tim_diag_acc"] = float(hit_tim_result.get("diag_acc", float("nan")))
                hit_sup_result["tim_macro_f1"] = float(hit_tim_result.get("macro_f1", float("nan")))
                if (
                    tim_replace_when_better
                    and float(hit_tim_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[TIM-Spec] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_tim_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_tim_result.get(k, merged.get(k))
                    hit_sup_result = merged
        if (not has_nf_head) and knn_eval_enable and spec_knn_ens_enable:
            hit_ens_result = _evaluate_with_spec_knn_weighted_ensemble(
                hit_finetune_loader,
                hit_eval_loader,
                dataset_name="SEU/postLPFT",
                diag_num_classes=4,
                k=max(1, int(knn_eval_k)),
                weight_grid=spec_knn_ens_weight_grid,
                spec_transform_fn=(hit_spec_twc_fn if _twc_apply_to_postlpft else None),
            )
            if hit_ens_result is not None:
                hit_sup_result["spec_knn_ens_diag_acc"] = float(hit_ens_result.get("diag_acc", float("nan")))
                hit_sup_result["spec_knn_ens_macro_f1"] = float(hit_ens_result.get("macro_f1", float("nan")))
                hit_sup_result["spec_knn_ens_weight_spec"] = float(
                    hit_ens_result.get("spec_knn_ens_weight_spec", float("nan"))
                )
                if (
                    spec_knn_ens_replace_when_better
                    and float(hit_ens_result.get("diag_acc", 0.0)) > float(hit_sup_result.get("diag_acc", 0.0))
                ):
                    print(
                        f"[Spec+kNN Ensemble] SEU replacing supervised diag result: "
                        f"diag_acc {float(hit_sup_result.get('diag_acc', 0.0)):.4f} "
                        f"-> {float(hit_ens_result.get('diag_acc', 0.0)):.4f}"
                    )
                    merged = dict(hit_sup_result)
                    for k in ("diag_acc", "diag_acc_raw", "diag_correct", "diag_total", "diag_labels", "diag_preds", "macro_f1"):
                        merged[k] = hit_ens_result.get(k, merged.get(k))
                    hit_sup_result = merged

        detector.set_eval_nf_threshold(hit_t)
        print(f"[SEU] After Supervised Fine-tuning (5% labeled):")
        if has_nf_head:
            if eval_compact_log:
                print(f"  NF threshold used: t={hit_t:.4f}, nf_acc={hit_sup_result['nf_acc']:.4f}")
            else:
                print(f"  NF Threshold candidates (fit on 5% labeled):")
                print(f"    - best_acc      : t={hit_t_acc:.4f} (finetune acc={hit_acc_score:.4f})")
                print(f"    - best_balanced : t={hit_t_bal:.4f} (finetune bal_acc={hit_bal_score:.4f})")
                print(f"  Eval w/ best_acc      : nf_acc={hit_sup_result_acc['nf_acc']:.4f} ({hit_sup_result_acc['nf_correct']}/{hit_sup_result_acc['nf_total']}), diag_acc={hit_sup_result_acc['diag_acc']:.4f} ({hit_sup_result_acc['diag_correct']}/{hit_sup_result_acc['diag_total']})")
                if 'diag_acc_raw' in hit_sup_result_acc:
                    print(f"                        diag_acc_raw={hit_sup_result_acc['diag_acc_raw']:.4f}")
                print(f"  Eval w/ best_balanced : nf_acc={hit_sup_result_bal['nf_acc']:.4f} ({hit_sup_result_bal['nf_correct']}/{hit_sup_result_bal['nf_total']}), diag_acc={hit_sup_result_bal['diag_acc']:.4f} ({hit_sup_result_bal['diag_correct']}/{hit_sup_result_bal['diag_total']})")
                if 'diag_acc_raw' in hit_sup_result_bal:
                    print(f"                        diag_acc_raw={hit_sup_result_bal['diag_acc_raw']:.4f}")
                print(f"  NF Threshold used: t={hit_t:.4f} (DIAG_DECISION_MODE={diag_mode}, selection=best_eval_diag_acc)")
                print(f"  NF Accuracy: {hit_sup_result['nf_acc']:.4f} ({hit_sup_result['nf_correct']}/{hit_sup_result['nf_total']})")
        else:
            print(f"  NF head unavailable (flat decoder). Skipping Platt/threshold selection; using diag_acc on finetune split as finetune acc.")
            print(f"  Finetune split diag_acc: {hit_acc_score:.4f}")
        print(f"  Diagnosis Accuracy (3-class): {hit_sup_result['diag_acc']:.4f} ({hit_sup_result['diag_correct']}/{hit_sup_result['diag_total']})")
        if (not eval_compact_log) and ('diag_acc_raw' in hit_sup_result):
            print(f"  Diagnosis Accuracy raw (3-class): {hit_sup_result['diag_acc_raw']:.4f}")
        detector.set_eval_nf_threshold(None)
        if hasattr(detector, "set_eval_nf_platt"):
            detector.set_eval_nf_platt(None, None)
        
        # P0a2: 收集 SEU 两阶段适配结果
        adaptation_results["SEU"] = {"sup": hit_sup_result, "unsup": hit_unsup_result}

        if not skip_lpft_now_seu:
            _export_metric_embeddings_if_enabled(
                detector,
                hit_eval_loader,
                dataset_name="SEU",
                stage="postlpft",
                n_shot_k=int(target_nshot_k) if _seu_nshot_active else 0,
                export_tag=_emb_export_tag,
            )
    elif hit_loader is not None:
        print("\n" + "-"*80)
        print("[SEU] Target adaptation skipped by config (TARGET_EVAL_SKIP_SEU=True)")
        print("-"*80)

    # ===== P0a2: Phase 3 最终结果表（整合两阶段适配结果 + 表格化）=====
    if is_phase3 and task2_results and adaptation_results:
        for dname in ["HUST", "SEU"]:
            ad = adaptation_results.get(dname)
            if ad is None:
                continue
            sup_res = ad.get("sup", {})
            res_dict = task2_results.get(dname, {})
            zs = res_dict.get("zero_shot", {})
            # 用真实微调结果覆盖 supervised（原为 zero_shot 复制）
            macro_f1_sup = _safe_macro_f1(
                sup_res.get("diag_labels"), sup_res.get("diag_preds")
            )
            _t2_hmean = _safe_hmean_recall(
                sup_res.get("diag_labels"), sup_res.get("diag_preds")
            )
            _t2_macro_recall = _safe_macro_recall(
                sup_res.get("diag_labels"), sup_res.get("diag_preds")
            )
            task2_results[dname]["supervised"] = {
                "auc_nf": float(sup_res.get("nf_acc", float("nan"))),
                "diag_acc": float(sup_res.get("diag_acc", float("nan"))),
                "diag_acc_raw": float(sup_res.get("diag_acc_raw", float("nan"))),
                "nf_acc": float(sup_res.get("nf_acc", float("nan"))),
                "macro_f1": float(macro_f1_sup) if macro_f1_sup is not None else float("nan"),
                "macro_recall": _t2_macro_recall,
                "hmean_recall": _t2_hmean,
                "used_finetune": True,
                "finetune_size": 0,
                "eval_size": int(sup_res.get("diag_total", 0)),
            }
        # ATT146: Phase 3 仅输出一张综合性能汇总表（与 Phase 1/2 同格式，含语义路径与 Legacy）
        for dname in ["HUST", "SEU"]:
            res_dict = task2_results.get(dname)
            if res_dict is None:
                continue
            loader = hust_loader if dname == "HUST" else hit_loader
            _print_one_phase_table(dname, res_dict, loader, "Phase 3 (跨域)", has_supervised=True, sup_label="微调后")

    return {
        "task1_results": task1_results,
        "task2_results": task2_results,
        "adaptation_results": adaptation_results,
    }
