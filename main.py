"""
主训练脚本 - 重构后的简洁版本
"""
import csv
import os
import re
from datetime import datetime

import torch.utils.data

# 设置 Hugging Face 镜像站点（必须在导入 sentence_transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置 tokenizers 并行性，避免多进程环境下的警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from utils.splitData import (
    split_dataset_train_val_test,
    split_dataset_train_val_test_stratified,
    split_by_condition_groups,
    subset_by_condition,
)
from training.train_alignment import train_alignment
from config import Config
import config as config_module  # ATT176: 用于 Task2 覆盖 SPEC_ALIGN_WARMUP_END_EPOCH 模块级变量（train_alignment 通过 import config as cfg 读取）
from collate import multimodal_collate_fn
from utils.training_setup import (
    setup_logging,
    load_and_preprocess_datasets,
    compute_class_weights_and_sampler,
    setup_dataloaders
)
from utils.model_factory import create_models
from utils.training_utils import (
    infer_checkpoint_tag,
    load_model_checkpoint,
    build_train_alignment_kwargs,
    load_checkpoint_prototypes
)
from utils.evaluation_utils import run_evaluation
from utils.checkpoint import resolve_readable_checkpoint
from utils.seed_utils import set_seed
from utils.optimizer_utils import assigned_parameter_ids, sanitize_optimizer_param_groups, unique_parameters
from zero_shot_eval.zero_shot_eval_hust import zero_shot_eval_hust
from training.sam_optimizer import SAM

# 使用配置类实例
cfg = Config

def resolve_eval_checkpoint(best_ckpt_path: str):
    """
    评估时优先使用 SWAD checkpoint（若存在），否则回退到 val-loss best checkpoint。
    """
    if not best_ckpt_path:
        return best_ckpt_path
    use_swad = bool(getattr(cfg, "EVAL_USE_SWAD_CHECKPOINT", False))
    if not use_swad:
        return best_ckpt_path
    # ATT157: train_alignment 保存命名为 epoch_{E}_swad.pth，
    # 不能直接对 epoch_{E}_best.pth 追加后缀，否则会拼成 epoch_{E}_best_swad.pth（不存在）。
    epoch_match = re.search(r"epoch_(\d+)", os.path.basename(best_ckpt_path))
    if not epoch_match:
        return best_ckpt_path
    swad_ckpt = os.path.join(
        os.path.dirname(best_ckpt_path),
        f"epoch_{epoch_match.group(1)}_swad.pth",
    )
    if os.path.exists(swad_ckpt):
        print(f"[Eval Ckpt] Using SWAD checkpoint for evaluation: {swad_ckpt}")
        return swad_ckpt
    return best_ckpt_path


def _coerce_history_best_epoch(best_epoch, best_ckpt_path) -> int | None:
    """
    train_alignment returns int epoch; EVAL_ONLY / PHASE_SKIP use str markers
    (e.g. 'eval_only', 'skipped'). Fall back to parsing epoch_N from filename.
    """
    if best_epoch is None:
        return None
    if isinstance(best_epoch, int):
        return int(best_epoch)
    if isinstance(best_epoch, str):
        s = best_epoch.strip()
        if s.isdigit():
            return int(s)
    if best_ckpt_path:
        m = re.search(r"epoch_(\d+)", os.path.basename(str(best_ckpt_path)))
        if m:
            return int(m.group(1))
    return None


if __name__ == '__main__':
    # ===== 0. 设置随机种子（必须在所有随机操作之前） =====
    set_seed(cfg.RANDOM_SEED)
    
    # ===== 1. 设置日志 =====
    setup_logging(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        config_module.print_framework_group_summary(prefix="[FrameworkGroups]")
    except Exception:
        pass
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"[GPU] device={gpu_name} vram={gpu_mem_gb:.1f}GB")
    else:
        print("[GPU] CUDA not available, using CPU")

    # ===== 2. 加载和预处理数据集 =====
    cwru_dataset, pu_dataset, hust_dataset, seu_dataset, hit_dataset, processor = load_and_preprocess_datasets(cfg)
    
    # 训练集划分：支持 Task1（域内跨工况）与原有路径。
    # 当启用跨工况任务时，source-domain 的 test split 是“留出工况 zero-shot”。
    enable_intra_condition = bool(getattr(cfg, "ENABLE_INTRA_CONDITION_TASK", True))
    intra_val_ratio = float(getattr(cfg, "INTRA_CONDITION_VAL_RATIO", 0.2))
    run_all_phases_requested = bool(getattr(cfg, "TASK1_RUN_ALL_PHASES_IN_ONE_SCRIPT", False))
    loo_requested = bool(getattr(cfg, "LOO_ENABLE", False))
    stmo_requested = bool(getattr(cfg, "STMO_ENABLE", False))
    strict_intra_split = bool(getattr(cfg, "STRICT_INTRA_CONDITION_SPLIT", True))
    _train_subsample_ratio = float(getattr(cfg, "TASK1_TRAIN_SUBSAMPLE_RATIO", 1.0))
    _train_subsample_ratio_pu = float(
        getattr(cfg, "TASK1_TRAIN_SUBSAMPLE_RATIO_PU", _train_subsample_ratio)
    )

    def _train_subsample_ratio_for(domain_name: str) -> float:
        if str(domain_name).upper() == "PU":
            return _train_subsample_ratio_pu
        return _train_subsample_ratio

    if abs(_train_subsample_ratio_pu - _train_subsample_ratio) > 1e-9:
        print(
            f"[TrainSubsample] per-domain: default={_train_subsample_ratio:.2f}, "
            f"PU={_train_subsample_ratio_pu:.2f}"
        )

    def _split_domain(dataset, domain_name, seed):
        _train_groups = getattr(cfg, f"{domain_name}_TRAIN_GROUPS", None)
        _test_groups = getattr(cfg, f"{domain_name}_TEST_GROUPS", None)
        _sub_ratio = _train_subsample_ratio_for(domain_name)
        return split_by_condition_groups(
            dataset,
            condition_ids=getattr(dataset, "condition_ids", None),
            train_groups=_train_groups,
            test_groups=_test_groups,
            val_ratio_from_train=intra_val_ratio,
            seed=seed,
            train_subsample_ratio=_sub_ratio,
            subsample_label=str(domain_name),
        )

    def _summarize_domain_conditions(dataset, domain_name):
        if dataset is None or not hasattr(dataset, "condition_ids"):
            print(f"[ConditionGroups] {domain_name}: no condition_ids (dataset unavailable)")
            return 0
        _ids = [int(x) for x in getattr(dataset, "condition_ids", [])]
        _uniq = sorted(set(_ids))
        _id2key = getattr(dataset, "condition_id_to_key", {})
        _keys = [str(_id2key.get(i, i)) for i in _uniq]
        _preview = _keys[:12]
        _suffix = "..." if len(_keys) > 12 else ""
        print(f"[ConditionGroups] {domain_name}: n_groups={len(_uniq)} groups={_preview}{_suffix}")
        return len(_uniq)

    def _split_or_fallback(dataset, domain_name, seed):
        _n_groups = _summarize_domain_conditions(dataset, domain_name)
        if _n_groups < 2:
            _msg = (
                f"{domain_name} has <2 condition groups (n={_n_groups}); "
                "cannot run intra-condition split/LOO."
            )
            if loo_requested or stmo_requested or run_all_phases_requested or strict_intra_split:
                raise RuntimeError(_msg)
            print(f"[Task1 Split] Warning: {_msg} Fallback to stratified random split.")
            return split_dataset_train_val_test_stratified(
                dataset, train_ratio=0.6, val_ratio=0.2, seed=seed
            ), {"train_groups": [], "test_groups": []}
        try:
            _tr, _vl, _te, _meta = _split_domain(dataset, domain_name, seed)
            return (_tr, _vl, _te), _meta
        except Exception as _e:
            _msg = f"{domain_name} condition split failed ({_e})"
            if loo_requested or stmo_requested or run_all_phases_requested or strict_intra_split:
                raise RuntimeError(_msg) from _e
            print(f"[Task1 Split] Warning: {_msg}, fallback to stratified random split.")
            return split_dataset_train_val_test_stratified(
                dataset, train_ratio=0.6, val_ratio=0.2, seed=seed
            ), {"train_groups": [], "test_groups": []}

    if enable_intra_condition:
        try:
            (cwru_train, cwru_val, cwru_test), cwru_split_meta = _split_or_fallback(
                cwru_dataset, "CWRU", cfg.RANDOM_SEED
            )
            (pu_train, pu_val, pu_test), pu_split_meta = _split_or_fallback(
                pu_dataset, "PU", cfg.RANDOM_SEED + 1
            )
            (hust_train, hust_val, hust_test), hust_split_meta = _split_or_fallback(
                hust_dataset, "HUST", cfg.RANDOM_SEED + 2
            )
            if seu_dataset is not None:
                (seu_train, seu_val, seu_test), seu_split_meta = _split_or_fallback(
                    seu_dataset, "SEU", cfg.RANDOM_SEED + 3
                )
            else:
                seu_train = seu_val = seu_test = None
                seu_split_meta = None
            if hit_dataset is not None:
                (hit_train, hit_val, hit_test), hit_split_meta = _split_or_fallback(
                    hit_dataset, "HIT", cfg.RANDOM_SEED + 4
                )
            else:
                hit_train = hit_val = hit_test = None
                hit_split_meta = None
            try:
                for _name, _dataset, _meta in [
                    ("CWRU", cwru_dataset, cwru_split_meta),
                    ("PU", pu_dataset, pu_split_meta),
                    ("HUST", hust_dataset, hust_split_meta),
                ]:
                    _map = getattr(_dataset, "condition_id_to_key", {})
                    _tr = [_map.get(i, str(i)) for i in _meta["train_groups"]]
                    _te = [_map.get(i, str(i)) for i in _meta["test_groups"]]
                    print(f"[Task1 Split] {_name} train_groups:", _tr, "test_groups:", _te)
                if seu_dataset is not None and seu_split_meta is not None:
                    _map = getattr(seu_dataset, "condition_id_to_key", {})
                    _tr = [_map.get(i, str(i)) for i in seu_split_meta["train_groups"]]
                    _te = [_map.get(i, str(i)) for i in seu_split_meta["test_groups"]]
                    print("[Task1 Split] SEU  train_groups:", _tr, "test_groups:", _te)
                if hit_dataset is not None and hit_split_meta is not None:
                    _map = getattr(hit_dataset, "condition_id_to_key", {})
                    _tr = [_map.get(i, str(i)) for i in hit_split_meta["train_groups"]]
                    _te = [_map.get(i, str(i)) for i in hit_split_meta["test_groups"]]
                    print("[Task1 Split] HIT  train_groups:", _tr, "test_groups:", _te)
            except Exception:
                pass
        except Exception as e:
            if loo_requested or stmo_requested or run_all_phases_requested or strict_intra_split:
                raise RuntimeError(
                    f"[Task1 Split] Critical: strict intra-condition protocol violated: {e}. "
                    "Abort to avoid accidental joint/random-split experiment."
                ) from e
            print(f"[Task1 Split] Warning: condition split failed ({e}), fallback to stratified random split.")
            enable_intra_condition = False
            cwru_train, cwru_val, cwru_test = split_dataset_train_val_test_stratified(
                cwru_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED
            )
            pu_train, pu_val, pu_test = split_dataset_train_val_test_stratified(
                pu_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED
            )
            hust_train, hust_val, hust_test = split_dataset_train_val_test_stratified(
                hust_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 2
            )
            if seu_dataset is not None:
                seu_train, seu_val, seu_test = split_dataset_train_val_test_stratified(
                    seu_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 3
                )
            else:
                seu_train = seu_val = seu_test = None
            if hit_dataset is not None:
                hit_train, hit_val, hit_test = split_dataset_train_val_test_stratified(
                    hit_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 4
                )
            else:
                hit_train = hit_val = hit_test = None
    else:
        # 回退旧路径：随机分层切分
        if loo_requested or stmo_requested or run_all_phases_requested or strict_intra_split:
            raise RuntimeError(
                "ENABLE_INTRA_CONDITION_TASK=0 but strict intra-condition/LOO/STMO protocol is requested. "
                "Set ENABLE_INTRA_CONDITION_TASK=1 and ensure condition extraction is valid."
            )
        cwru_train, cwru_val, cwru_test = split_dataset_train_val_test_stratified(
            cwru_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED
        )
        pu_train, pu_val, pu_test = split_dataset_train_val_test_stratified(
                pu_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED
            )
        hust_train, hust_val, hust_test = split_dataset_train_val_test_stratified(
            hust_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 2
        )
        if seu_dataset is not None:
            seu_train, seu_val, seu_test = split_dataset_train_val_test_stratified(
                seu_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 3
            )
        else:
            seu_train = seu_val = seu_test = None
        if hit_dataset is not None:
            hit_train, hit_val, hit_test = split_dataset_train_val_test_stratified(
                hit_dataset, train_ratio=0.6, val_ratio=0.2, seed=cfg.RANDOM_SEED + 4
            )
        else:
            hit_train = hit_val = hit_test = None
    # 域内跨工况隔离：仅用同一域训练时，train/val 只用该域数据（避免跨域干扰）
    use_isolated = bool(getattr(cfg, "TASK1_TRAIN_PER_DOMAIN_ISOLATED", False)) and enable_intra_condition
    isolated_domain = getattr(cfg, "TASK1_ISOLATED_DOMAIN", None)
    if use_isolated and isolated_domain == "CWRU":
        train_dataset = cwru_train
        val_dataset = cwru_val
        print("[Task1 Isolated] Training and validation use CWRU only; evaluation will run Task1-CWRU only.")
    elif use_isolated and isolated_domain == "PU":
        train_dataset = pu_train
        val_dataset = pu_val
        print("[Task1 Isolated] Training and validation use PU only; evaluation will run Task1-PU only.")
    else:
        train_dataset = torch.utils.data.ConcatDataset([cwru_train, pu_train])
        val_dataset = torch.utils.data.ConcatDataset([cwru_val, pu_val])

    run_all_phases = bool(run_all_phases_requested) and enable_intra_condition

    if run_all_phases:
        # ===== 一次脚本跑完多阶段：Task1-CWRU → Task1-PU → Task1-HUST → Task1-SEU → (optional) Task2 =====
        eval_only_mode = bool(getattr(cfg, "EVAL_ONLY_MODE", False))
        eval_only_phases_cfg = getattr(cfg, "EVAL_ONLY_PHASES", ["Task1-CWRU", "Task1-PU", "Task1-HUST", "Task1-SEU", "Task1-HIT"])
        if isinstance(eval_only_phases_cfg, (list, tuple, set)):
            eval_only_phases = {str(x) for x in eval_only_phases_cfg}
        elif isinstance(eval_only_phases_cfg, str):
            eval_only_phases = {eval_only_phases_cfg}
        else:
            eval_only_phases = {"Task1-CWRU", "Task1-PU", "Task1-HUST", "Task1-SEU", "Task1-HIT"}
        eval_only_ckpt_map = getattr(cfg, "EVAL_ONLY_CHECKPOINTS", {})
        cwru_es_metric = getattr(cfg, "EARLY_STOP_METRIC_TASK1_CWRU", None)
        pu_es_metric = getattr(cfg, "EARLY_STOP_METRIC_TASK1_PU", None)
        _skip_task2 = bool(getattr(cfg, "SKIP_TASK2", True))
        _loo_enable = bool(getattr(cfg, "LOO_ENABLE", False))
        _stmo_enable = bool(getattr(cfg, "STMO_ENABLE", False))
        _phase_eval_overrides = {}
        phases = [
            ("Task1-CWRU", "CWRU", cwru_train, cwru_val, "task1_cwru"),
            ("Task1-PU", "PU", pu_train, pu_val, "task1_pu"),
            ("Task1-HUST", "HUST", hust_train, hust_val, "task1_hust"),
        ]
        if seu_train is not None:
            phases.append(("Task1-SEU", "SEU", seu_train, seu_val, "task1_seu"))
        if hit_train is not None:
            phases.append(("Task1-HIT", "HIT", hit_train, hit_val, "task1_hit"))
        if not _skip_task2:
            phases.append(
                ("Task2", None, torch.utils.data.ConcatDataset([cwru_train, pu_train]), torch.utils.data.ConcatDataset([cwru_val, pu_val]), "task2")
            )
        _loo_phase_meta = {}
        _stmo_phase_meta = {}
        if _loo_enable:
            print(f"[LOO] enabled for domains={getattr(cfg, 'LOO_DOMAINS', ['CWRU','PU','HUST','SEU'])}")
            if bool(getattr(cfg, "TARGET_NSHOT_ENABLE", False)) and int(getattr(cfg, "TARGET_NSHOT_K", 0)) > 0:
                print(
                    f"[N-shot] TARGET_NSHOT_K={int(getattr(cfg, 'TARGET_NSHOT_K', 0))} "
                    f"repeat={int(getattr(cfg, 'TARGET_NSHOT_REPEAT', 1))} | "
                    f"CWRU_skip_sup_ft={bool(getattr(cfg, 'TASK1_CWRU_SKIP_SUPERVISED_FINETUNE', True))} | "
                    f"skip_lpft_fewshot={bool(getattr(cfg, 'TARGET_SUP_SKIP_LPFT_FEWSHOT_ENABLE', True))}"
                )
            _loo_strict = bool(getattr(cfg, "LOO_STRICT_MODE", True))
            _loo_nested_enable = bool(getattr(cfg, "LOO_NESTED_ENABLE", False))
            _loo_nested_policy = str(getattr(cfg, "LOO_NESTED_INNER_POLICY", "rotate")).strip().lower()
            if _loo_nested_policy not in {"rotate", "first"}:
                _loo_nested_policy = "rotate"
            _loo_nested_phase_allow = set(getattr(cfg, "LOO_NESTED_APPLY_PHASES", []) or [])
            _loo_nested_domain_allow = set([str(x).upper() for x in (getattr(cfg, "LOO_NESTED_APPLY_DOMAINS", []) or [])])
            _loo_nested_test_group_allow = set([str(x) for x in (getattr(cfg, "LOO_NESTED_APPLY_TEST_GROUPS", []) or [])])
            _loo_nested_select_only = os.environ.get("LOO_NESTED_SELECT_ONLY", "0") == "1"
            _loo_nested_inner_group = os.environ.get("LOO_NESTED_INNER_GROUP", "").strip()
            _domain_cfg = {
                "CWRU": (cwru_dataset, cfg.RANDOM_SEED, "task1_cwru"),
                "PU": (pu_dataset, cfg.RANDOM_SEED + 1, "task1_pu"),
                "HUST": (hust_dataset, cfg.RANDOM_SEED + 2, "task1_hust"),
                "SEU": (seu_dataset, cfg.RANDOM_SEED + 3, "task1_seu"),
                "HIT": (hit_dataset, cfg.RANDOM_SEED + 4, "task1_hit"),
            }
            _requested_domains = [str(x).upper() for x in getattr(cfg, "LOO_DOMAINS", ["CWRU", "PU", "HUST", "SEU", "HIT"])]
            _loo_phases = []
            _invalid_loo_domains = []
            for _dom in _requested_domains:
                if _dom not in _domain_cfg:
                    _invalid_loo_domains.append((_dom, "unknown domain"))
                    continue
                _ds, _seed, _tag = _domain_cfg[_dom]
                if _ds is None or not hasattr(_ds, "condition_ids"):
                    _invalid_loo_domains.append((_dom, "missing dataset/condition_ids"))
                    continue
                _uniq_cond_ids = sorted(set(int(x) for x in getattr(_ds, "condition_ids", [])))
                if len(_uniq_cond_ids) < 2:
                    _invalid_loo_domains.append((_dom, f"n_condition_groups={len(_uniq_cond_ids)}"))
                    continue
                _cid_to_key = getattr(_ds, "condition_id_to_key", {})
                for _outer_fold_idx, _test_gid in enumerate(_uniq_cond_ids):
                    _train_gids = [g for g in _uniq_cond_ids if g != _test_gid]
                    if len(_train_gids) == 0:
                        continue
                    _tr, _vl, _te, _meta = split_by_condition_groups(
                        _ds,
                        condition_ids=getattr(_ds, "condition_ids", None),
                        train_groups=_train_gids,
                        test_groups=[_test_gid],
                        val_ratio_from_train=intra_val_ratio,
                        seed=int(_seed),
                        train_subsample_ratio=_train_subsample_ratio_for(_dom),
                        subsample_label=f"{_dom}-LOO",
                    )
                    _tkey = str(_cid_to_key.get(_test_gid, _test_gid))
                    _pname = f"Task1-{_dom}-LOO-{_tkey}"
                    _ptag = f"{_tag}_loo_{_test_gid}"
                    _train_keys = [str(_cid_to_key.get(g, g)) for g in _train_gids]
                    _nested_hit = False
                    _nested_inner_key = ""
                    _nested_train_keys = []
                    if _loo_nested_enable and len(_train_gids) >= 2:
                        _pass_phase = (len(_loo_nested_phase_allow) == 0) or (_pname in _loo_nested_phase_allow)
                        _pass_domain = (len(_loo_nested_domain_allow) == 0) or (_dom in _loo_nested_domain_allow)
                        _pass_test_group = (len(_loo_nested_test_group_allow) == 0) or (_tkey in _loo_nested_test_group_allow)
                        _nested_hit = bool(_pass_phase and _pass_domain and _pass_test_group)
                    if _nested_hit:
                        if _loo_nested_inner_group:
                            _inner_candidates = [
                                int(g) for g in _train_gids
                                if str(g) == _loo_nested_inner_group
                                or str(_cid_to_key.get(g, g)) == _loo_nested_inner_group
                            ]
                            if len(_inner_candidates) != 1:
                                raise RuntimeError(
                                    f"[LOO-Nested] requested inner group={_loo_nested_inner_group!r} "
                                    f"not uniquely present in outer-train groups={_train_gids}"
                                )
                            _inner_val_gid = _inner_candidates[0]
                        elif _loo_nested_policy == "first":
                            _inner_val_gid = int(_train_gids[0])
                        else:
                            _inner_val_gid = int(_train_gids[_outer_fold_idx % len(_train_gids)])
                        _inner_train_gids = [int(g) for g in _train_gids if int(g) != int(_inner_val_gid)]
                        if len(_inner_train_gids) == 0:
                            _nested_hit = False
                        else:
                            _inner_tr, _inner_vl, _inner_te, _inner_meta = split_by_condition_groups(
                                _ds,
                                condition_ids=getattr(_ds, "condition_ids", None),
                                train_groups=_inner_train_gids,
                                test_groups=[_inner_val_gid],
                                val_ratio_from_train=intra_val_ratio,
                                seed=int(_seed),
                                train_subsample_ratio=_train_subsample_ratio_for(_dom),
                                subsample_label=f"{_dom}-LOO-nested",
                            )
                            if len(_inner_vl) > 0:
                                _tr = torch.utils.data.ConcatDataset([_inner_tr, _inner_vl])
                            else:
                                _tr = _inner_tr
                            _vl = _inner_te
                            _nested_inner_key = str(_cid_to_key.get(_inner_val_gid, _inner_val_gid))
                            _nested_train_keys = [str(_cid_to_key.get(g, g)) for g in _inner_train_gids]
                            print(
                                f"[LOO-Nested] {_pname}: outer_test={_tkey}, "
                                f"inner_val={_nested_inner_key}, inner_train={_nested_train_keys}, "
                                f"policy={_loo_nested_policy}"
                            )
                    _loo_phases.append((_pname, _dom, _tr, _vl, _ptag))
                    _loo_phase_meta[_pname] = {
                        "domain": _dom,
                        "train_groups": (_nested_train_keys if (_loo_nested_select_only and _nested_hit) else _train_keys),
                        "test_group": (_nested_inner_key if (_loo_nested_select_only and _nested_hit) else _tkey),
                        "outer_test_group": _tkey,
                        "nested_inner_enabled": bool(_nested_hit),
                        "nested_inner_val_group": _nested_inner_key,
                        "nested_inner_train_groups": _nested_train_keys,
                    }
                    if _dom == "CWRU":
                        _phase_eval_overrides[_pname] = {"cwru_test": _te}
                    elif _dom == "PU":
                        _phase_eval_overrides[_pname] = {"pu_test": _te}
                    elif _dom == "HUST":
                        _eval_hust = _vl if (_loo_nested_select_only and _nested_hit) else _te
                        _phase_eval_overrides[_pname] = {"hust_test": _eval_hust}
                    elif _dom == "SEU":
                        _phase_eval_overrides[_pname] = {"seu_test": _te}
                    elif _dom == "HIT":
                        _phase_eval_overrides[_pname] = {"hit_test": _te}
            if _invalid_loo_domains:
                _msg = ", ".join([f"{d}({why})" for d, why in _invalid_loo_domains])
                if _loo_strict:
                    raise RuntimeError(f"[LOO] Invalid requested domains under strict mode: {_msg}")
                print(f"[LOO] Warning: skip invalid domains: {_msg}")
            if len(_loo_phases) > 0:
                phases = _loo_phases
                if not _skip_task2:
                    phases.append(
                        ("Task2", None, torch.utils.data.ConcatDataset([cwru_train, pu_train]), torch.utils.data.ConcatDataset([cwru_val, pu_val]), "task2")
                    )
            elif _loo_enable and _loo_strict:
                raise RuntimeError("[LOO] No valid LOO phases were generated. Abort under LOO_STRICT_MODE=1.")

        if _stmo_enable:
            print(f"[STMO] enabled for domains={getattr(cfg, 'STMO_DOMAINS', ['CWRU','PU','HUST','SEU'])}")
            _stmo_strict = bool(getattr(cfg, "STMO_STRICT_MODE", True))
            _domain_cfg = {
                "CWRU": (cwru_dataset, cfg.RANDOM_SEED, "task1_cwru"),
                "PU": (pu_dataset, cfg.RANDOM_SEED + 1, "task1_pu"),
                "HUST": (hust_dataset, cfg.RANDOM_SEED + 2, "task1_hust"),
                "SEU": (seu_dataset, cfg.RANDOM_SEED + 3, "task1_seu"),
                "HIT": (hit_dataset, cfg.RANDOM_SEED + 4, "task1_hit"),
            }
            _requested_domains = [str(x).upper() for x in getattr(cfg, "STMO_DOMAINS", ["CWRU", "PU", "HUST", "SEU", "HIT"])]
            _stmo_phases = []
            _invalid_stmo_domains = []
            for _dom in _requested_domains:
                if _dom not in _domain_cfg:
                    _invalid_stmo_domains.append((_dom, "unknown domain"))
                    continue
                _ds, _seed, _tag = _domain_cfg[_dom]
                if _ds is None or not hasattr(_ds, "condition_ids"):
                    _hint = ""
                    if _dom == "HIT":
                        _hint = f" (HIT_PATH={getattr(cfg, 'HIT_PATH', None)}, ENABLE_HIT_DATASET={getattr(cfg, 'ENABLE_HIT_DATASET', True)})"
                    _invalid_stmo_domains.append((_dom, f"missing dataset/condition_ids{_hint}"))
                    continue
                _uniq_cond_ids = sorted(set(int(x) for x in getattr(_ds, "condition_ids", [])))
                if len(_uniq_cond_ids) < 2:
                    _invalid_stmo_domains.append((_dom, f"n_condition_groups={len(_uniq_cond_ids)}"))
                    continue
                _cid_to_key = getattr(_ds, "condition_id_to_key", {})
                for _train_gid in _uniq_cond_ids:
                    _test_gids = [g for g in _uniq_cond_ids if g != _train_gid]
                    if len(_test_gids) == 0:
                        continue
                    _tr, _vl, _te, _meta = split_by_condition_groups(
                        _ds,
                        condition_ids=getattr(_ds, "condition_ids", None),
                        train_groups=[_train_gid],
                        test_groups=_test_gids,
                        val_ratio_from_train=intra_val_ratio,
                        seed=int(_seed),
                        train_subsample_ratio=_train_subsample_ratio_for(_dom),
                        subsample_label=f"{_dom}-STMO",
                    )
                    _train_key = str(_cid_to_key.get(_train_gid, _train_gid))
                    _test_keys = [str(_cid_to_key.get(g, g)) for g in _test_gids]
                    _pname = f"Task1-{_dom}-STMO-train_{_train_key}"
                    _ptag = f"{_tag}_stmo_train_{_train_gid}"
                    _stmo_phases.append((_pname, _dom, _tr, _vl, _ptag))
                    _stmo_phase_meta[_pname] = {
                        "domain": _dom,
                        "train_group": _train_key,
                        "test_groups": _test_keys,
                        "test_gids": _test_gids,
                        "cid_to_key": _cid_to_key,
                    }
                    if _dom == "CWRU":
                        _phase_eval_overrides[_pname] = {"cwru_test": _te}
                    elif _dom == "PU":
                        _phase_eval_overrides[_pname] = {"pu_test": _te}
                    elif _dom == "HUST":
                        _phase_eval_overrides[_pname] = {"hust_test": _te}
                    elif _dom == "SEU":
                        _phase_eval_overrides[_pname] = {"seu_test": _te}
                    elif _dom == "HIT":
                        _phase_eval_overrides[_pname] = {"hit_test": _te}
            if _invalid_stmo_domains:
                _msg = ", ".join([f"{d}({why})" for d, why in _invalid_stmo_domains])
                if _stmo_strict:
                    raise RuntimeError(f"[STMO] Invalid requested domains under strict mode: {_msg}")
                print(f"[STMO] Warning: skip invalid domains: {_msg}")
            if len(_stmo_phases) > 0:
                phases = _stmo_phases
                if not _skip_task2:
                    phases.append(
                        ("Task2", None, torch.utils.data.ConcatDataset([cwru_train, pu_train]), torch.utils.data.ConcatDataset([cwru_val, pu_val]), "task2")
                    )
            elif _stmo_enable and _stmo_strict:
                raise RuntimeError("[STMO] No valid STMO phases were generated. Abort under STMO_STRICT_MODE=1.")

        _protocol_train_only = list(getattr(cfg, "LOO_TRAIN_ONLY_PHASES", None) or [])
        if _stmo_enable:
            _protocol_train_only = list(getattr(cfg, "STMO_TRAIN_ONLY_PHASES", None) or []) or _protocol_train_only
        if (_loo_enable or _stmo_enable) and len(_protocol_train_only) > 0:
            _train_only_set = set(_protocol_train_only)
            _n_before = len(phases)
            phases = [p for p in phases if p[0] in _train_only_set]
            _skipped = _n_before - len(phases)
            _proto_label = "STMO" if _stmo_enable else "LOO"
            print(
                f"[{_proto_label}] train-only filter: {_n_before} -> {len(phases)} phases "
                f"(fully skipped {_skipped}, no train/eval)"
            )
            if len(phases) == 0:
                raise RuntimeError(
                    f"[{_proto_label}] train-only filter matched no phases. "
                    "Check phase names against logs (e.g. Task1-CWRU-STMO-train_speed_1750)."
                )

        phase_ckpt_paths = []
        phase_eval_results = {"Task1": {}, "Task2": {}}
        loo_records = {"CWRU": [], "PU": [], "HUST": [], "SEU": [], "HIT": []}
        loo_detail_rows = {"CWRU": [], "PU": [], "HUST": [], "SEU": [], "HIT": []}
        stmo_records = {"CWRU": [], "PU": [], "HUST": [], "SEU": [], "HIT": []}
        stmo_detail_rows = {"CWRU": [], "PU": [], "HUST": [], "SEU": [], "HIT": []}
        _loo_metric_source = str(getattr(cfg, "LOO_SUMMARY_METRIC_SOURCE", "final_decision")).strip().lower()
        _stmo_metric_source = str(getattr(cfg, "STMO_SUMMARY_METRIC_SOURCE", _loo_metric_source)).strip().lower()

        def _pick_protocol_metrics(_zs, metric_source):
            if not isinstance(_zs, dict):
                return float("nan"), float("nan"), float("nan")
            if metric_source == "fusion":
                _acc = float(_zs.get("diag_acc", float("nan")))
                _rec = float(_zs.get("macro_recall", float("nan")))
                _f1 = float(_zs.get("macro_f1", float("nan")))
                return _acc, _rec, _f1
            if metric_source in ("best_path", "final_decision", "top1"):
                _acc = float(_zs.get("final_decision_diag_acc", float("nan")))
                _rec = float(_zs.get("final_decision_macro_recall", float("nan")))
                _f1 = float(_zs.get("final_decision_macro_f1", float("nan")))
                _acc_vote = float(_zs.get("final_decision_voting_acc", float("nan")))
                _rec_vote = float(_zs.get("final_decision_voting_macro_recall", float("nan")))
                _f1_vote = float(_zs.get("final_decision_voting_macro_f1", float("nan")))
                if _acc_vote == _acc_vote and _acc_vote >= 0.0:
                    _acc, _rec, _f1 = _acc_vote, _rec_vote, _f1_vote
                return _acc, _rec, _f1
            _acc = float(_zs.get("spec_head_diag_acc", float("nan")))
            _rec = float(_zs.get("spec_head_macro_recall", float("nan")))
            _f1 = float(_zs.get("spec_head_macro_f1", float("nan")))
            return _acc, _rec, _f1

        def _pick_loo_metrics(_zs):
            return _pick_protocol_metrics(_zs, _loo_metric_source)

        def _pick_stmo_metrics(_zs):
            return _pick_protocol_metrics(_zs, _stmo_metric_source)

        def _make_condition_test_loader(test_subset):
            return torch.utils.data.DataLoader(
                test_subset,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                collate_fn=multimodal_collate_fn,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False,
            )

        def _run_iso_domain_eval(_detector, _iso_domain, _test_loader, _emb_export_tag=""):
            if _iso_domain == "CWRU":
                return run_evaluation(
                    _detector, _test_loader, None, None, hit_loader=None,
                    task1_cwru_loader=_test_loader, task1_pu_loader=None,
                    emb_export_tag=_emb_export_tag,
                )
            if _iso_domain == "PU":
                return run_evaluation(
                    _detector, None, _test_loader, None, hit_loader=None,
                    task1_cwru_loader=None, task1_pu_loader=_test_loader,
                    emb_export_tag=_emb_export_tag,
                )
            if _iso_domain == "HUST":
                return run_evaluation(
                    _detector, None, None, _test_loader, hit_loader=None,
                    task1_cwru_loader=None, task1_pu_loader=None,
                    emb_export_tag=_emb_export_tag,
                )
            if _iso_domain == "SEU":
                return run_evaluation(
                    _detector, None, None, None, hit_loader=_test_loader,
                    task1_cwru_loader=None, task1_pu_loader=None,
                    emb_export_tag=_emb_export_tag,
                )
            if _iso_domain == "HIT":
                return run_evaluation(
                    _detector, None, None, None, hit_loader=_test_loader,
                    task1_cwru_loader=None, task1_pu_loader=None,
                    emb_export_tag=_emb_export_tag,
                )
            return None

        def _extract_domain_zero_shot(_eval_ret, _iso_domain):
            if not isinstance(_eval_ret, dict):
                return {}
            _task1_res = _eval_ret.get("task1_results", {}) or {}
            _task2_res = _eval_ret.get("task2_results", {}) or {}
            if _iso_domain in ("CWRU", "PU"):
                _dres = _task1_res.get(_iso_domain, {})
            elif _iso_domain in ("HUST", "SEU", "HIT"):
                _dres = _task2_res.get(_iso_domain, {})
            else:
                _dres = {}
            if isinstance(_dres, dict):
                _zs = _dres.get("zero_shot", {})
                return _zs if isinstance(_zs, dict) else {}
            return {}
        _n_phases = max(1, len(phases))
        for phase_idx, (phase_name, iso_domain, tr_ds, vl_ds, ckpt_tag) in enumerate(phases):
            # ATT167: 统一各阶段种子为 RANDOM_SEED（回退 ATT166 的 +phase_idx 偏移）。
            # ATT166 post-analysis：使用 42/43/44 导致 PU(seed=43)/Task2(seed=44) 初始化与历史基线不同，
            # 引入额外 ~5-7% 性能差异（PU Ensemble 0.638→0.562），无法判断 depth=1 的真实容量损失。
            # 统一使用 RANDOM_SEED 后，各阶段初始化与 BN 基线（单次 set_seed(42)）最大程度对齐，
            # 可以精确测量 depth=1 vs depth=2 的性能差异，为后续架构决策提供可靠参照。
            _phase_seed = cfg.RANDOM_SEED
            set_seed(_phase_seed)
            print("\n" + "=" * 80)
            _phase_train_desc = "CWRU+PU" if iso_domain is None else str(iso_domain)
            print(f"[Phase {phase_idx + 1}/{_n_phases}] {phase_name}（仅用{_phase_train_desc}训练）")
            print(f"[Phase Seed] seed={_phase_seed} (unified=True, ATT167)")
            print("=" * 80)
            _train_dataset = tr_ds
            _val_dataset = vl_ds
            _sampler, _precomputed_class_weights = compute_class_weights_and_sampler(_train_dataset, cfg, device)
            # ATT379(S0): per-phase batch size to avoid PU OOM.
            _orig_batch_size = cfg.BATCH_SIZE
            if phase_name == "Task1-PU":
                cfg.BATCH_SIZE = int(getattr(cfg, "BATCH_SIZE_TASK1_PU", cfg.BATCH_SIZE))
                print(f"[ATT379] PU batch_size override: {_orig_batch_size} -> {cfg.BATCH_SIZE}")
            elif phase_name == "Task2":
                cfg.BATCH_SIZE = int(getattr(cfg, "BATCH_SIZE_TASK2", cfg.BATCH_SIZE))
                print(f"[ATT379] Task2 batch_size override: {_orig_batch_size} -> {cfg.BATCH_SIZE}")

            # ATT397: Log episodic training status (in-batch splitting, no custom sampler needed)
            _use_episodic = bool(getattr(cfg, "USE_EPISODIC_TRAINING", False))
            if _use_episodic:
                _ep_split_key = str(getattr(cfg, "EPISODIC_SPLIT_KEY", "auto"))
                if _ep_split_key == "auto":
                    _ep_split_key = "domain" if phase_name == "Task2" else "condition"
                print(f"[ATT397] Episodic training ENABLED: phase={phase_name}, "
                      f"split_key={_ep_split_key} (in-batch condition splitting)")

            _ovr = _phase_eval_overrides.get(phase_name, {})
            _cwru_test_ref = _ovr.get("cwru_test", cwru_test if phase_name == "Task1-CWRU" else None)
            _pu_test_ref = _ovr.get("pu_test", pu_test if phase_name == "Task1-PU" else None)
            _hust_eval_ref = _ovr.get("hust_test", hust_test if phase_name == "Task1-HUST" else (hust_dataset if phase_name == "Task2" else None))
            _seu_eval_ref = _ovr.get("seu_test", seu_test if phase_name == "Task1-SEU" else (seu_dataset if phase_name == "Task2" else None))
            _hit_eval_ref = _ovr.get("hit_test", hit_test if phase_name == "Task1-HIT" else None)
            _target_loader_ds = _hit_eval_ref if _hit_eval_ref is not None else _seu_eval_ref
            _train_loader, _cwru_test_loader, _pu_test_loader, _hust_loader, _hit_loader = setup_dataloaders(
                cfg, _train_dataset, _cwru_test_ref, _pu_test_ref, _hust_eval_ref, _sampler, hit_dataset=_target_loader_ds
            )
            _nested_select_phase = bool(
                os.environ.get("LOO_NESTED_SELECT_ONLY", "0") == "1"
                and _loo_phase_meta.get(phase_name, {}).get("nested_inner_enabled", False)
            )
            if _nested_select_phase:
                # Keep the inner held-out condition away from all training-time target monitors.
                _hust_loader = None
                print("[LOO-Nested-Select] target monitor disabled; scoring only the inner validation loader after training.")
            _val_loader = torch.utils.data.DataLoader(
                _val_dataset,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                collate_fn=multimodal_collate_fn,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False,
                generator=torch.Generator().manual_seed(int(cfg.RANDOM_SEED) + 1000 + int(phase_idx)),  # isolate validation worker seeding from training RNG
            )
            signal_encoder, spec_encoder, text_encoder, clip_model, fusion_module, fusion_decoder, detector, wst_encoder, env_encoder, scd_encoder = create_models(cfg, device)
            # ATT463: freeze fusion module + decoder when split_arch is enabled
            if bool(getattr(cfg, "SPLIT_ARCH_ENABLE", False)):
                _frozen_count = 0
                for _p in fusion_module.parameters():
                    _p.requires_grad = False
                    _frozen_count += 1
                for _p in fusion_decoder.parameters():
                    _p.requires_grad = False
                    _frozen_count += 1
                print(f"[ATT463-SplitArch] Frozen {_frozen_count} params in fusion_module + fusion_decoder")
            diag_lr_mult = float(getattr(cfg, "DIAG_LINEAR_LR_MULT", 0.10))
            diag_params = []
            if hasattr(fusion_decoder, "diag_linear_head"):
                diag_params += list(fusion_decoder.diag_linear_head.parameters())
            if hasattr(fusion_decoder, "diag_linear_scale"):
                diag_params += [fusion_decoder.diag_linear_scale]
            diag_param_ids = {id(p) for p in diag_params}
            _agb = getattr(detector, "ablation_gate_bank", None)
            _gate_param_ids = {id(p) for p in _agb.parameters()} if _agb is not None else set()

            # ATT432: modality-specific learning rate multipliers
            _mslr_sig = float(getattr(cfg, "MSLR_SIGNAL_MULT", 1.0))
            _mslr_spe = float(getattr(cfg, "MSLR_SPEC_MULT", 1.0))
            _mslr_fus = float(getattr(cfg, "MSLR_FUSION_MULT", 3.0))
            _mslr_dec = float(getattr(cfg, "MSLR_DECODER_MULT", 1.0))
            signal_param_ids = {id(p) for p in signal_encoder.parameters()}
            spec_param_ids = {id(p) for p in spec_encoder.parameters()}
            fusion_param_ids = {id(p) for p in fusion_module.parameters()}
            signal_params, spec_params, fusion_params, decoder_params = [], [], [], []
            for p in (list(signal_encoder.parameters()) + list(spec_encoder.parameters()) +
                      list(fusion_module.parameters()) + list(fusion_decoder.parameters())):
                if id(p) in diag_param_ids or id(p) in _gate_param_ids:
                    continue
                if not p.requires_grad:
                    continue
                if id(p) in signal_param_ids:
                    signal_params.append(p)
                elif id(p) in spec_param_ids:
                    spec_params.append(p)
                elif id(p) in fusion_param_ids:
                    fusion_params.append(p)
                else:
                    decoder_params.append(p)
            _base_lr = cfg.LEARNING_RATE
            # ATT446/ATT449: add wst_encoder + env_encoder params to spec group
            _wst_params = list(wst_encoder.parameters()) if wst_encoder is not None else []
            _env_params = list(env_encoder.parameters()) if env_encoder is not None else []
            _scd_params = list(scd_encoder.parameters()) if scd_encoder is not None else []
            if _gate_param_ids:
                _wst_params = [p for p in _wst_params if id(p) not in _gate_param_ids]
                _env_params = [p for p in _env_params if id(p) not in _gate_param_ids]
                _scd_params = [p for p in _scd_params if id(p) not in _gate_param_ids]
            _param_groups = [
                {"params": unique_parameters(signal_params), "lr": _base_lr * _mslr_sig, "name": "signal"},
                {
                    "params": unique_parameters(spec_params + _wst_params + _env_params + _scd_params),
                    "lr": _base_lr * _mslr_spe,
                    "name": "spec",
                },
                {"params": unique_parameters(fusion_params), "lr": _base_lr * _mslr_fus, "name": "fusion"},
                {"params": unique_parameters(decoder_params), "lr": _base_lr * _mslr_dec, "name": "decoder"},
                {"params": unique_parameters(diag_params), "lr": _base_lr * diag_lr_mult, "name": "diag"},
            ]
            if _agb is not None:
                _gate_lr = _base_lr * float(getattr(cfg, "ABLATION_GATE_LR_MULT", 0.5))
                _assigned = assigned_parameter_ids(_param_groups)
                _gate_params = [p for p in _agb.parameters() if id(p) not in _assigned]
                if _gate_params:
                    _param_groups.append(
                        {"params": _gate_params, "lr": _gate_lr, "name": "ablation_gates"}
                    )
            _param_groups = sanitize_optimizer_param_groups(_param_groups)
            print(f"[ATT432-MSLR] LR groups: signal={_base_lr*_mslr_sig:.2e}, "
                  f"spec={_base_lr*_mslr_spe:.2e}, fusion={_base_lr*_mslr_fus:.2e}, "
                  f"decoder={_base_lr*_mslr_dec:.2e}, diag={_base_lr*diag_lr_mult:.2e}")
            _use_sam = bool(getattr(cfg, "USE_SAM_OPTIMIZER", False))
            if _use_sam:
                _sam_rho = float(getattr(cfg, "SAM_RHO", 0.05))
                _sam_adaptive = bool(getattr(cfg, "SAM_ADAPTIVE", False))
                _optimizer = SAM(_param_groups, torch.optim.Adam, rho=_sam_rho, adaptive=_sam_adaptive)
                print(f"[ATT397] SAM optimizer ENABLED: rho={_sam_rho}, adaptive={_sam_adaptive}")
            else:
                _optimizer = torch.optim.Adam(_param_groups)
            history = None
            _phase_eval_only = eval_only_mode and (phase_name in eval_only_phases)
            # ATT409: Log structural config for this experiment
            _att409_rm_ln = bool(getattr(cfg, "FUSION_PROJ_REMOVE_LAYERNORM", False))
            _att409_da_disable_t1 = bool(getattr(cfg, "FUSION_DOMAIN_ALIGN_DISABLE_ON_TASK1", True))
            _att409_vicreg = bool(getattr(cfg, "USE_GLOBAL_VICREG_LOSS", False))
            _att409_decorr = bool(getattr(cfg, "USE_GLOBAL_DECORR_LOSS", False))
            _att409_core_only = bool(getattr(cfg, "CORE_LOSS_ONLY_MODE", False))
            _att410_skip_pw = bool(getattr(cfg, "EARLY_STOP_SKIP_PREWINDOW_BASELINE", True))
            _att410_spec_ft = str(getattr(cfg, "SPEC_FEATURE_TYPE", "swt"))
            _att412_clip_w = float(getattr(cfg, "CLIP_STYLE_CONTRASTIVE_WEIGHT", 0.0))
            _att412_text_etf = bool(getattr(cfg, "USE_TEXT_ETF_ALIGNMENT_LOSS", False))
            _att412_text_etf_w = float(getattr(cfg, "TEXT_ETF_ALIGNMENT_WEIGHT", 0.0))
            _att413_etf_proto = bool(getattr(cfg, "USE_SIMPLEX_ETF_PROTOTYPES", False))
            _att416_sig_gate = bool(getattr(cfg, "FUSION_SIGNAL_ENTROPY_GATE", False))
            _att416_sig_gate_floor = float(getattr(cfg, "FUSION_SIGNAL_ENTROPY_GATE_FLOOR", 0.1))
            _att417_fishr = bool(getattr(cfg, "USE_FISHR_LOSS", False))
            _att417_es_cwru = getattr(cfg, "EARLY_STOP_METRIC_TASK1_CWRU", None)
            _att418_ms_enable = bool(getattr(cfg, "MULTISCALE_STFT_ENABLE", False))
            _att418_ms_nffts = getattr(cfg, "MULTISCALE_STFT_N_FFTS", ())
            _att420_spec_res = bool(getattr(cfg, "FUSION_SPEC_RESIDUAL_ENABLE", False))
            _att420_spec_res_alpha = float(getattr(cfg, "FUSION_SPEC_RESIDUAL_INIT_ALPHA", 0.5))
            print(f"[ATT420-Config] VICReg={_att409_vicreg}, DECORR={_att409_decorr}, "
                  f"CORE_LOSS_ONLY={_att409_core_only}, "
                  f"ETF_PROTO={_att413_etf_proto}, "
                  f"CLIP_STYLE_W={_att412_clip_w}, TEXT_ETF={_att412_text_etf}(w={_att412_text_etf_w}), "
                  f"SIG_ENTROPY_GATE={_att416_sig_gate}(floor={_att416_sig_gate_floor}), "
                  f"FISHR={_att417_fishr}, ES_CWRU={_att417_es_cwru}, "
                  f"skip_prewindow={_att410_skip_pw}, "
                  f"SPEC_FEATURE_TYPE={_att410_spec_ft}, "
                  f"MS_STFT={_att418_ms_enable}(scales={_att418_ms_nffts}), "
                  f"SPEC_RES={_att420_spec_res}(alpha_init={_att420_spec_res_alpha})")

            _orig_early_stop_metric = cfg.EARLY_STOP_METRIC
            _orig_early_stop_min_epochs = cfg.EARLY_STOP_MIN_EPOCHS
            _orig_skip_prewindow = getattr(cfg, 'EARLY_STOP_SKIP_PREWINDOW_BASELINE', False)
            _orig_spec_align_warmup_end = getattr(cfg, 'SPEC_ALIGN_WARMUP_END_EPOCH', 8)

            # ATT410: Reverted ATT409 P1. skip_prewindow=False caused E1 (untrained) to be
            # selected as best in CORE_LOSS_ONLY mode (val_loss anti-correlated with diag perf).
            # Restoring global default True: at min_epochs, discard pre-window best and track
            # from inf, allowing E10+ to compete. ATT376 used True and achieved 0.9624.
            # (ATT409 forced False → selected E1 → SpecHead=0.1935, catastrophic.)
            if (phase_name == "Task1-CWRU") and cwru_es_metric:
                cfg.EARLY_STOP_METRIC = cwru_es_metric
                print(f"[Early Stop Override] {phase_name} metric: {cfg.EARLY_STOP_METRIC}")
            if (phase_name == "Task1-PU") and pu_es_metric:
                cfg.EARLY_STOP_METRIC = pu_es_metric
                print(f"[Early Stop Override] {phase_name} metric: {cfg.EARLY_STOP_METRIC}")
            # ATT168: Task2 专属 min_epochs + skip_prewindow_baseline。
            # Task2 val_loss 在 E2 处出现噪声极小值（E1=0.268, E2=0.190, E3=0.475...），
            # pre-window 机制将 E2 设为 ES 基线（0.190），此后所有 epoch 均无法超越该基线，
            # 导致 ES 在 E14 触发并选中欠训练的 E2 checkpoint，HUST/HIT 性能严重退化。
            # skip_prewindow_baseline=True：在 min_epochs 处丢弃 pre-window 最优，以 inf 重新追踪，
            # 使 E10+ 的真实收敛 checkpoint（如 E12 val_loss≈0.39）能被正常选中。
            # ATT175: Task2 专属 SPEC_ALIGN_WARMUP_END_EPOCH 覆盖。
            # 方案A实证：CWRU/PU 使用 warmup_end=5 提升 SpecProto（+11pp），但 Task2 中
            # E5 全量激活 L_align 与 Fusion CE 梯度产生严重冲突（val_loss E10+ 振荡爆炸），
            # HUST/HIT FusionRaw/FusionProto 崩溃至 0.23/0.24。
            # Task2 使用 warmup_end=8 允许融合网络先收敛，再引入对齐压力，历史上无崩溃记录。
            if phase_name == "Task2":
                _task2_min_epochs = getattr(cfg, 'EARLY_STOP_MIN_EPOCHS_TASK2', None)
                if _task2_min_epochs is not None:
                    cfg.EARLY_STOP_MIN_EPOCHS = _task2_min_epochs
                _task2_es_metric = getattr(cfg, 'EARLY_STOP_METRIC_TASK2', None)
                if _task2_es_metric:
                    cfg.EARLY_STOP_METRIC = _task2_es_metric
                cfg.EARLY_STOP_SKIP_PREWINDOW_BASELINE = True
                _task2_warmup_end = getattr(cfg, 'SPEC_ALIGN_WARMUP_END_EPOCH_TASK2', 8)
                cfg.SPEC_ALIGN_WARMUP_END_EPOCH = _task2_warmup_end
                config_module.SPEC_ALIGN_WARMUP_END_EPOCH = _task2_warmup_end  # ATT176: train_alignment 通过 config 模块读取
                print(f"[Early Stop Override] Task2 metric={cfg.EARLY_STOP_METRIC}, "
                      f"min_epochs={cfg.EARLY_STOP_MIN_EPOCHS}, "
                      f"skip_prewindow_baseline=True (ATT168), "
                      f"spec_align_warmup_end={cfg.SPEC_ALIGN_WARMUP_END_EPOCH} (ATT175)")
            _phase_skip_map = {
                "Task1-CWRU": bool(getattr(cfg, "SKIP_TASK1_CWRU", False)),
                "Task1-PU": bool(getattr(cfg, "SKIP_TASK1_PU", False)),
                "Task2": bool(getattr(cfg, "SKIP_TASK2", False)),
            }
            _phase_resume_map = {
                "Task1-CWRU": str(getattr(cfg, "RESUME_TASK1_CWRU_CHECKPOINT", "") or ""),
                "Task1-PU": str(getattr(cfg, "RESUME_TASK1_PU_CHECKPOINT", "") or ""),
                "Task2": str(getattr(cfg, "RESUME_TASK2_CHECKPOINT", "") or ""),
            }
            _should_skip = _phase_skip_map.get(phase_name, False)
            _resume_ckpt_path = _phase_resume_map.get(phase_name, "")
            if _should_skip:
                print(f"\n[PHASE_SKIP] {phase_name}: SKIPPED by SKIP_{phase_name.replace('-','_').upper()}=1")
            if _resume_ckpt_path:
                print(f"[PHASE_RESUME] {phase_name}: will resume from {_resume_ckpt_path}")
            try:
                if _should_skip:
                    _skip_ckpt = None
                    if isinstance(eval_only_ckpt_map, dict):
                        _skip_ckpt = (
                            eval_only_ckpt_map.get(phase_name)
                            or eval_only_ckpt_map.get(ckpt_tag)
                            or eval_only_ckpt_map.get(iso_domain)
                        )
                    if not _skip_ckpt and _resume_ckpt_path:
                        _skip_ckpt = _resume_ckpt_path
                    if not _skip_ckpt:
                        _skip_ckpt = getattr(cfg, "CHECKPOINT_PATH", None)
                    if _skip_ckpt and os.path.exists(str(_skip_ckpt)):
                        _eval_ckpt = resolve_eval_checkpoint(str(_skip_ckpt))
                        print(f"[PHASE_SKIP] {phase_name}: loading checkpoint={_eval_ckpt} for evaluation")
                        load_model_checkpoint(
                            _eval_ckpt, signal_encoder, spec_encoder, text_encoder,
                            fusion_module, fusion_decoder, optimizer=None, device=device
                        )
                        history = {"best_checkpoint_path": _eval_ckpt, "best_metric_epoch": "skipped"}
                    else:
                        print(f"[PHASE_SKIP] {phase_name}: no checkpoint found, skipping entirely")
                        phase_ckpt_paths.append((phase_name, "(SKIPPED, no checkpoint)"))
                        continue
                elif _phase_eval_only:
                    _raw_ckpt = None
                    if isinstance(eval_only_ckpt_map, dict):
                        _raw_ckpt = (
                            eval_only_ckpt_map.get(phase_name)
                            or eval_only_ckpt_map.get(ckpt_tag)
                            or eval_only_ckpt_map.get(iso_domain)
                        )
                    if not _raw_ckpt:
                        _raw_ckpt = getattr(cfg, "CHECKPOINT_PATH", None)
                    if not _raw_ckpt:
                        raise ValueError(
                            f"[EVAL_ONLY_MODE] No checkpoint configured for phase={phase_name}. "
                            f"Set EVAL_ONLY_CHECKPOINTS['{phase_name}'] or CHECKPOINT_PATH."
                        )
                    _eval_ckpt = resolve_eval_checkpoint(str(_raw_ckpt))
                    if not os.path.exists(_eval_ckpt):
                        raise FileNotFoundError(
                            f"[EVAL_ONLY_MODE] Checkpoint not found for phase={phase_name}: {_eval_ckpt}"
                        )
                    print(f"[EVAL_ONLY_MODE] {phase_name}: skip training, load checkpoint={_eval_ckpt}")
                    load_model_checkpoint(
                        _eval_ckpt, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, optimizer=None, device=device
                    )
                    history = {"best_checkpoint_path": _eval_ckpt, "best_metric_epoch": "eval_only"}
                else:
                    _resume_start_epoch = 0
                    if _resume_ckpt_path and os.path.exists(_resume_ckpt_path):
                        print(f"[PHASE_RESUME] {phase_name}: loading resume checkpoint...")
                        _resume_start_epoch, _ = load_model_checkpoint(
                            _resume_ckpt_path, signal_encoder, spec_encoder, text_encoder,
                            fusion_module, fusion_decoder, optimizer=_optimizer, device=device
                        )
                        if _resume_start_epoch is None or _resume_start_epoch < 0:
                            _resume_start_epoch = 0
                        print(f"[PHASE_RESUME] {phase_name}: resuming from epoch {_resume_start_epoch}")
                    history = train_alignment(**build_train_alignment_kwargs(
                        cfg, _train_loader, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, _optimizer, device, clip_model,
                        _hust_loader, _resume_start_epoch, _precomputed_class_weights, ckpt_tag,
                        hit_loader=_hit_loader, val_loader=_val_loader, wst_encoder=wst_encoder, env_encoder=env_encoder, scd_encoder=scd_encoder,
                    ))
            finally:
                cfg.EARLY_STOP_METRIC = _orig_early_stop_metric
                cfg.EARLY_STOP_MIN_EPOCHS = _orig_early_stop_min_epochs
                cfg.EARLY_STOP_SKIP_PREWINDOW_BASELINE = _orig_skip_prewindow
                cfg.BATCH_SIZE = _orig_batch_size
                cfg.SPEC_ALIGN_WARMUP_END_EPOCH = _orig_spec_align_warmup_end
                config_module.SPEC_ALIGN_WARMUP_END_EPOCH = _orig_spec_align_warmup_end  # ATT176: 还原 config 模块级变量
            _checkpoint_path = None
            if isinstance(history, dict):
                best_ckpt = history.get("best_checkpoint_path", None)
                best_epoch = history.get("best_metric_epoch", None)
                if best_ckpt:
                    eval_ckpt = resolve_eval_checkpoint(str(best_ckpt))
                    _ckpt_dir = os.path.dirname(os.path.abspath(eval_ckpt)) if eval_ckpt else ""
                    _best_epoch_int = _coerce_history_best_epoch(best_epoch, eval_ckpt)
                    _readable = resolve_readable_checkpoint(
                        eval_ckpt,
                        checkpoint_dir=_ckpt_dir or None,
                        checkpoint_tag=ckpt_tag,
                        best_epoch=_best_epoch_int,
                    )
                    if _readable:
                        eval_ckpt = _readable
                    elif eval_ckpt and os.path.exists(eval_ckpt):
                        raise RuntimeError(
                            f"[Checkpoint] Unreadable best checkpoint for {phase_name}: {eval_ckpt}. "
                            f"Remove the truncated file and re-run this fold, or copy a valid "
                            f"*_pre_best.pth to {eval_ckpt}."
                        )
                    load_model_checkpoint(eval_ckpt, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, optimizer=None, device=device)
                    _checkpoint_path = eval_ckpt
                    load_checkpoint_prototypes(_checkpoint_path, detector, cfg, device)
                    if os.environ.get("FROZEN_PROTO_DIAG_ENABLE", "0") == "1":
                        from utils.frozen_proto_probe import run_frozen_proto_probe
                        run_frozen_proto_probe(
                            detector=detector,
                            train_dataset=_train_dataset,
                            val_loader=_val_loader,
                            target_loader=_hust_loader,
                            checkpoint_path=_checkpoint_path,
                            output_path=os.environ.get("FROZEN_PROTO_DIAG_OUTPUT", ""),
                        )
                        if os.environ.get("FROZEN_PROTO_DIAG_ONLY", "0") == "1":
                            break
                    if hasattr(fusion_decoder, "set_diag_linear_ramp"):
                        fusion_decoder.set_diag_linear_ramp(1.0)
                    if iso_domain is None and best_ckpt and _hust_loader is not None:
                        best_text_prototypes = detector._active_text_prototypes()
                        if best_text_prototypes is not None:
                            print("\n" + "=" * 80)
                            _ep_disp = _best_epoch_int if _best_epoch_int is not None else best_epoch
                            print(f"[Phase 3] Zero-shot Evaluation Best Checkpoint Epoch {_ep_disp}")
                            print("=" * 80 + "\n")
                            best_metrics = zero_shot_eval_hust(
                                signal_encoder=signal_encoder, spec_encoder=spec_encoder,
                                fusion_module=fusion_module, fusion_decoder=fusion_decoder,
                                text_prototypes=best_text_prototypes, hust_loader=_hust_loader,
                                device=device, enable_finetuning=False, finetune_samples=0,
                                finetune_lr=cfg.ZERO_SHOT_FINETUNING_LR, finetune_steps=cfg.ZERO_SHOT_FINETUNING_STEPS
                            )
                            print(f"[Best Epoch Zero-shot] AUC_nf={best_metrics.get('auc_nf', float('nan'))}")
                            print(f"[Best Epoch Zero-shot] Fusion Decoder acc={best_metrics.get('fusion_decoder_acc', 0.0):.4f}")
            if _checkpoint_path is None and isinstance(history, dict):
                _checkpoint_path = history.get("best_checkpoint_path", None)
            phase_ckpt_paths.append((phase_name, _checkpoint_path or f"(see checkpoints/epoch_*_{ckpt_tag}.pth)"))
            _emb_export_tag = ""
            if _loo_enable:
                _emb_meta = _loo_phase_meta.get(phase_name, {})
                _emb_export_tag = str(_emb_meta.get("test_group", "") or "").strip()

            if _stmo_enable and iso_domain in ("CWRU", "PU", "HUST", "SEU", "HIT"):
                _stmo_meta = _stmo_phase_meta.get(phase_name, {})
                _test_gids = _stmo_meta.get("test_gids", [])
                _train_key = str(_stmo_meta.get("train_group", "unknown"))
                _cid_map = _stmo_meta.get("cid_to_key", {})
                if iso_domain == "CWRU":
                    _merged_te = _ovr.get("cwru_test", _cwru_test_ref)
                elif iso_domain == "PU":
                    _merged_te = _ovr.get("pu_test", _pu_test_ref)
                elif iso_domain == "HUST":
                    _merged_te = _ovr.get("hust_test", _hust_eval_ref)
                elif iso_domain == "SEU":
                    _merged_te = _ovr.get("seu_test", _seu_eval_ref)
                else:
                    _merged_te = _ovr.get("hit_test", _hit_eval_ref)
                if _merged_te is None:
                    print(f"[STMO] Warning: missing merged test set for phase={phase_name}")
                else:
                    for _test_gid in _test_gids:
                        _test_key = str(_cid_map.get(_test_gid, _test_gid))
                        _te_cond = subset_by_condition(_merged_te, _test_gid)
                        if len(_te_cond) == 0:
                            print(f"[STMO] skip empty test subset: train={_train_key} test={_test_key}")
                            continue
                        _te_loader = _make_condition_test_loader(_te_cond)
                        _emb_tag = f"train_{_train_key}_test_{_test_key}"
                        print("\n" + "-" * 80)
                        print(
                            f"[STMO Eval] phase={phase_name} train={_train_key} "
                            f"test={_test_key} n={len(_te_cond)}"
                        )
                        print("-" * 80)
                        eval_ret = _run_iso_domain_eval(detector, iso_domain, _te_loader, _emb_tag)
                        if isinstance(eval_ret, dict):
                            _task1_res = eval_ret.get("task1_results", {}) or {}
                            _task2_res = eval_ret.get("task2_results", {}) or {}
                            for _name, _res in _task1_res.items():
                                if _res is not None:
                                    phase_eval_results["Task1"][_name] = _res
                            for _name, _res in _task2_res.items():
                                if _res is not None:
                                    phase_eval_results["Task2"][_name] = _res
                            _zs = _extract_domain_zero_shot(eval_ret, iso_domain)
                            _acc, _rec, _f1 = _pick_stmo_metrics(_zs)
                            if _acc == _acc and iso_domain in stmo_records:
                                stmo_records[iso_domain].append(_acc)
                            if iso_domain in stmo_detail_rows:
                                stmo_detail_rows[iso_domain].append(
                                    {
                                        "phase_name": phase_name,
                                        "train_group": _train_key,
                                        "test_group": _test_key,
                                        "acc": _acc,
                                        "recall": _rec,
                                        "macro_f1": _f1,
                                    }
                                )
            else:
                eval_ret = None
                if _nested_select_phase:
                    _inner_group = _loo_phase_meta.get(phase_name, {}).get("nested_inner_val_group", "unknown")
                    print(f"[LOO-Nested-Select] outer_test={_loo_phase_meta.get(phase_name, {}).get('outer_test_group', 'unknown')} inner_score_group={_inner_group}")
                    eval_ret = _run_iso_domain_eval(detector, iso_domain, _val_loader, f"nested_inner_{_inner_group}")
                elif iso_domain == "CWRU":
                    eval_ret = run_evaluation(detector, _cwru_test_loader, None, None, hit_loader=None,
                        task1_cwru_loader=_cwru_test_loader, task1_pu_loader=None,
                        emb_export_tag=_emb_export_tag)
                elif iso_domain == "PU":
                    eval_ret = run_evaluation(detector, None, _pu_test_loader, None, hit_loader=None,
                        task1_cwru_loader=None, task1_pu_loader=_pu_test_loader,
                        emb_export_tag=_emb_export_tag)
                elif iso_domain == "HUST":
                    eval_ret = run_evaluation(detector, None, None, _hust_loader, hit_loader=None,
                        task1_cwru_loader=None, task1_pu_loader=None,
                        emb_export_tag=_emb_export_tag)
                elif iso_domain == "SEU":
                    eval_ret = run_evaluation(detector, None, None, None, hit_loader=_hit_loader,
                        task1_cwru_loader=None, task1_pu_loader=None,
                        emb_export_tag=_emb_export_tag)
                elif iso_domain == "HIT":
                    eval_ret = run_evaluation(detector, None, None, None, hit_loader=_hit_loader,
                        task1_cwru_loader=None, task1_pu_loader=None,
                        emb_export_tag=_emb_export_tag)
                else:
                    eval_ret = run_evaluation(detector, None, None, _hust_loader, hit_loader=_hit_loader,
                        task1_cwru_loader=None, task1_pu_loader=None,
                        twc_source_cwru_loader=_cwru_test_loader, twc_source_pu_loader=_pu_test_loader,
                        emb_export_tag=_emb_export_tag)
                if isinstance(eval_ret, dict):
                    _task1_res = eval_ret.get("task1_results", {}) or {}
                    _task2_res = eval_ret.get("task2_results", {}) or {}
                    for _name, _res in _task1_res.items():
                        if _res is not None:
                            phase_eval_results["Task1"][_name] = _res
                    for _name, _res in _task2_res.items():
                        if _res is not None:
                            phase_eval_results["Task2"][_name] = _res
                    if _loo_enable:
                        if iso_domain in ("CWRU", "PU"):
                            _dres = _task1_res.get(iso_domain, {})
                        elif iso_domain in ("HUST", "SEU", "HIT"):
                            _dres = _task2_res.get(iso_domain, {})
                        else:
                            _dres = {}
                        if isinstance(_dres, dict):
                            _zs = _dres.get("zero_shot", {}) if isinstance(_dres.get("zero_shot", {}), dict) else {}
                            _acc, _rec, _f1 = _pick_loo_metrics(_zs)
                            if _acc == _acc and iso_domain in loo_records:
                                loo_records[iso_domain].append(_acc)
                            _meta = _loo_phase_meta.get(phase_name, {})
                            if iso_domain in loo_detail_rows and _meta:
                                loo_detail_rows[iso_domain].append(
                                    {
                                        "train_groups": _meta.get("train_groups", []),
                                        "test_group": _meta.get("test_group", "unknown"),
                                        "acc": _acc,
                                        "recall": _rec,
                                        "macro_f1": _f1,
                                    }
                                )

        def _fmt_float(_v):
            try:
                _x = float(_v)
                if _x != _x:
                    return "N/A"
                return f"{_x:.4f}"
            except Exception:
                return "N/A"

        if _loo_enable:
            print("\n" + "-" * 80)
            print(f"[LOO Summary] metric_source={_loo_metric_source} (final_decision=统一最终决策输出) mean±std")
            for _dom, _vals in loo_records.items():
                if len(_vals) == 0:
                    continue
                _arr = torch.tensor(_vals, dtype=torch.float32)
                print(f"  {_dom}: {float(_arr.mean().item()):.4f} ± {float(_arr.std(unbiased=False).item()):.4f} (n={len(_vals)})")
            print("-" * 80)
            for _dom in ("CWRU", "PU", "HUST", "SEU", "HIT"):
                _rows = loo_detail_rows.get(_dom, [])
                if not _rows:
                    continue
                print("\n" + "=" * 96)
                print(f"【{_dom} 域内跨工况训练结果】")
                print("=" * 96)
                print(f"  {'训练工况':<40} | {'测试工况':<20} | {'Acc':>8} | {'Recall':>8} | {'macro_f1':>8}")
                print("-" * 96)
                for _r in _rows:
                    _tr = ",".join(str(x) for x in _r.get("train_groups", []))
                    _te = str(_r.get("test_group", "unknown"))
                    print(
                        f"  {_tr[:40]:<40} | {_te[:20]:<20} | "
                        f"{_fmt_float(_r.get('acc')):>8} | {_fmt_float(_r.get('recall')):>8} | {_fmt_float(_r.get('macro_f1')):>8}"
                    )
                print("=" * 96)

        if _stmo_enable:
            print("\n" + "-" * 80)
            print(f"[STMO Summary] metric_source={_stmo_metric_source} (per held-out test condition) mean±std")
            for _dom, _vals in stmo_records.items():
                if len(_vals) == 0:
                    continue
                _arr = torch.tensor(_vals, dtype=torch.float32)
                print(f"  {_dom}: {float(_arr.mean().item()):.4f} ± {float(_arr.std(unbiased=False).item()):.4f} (n={len(_vals)})")
            print("-" * 80)
            _all_stmo_rows = []
            for _dom in ("CWRU", "PU", "HUST", "SEU", "HIT"):
                _rows = stmo_detail_rows.get(_dom, [])
                if not _rows:
                    continue
                print("\n" + "=" * 96)
                print(f"【{_dom} STMO 单工况训练 / 多工况分别测试结果】")
                print("=" * 96)
                print(f"  {'训练工况':<40} | {'测试工况':<20} | {'Acc':>8} | {'Recall':>8} | {'macro_f1':>8}")
                print("-" * 96)
                for _r in _rows:
                    _tr = str(_r.get("train_group", "unknown"))
                    _te = str(_r.get("test_group", "unknown"))
                    print(
                        f"  {_tr[:40]:<40} | {_te[:20]:<20} | "
                        f"{_fmt_float(_r.get('acc')):>8} | {_fmt_float(_r.get('recall')):>8} | {_fmt_float(_r.get('macro_f1')):>8}"
                    )
                    _all_stmo_rows.append(
                        {
                            "domain": _dom,
                            "phase_name": _r.get("phase_name", ""),
                            "train_group": _tr,
                            "test_group": _te,
                            "acc": _r.get("acc"),
                            "recall": _r.get("recall"),
                            "macro_f1": _r.get("macro_f1"),
                        }
                    )
                print("=" * 96)
            if len(_all_stmo_rows) > 0:
                _results_dir = os.path.join(os.path.dirname(__file__), "experiments", "results")
                os.makedirs(_results_dir, exist_ok=True)
                _csv_path = os.path.join(
                    _results_dir,
                    f"STMO_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                with open(_csv_path, "w", newline="", encoding="utf-8") as _csv_f:
                    _writer = csv.DictWriter(
                        _csv_f,
                        fieldnames=["domain", "phase_name", "train_group", "test_group", "acc", "recall", "macro_f1"],
                    )
                    _writer.writeheader()
                    _writer.writerows(_all_stmo_rows)
                print(f"[STMO] Saved summary CSV: {_csv_path}")
        print("\n" + "=" * 80)
        print("【三阶段汇总 (All-Phases Summary)】")
        print("=" * 80)
        for name, ckpt in phase_ckpt_paths:
            print(f"  {name}: {ckpt}")
        print("=" * 80)
        print("\n" + "=" * 112)
        print("【全阶段最终性能汇总表 (Final Performance Summary - All Stages)】")
        print("=" * 112)
        print("  异常检测 (Normal vs Fault)")
        print("-" * 112)
        print(f"  {'Dataset':<10} {'Phase':<12} {'AUC_nf(ZS)':>12} {'AUC_nf(Sup)':>12} {'Δ(Sup-ZS)':>12}")
        print("-" * 112)

        def _fmt_metric(v):
            try:
                _v = float(v)
                if _v != _v:
                    return "N/A"
                return f"{_v:.4f}"
            except Exception:
                return "N/A"

        def _fmt_delta(a, b):
            try:
                _a = float(a)
                _b = float(b)
                if (_a != _a) or (_b != _b):
                    return "N/A"
                return f"{(_a - _b):+.4f}"
            except Exception:
                return "N/A"

        _rows = []
        _phase_mapping = [
            ("CWRU", "Task1", "Phase 1/2"),
            ("PU", "Task1", "Phase 1/2"),
            ("HUST", "Task2", "Phase 3"),
            ("SEU", "Task2", "Phase 3"),
        ]
        for _dname, _task_name, _phase_label in _phase_mapping:
            _res = phase_eval_results.get(_task_name, {}).get(_dname, {})
            _zs = _res.get("zero_shot", {}) if isinstance(_res, dict) else {}
            _sup = _res.get("supervised", {}) if isinstance(_res, dict) else {}
            _rows.append(
                {
                    "dataset": _dname,
                    "phase": _phase_label,
                    "zs_auc_nf": _zs.get("auc_nf", float("nan")),
                    "sup_auc_nf": _sup.get("auc_nf", float("nan")),
                    "zs_spec_head": _zs.get("spec_head_diag_acc", float("nan")),
                    "zs_fusion_diag": _zs.get("diag_acc", float("nan")),
                    "sup_final_diag": _sup.get("diag_acc", float("nan")),
                    "zs_ens_diag": _zs.get("ensemble_diag_acc", float("nan")),
                    "zs_ens_f1": _zs.get("ensemble_macro_f1", float("nan")),
                }
            )

        for _r in _rows:
            print(
                f"  {_r['dataset']:<10} {_r['phase']:<12} "
                f"{_fmt_metric(_r['zs_auc_nf']):>12} {_fmt_metric(_r['sup_auc_nf']):>12} "
                f"{_fmt_delta(_r['sup_auc_nf'], _r['zs_auc_nf']):>12}"
            )

        print("-" * 112)
        print("  故障诊断 (Diagnosis)")
        print("-" * 112)
        print(
            f"  {'Dataset':<10} {'Phase':<12} {'SpecHead(ZS)':>14} {'Fusion(ZS)':>12} "
            f"{'Final(Sup)':>12} {'Ens(ZS)':>10} {'EnsF1(ZS)':>12}"
        )
        print("-" * 112)
        for _r in _rows:
            print(
                f"  {_r['dataset']:<10} {_r['phase']:<12} "
                f"{_fmt_metric(_r['zs_spec_head']):>14} {_fmt_metric(_r['zs_fusion_diag']):>12} "
                f"{_fmt_metric(_r['sup_final_diag']):>12} {_fmt_metric(_r['zs_ens_diag']):>10} "
                f"{_fmt_metric(_r['zs_ens_f1']):>12}"
            )
        print("=" * 112)

    else:
        # ===== 3. 计算类别权重和创建采样器 =====
        sampler, precomputed_class_weights = compute_class_weights_and_sampler(
            train_dataset, cfg, device
        )

        # ===== 4. 创建DataLoader =====
        train_loader, cwru_test_loader, pu_test_loader, hust_loader, hit_loader = setup_dataloaders(
            cfg, train_dataset, cwru_test, pu_test, hust_dataset, sampler, hit_dataset=seu_dataset
        )
        
        # 创建验证集 DataLoader（用于 early-stop，不使用 sampler）
        # 必须使用 multimodal_collate_fn 以生成 signal_mask 和 spec_mask
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
            persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False,
            generator=torch.Generator().manual_seed(int(cfg.RANDOM_SEED) + 2000),  # isolate validation worker seeding from training RNG
        )
    
        # ===== 5. 初始化模型 =====
        signal_encoder, spec_encoder, text_encoder, clip_model, fusion_module, fusion_decoder, detector, wst_encoder, env_encoder, scd_encoder = create_models(
            cfg, device
        )
    
        # ===== 6. 初始化优化器和训练模式处理 =====
        mode = cfg.TRAIN_MODE
        checkpoint_path = cfg.CHECKPOINT_PATH
        checkpoint_tag = infer_checkpoint_tag(checkpoint_path)
        
        start_epoch = 0
        best_loss = float("inf")
        optimizer = None
        signal_lr_mult = float(getattr(cfg, "SIGNAL_ENCODER_LR_MULT", 1.0))
    
        if mode in ["resume", "retrain"]:
            # NOTE:
            # - For "resume": keep the optimizer structure unchanged to avoid checkpoint optimizer-state mismatch.
            # - For "retrain": add a low-lr param group for the late-enabled diag_linear head (D2).
            if mode == "resume":
                signal_params = list(signal_encoder.parameters())
                other_params = (
                    list(spec_encoder.parameters()) +
                    list(fusion_module.parameters()) +
                    list(fusion_decoder.parameters()) +
                    (list(wst_encoder.parameters()) if wst_encoder is not None else []) +
                    (list(env_encoder.parameters()) if env_encoder is not None else []) +
                    (list(scd_encoder.parameters()) if scd_encoder is not None else [])
                )
                optimizer = torch.optim.Adam(
                    [
                        {"params": signal_params, "lr": cfg.LEARNING_RATE * signal_lr_mult},
                        {"params": other_params, "lr": cfg.LEARNING_RATE},
                    ]
                )
            else:
                diag_lr_mult = float(getattr(cfg, "DIAG_LINEAR_LR_MULT", 0.10))
                # Collect D2 params (may be frozen initially; still include them in optimizer)
                diag_params = []
                if hasattr(fusion_decoder, "diag_linear_head"):
                    diag_params += list(fusion_decoder.diag_linear_head.parameters())
                if hasattr(fusion_decoder, "diag_linear_scale"):
                    diag_params += [fusion_decoder.diag_linear_scale]
                diag_param_ids = {id(p) for p in diag_params}
                signal_param_ids = {id(p) for p in signal_encoder.parameters()}
    
                signal_params = []
                base_params = []
                _all_params = (list(signal_encoder.parameters()) +
                               list(spec_encoder.parameters()) +
                               list(fusion_module.parameters()) +
                               list(fusion_decoder.parameters()) +
                               (list(wst_encoder.parameters()) if wst_encoder is not None else []) +
                               (list(env_encoder.parameters()) if env_encoder is not None else []) +
                               (list(scd_encoder.parameters()) if scd_encoder is not None else []))
                for p in _all_params:
                    if id(p) not in diag_param_ids:
                        if id(p) in signal_param_ids:
                            signal_params.append(p)
                        else:
                            base_params.append(p)
    
                optimizer = torch.optim.Adam(
                    [
                        {"params": signal_params, "lr": cfg.LEARNING_RATE * signal_lr_mult},
                        {"params": base_params, "lr": cfg.LEARNING_RATE},
                        {"params": diag_params, "lr": cfg.LEARNING_RATE * diag_lr_mult},
                    ]
                )
    
        # ===== 7. 根据训练模式执行相应操作 =====
        if mode == "resume":
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Cannot resume training: checkpoint not found at {checkpoint_path}")
    
            print(f"Mode: Resume training")
            start_epoch, best_loss = load_model_checkpoint(
                checkpoint_path, signal_encoder, spec_encoder, text_encoder,
                fusion_module, fusion_decoder, optimizer=optimizer, device=device,
            )
            print(f"Resuming from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
    
            # 继续训练
            history = train_alignment(**build_train_alignment_kwargs(
                cfg, train_loader, signal_encoder, spec_encoder, text_encoder,
                fusion_module, fusion_decoder, optimizer, device, clip_model,
                hust_loader, start_epoch, precomputed_class_weights, checkpoint_tag,
                hit_loader=hit_loader, val_loader=val_loader, wst_encoder=wst_encoder, env_encoder=env_encoder, scd_encoder=scd_encoder,
            ))
    
            # 训练结束后：若有 early-stop/best checkpoint 信息，则回到 best checkpoint 再做最终评估
            if isinstance(history, dict):
                best_ckpt = history.get("best_checkpoint_path", None)
                best_epoch = history.get("best_metric_epoch", None)
                if best_ckpt and os.path.exists(best_ckpt):
                    eval_ckpt = resolve_eval_checkpoint(best_ckpt)
                    print(f"[Early Stop] Reloading best checkpoint for evaluation: {eval_ckpt} (epoch={best_epoch})")
                    load_model_checkpoint(
                        eval_ckpt, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, optimizer=None, device=device
                    )
                    checkpoint_path = eval_ckpt
    
                    # ---- Best-epoch zero-shot (logged) ----
                    # Ensure prototypes match the best checkpoint weights.
                    load_checkpoint_prototypes(checkpoint_path, detector, cfg, device)
                    best_text_prototypes = detector._active_text_prototypes()
                    if hasattr(fusion_decoder, "set_diag_linear_ramp"):
                        # For evaluation/logging, use full contribution (no warmup scaling).
                        fusion_decoder.set_diag_linear_ramp(1.0)
                    if best_text_prototypes is not None and hust_loader is not None and not (use_isolated and isolated_domain in ("CWRU", "PU")):
                        print("\n" + "=" * 80)
                        print(f"[Zero-shot Evaluation Best Checkpoint Epoch {best_epoch}]")
                        print("=" * 80 + "\n")
                        best_metrics = zero_shot_eval_hust(
                            signal_encoder=signal_encoder,
                            spec_encoder=spec_encoder,
                            fusion_module=fusion_module,
                            fusion_decoder=fusion_decoder,
                            text_prototypes=best_text_prototypes,
                            hust_loader=hust_loader,
                            device=device,
                            enable_finetuning=False,
                            finetune_samples=0,
                            finetune_lr=cfg.ZERO_SHOT_FINETUNING_LR,
                            finetune_steps=cfg.ZERO_SHOT_FINETUNING_STEPS
                        )
                        # Keep formatting compact but informative (raw vs protected + confusion)
                        print(f"[Best Epoch Zero-shot] AUC_nf={best_metrics.get('auc_nf', float('nan'))}")
                        print(f"[Best Epoch Zero-shot] Fusion Decoder acc={best_metrics.get('fusion_decoder_acc', 0.0):.4f}")
                        if 'fusion_decoder_acc_raw' in best_metrics:
                            print(f"[Best Epoch Zero-shot] Fusion Decoder acc_raw={best_metrics.get('fusion_decoder_acc_raw', 0.0):.4f}")
                        cm = best_metrics.get("fusion_decoder_confusion_matrix", None)
                        if cm is not None:
                            try:
                                print("[Best Epoch Zero-shot] Confusion Matrix (Fusion Decoder):")
                                print(cm)
                            except Exception:
                                pass
                        cm_raw = best_metrics.get("fusion_decoder_confusion_matrix_raw", None)
                        if cm_raw is not None:
                            try:
                                print("[Best Epoch Zero-shot] Confusion Matrix raw (Fusion Decoder):")
                                print(cm_raw)
                            except Exception:
                                pass
                elif best_ckpt:
                    print(f"[Early Stop] Warning: best_checkpoint_path={best_ckpt} does not exist, using default checkpoint_path={checkpoint_path}")
    
        elif mode == "retrain":
            print(f"Mode: Retrain")
    
            if cfg.START_FROM_SCRATCH:
                print("Starting training from scratch")
                start_epoch = 0
            else:
                if os.path.exists(checkpoint_path):
                    print("Loading pretrained weights for initialization")
                    load_model_checkpoint(
                        checkpoint_path, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, optimizer=None, device=device
                    )
                    start_epoch = 0
                    print("Initialized with pretrained weights")
                else:
                    print("Warning: Checkpoint not found, starting from scratch")
                    start_epoch = 0
    
            # 开始训练
            history = train_alignment(**build_train_alignment_kwargs(
                cfg, train_loader, signal_encoder, spec_encoder, text_encoder,
                fusion_module, fusion_decoder, optimizer, device, clip_model,
                hust_loader, start_epoch, precomputed_class_weights, checkpoint_tag,
                hit_loader=hit_loader, val_loader=val_loader, wst_encoder=wst_encoder, env_encoder=env_encoder, scd_encoder=scd_encoder,
            ))
    
            # 训练结束后：若有 early-stop/best checkpoint 信息，则回到 best checkpoint 再做最终评估
            if isinstance(history, dict):
                best_ckpt = history.get("best_checkpoint_path", None)
                best_epoch = history.get("best_metric_epoch", None)
                if best_ckpt and os.path.exists(best_ckpt):
                    eval_ckpt = resolve_eval_checkpoint(best_ckpt)
                    print(f"[Early Stop] Reloading best checkpoint for evaluation: {eval_ckpt} (epoch={best_epoch})")
                    load_model_checkpoint(
                        eval_ckpt, signal_encoder, spec_encoder, text_encoder,
                        fusion_module, fusion_decoder, optimizer=None, device=device
                    )
                    checkpoint_path = eval_ckpt
    
                    # ---- Best-epoch zero-shot (logged) ----
                    load_checkpoint_prototypes(checkpoint_path, detector, cfg, device)
                    best_text_prototypes = detector._active_text_prototypes()
                    if hasattr(fusion_decoder, "set_diag_linear_ramp"):
                        fusion_decoder.set_diag_linear_ramp(1.0)
                    if best_text_prototypes is not None and hust_loader is not None and not (use_isolated and isolated_domain in ("CWRU", "PU")):
                        print("\n" + "=" * 80)
                        print(f"[Zero-shot Evaluation Best Checkpoint Epoch {best_epoch}]")
                        print("=" * 80 + "\n")
                        best_metrics = zero_shot_eval_hust(
                            signal_encoder=signal_encoder,
                            spec_encoder=spec_encoder,
                            fusion_module=fusion_module,
                            fusion_decoder=fusion_decoder,
                            text_prototypes=best_text_prototypes,
                            hust_loader=hust_loader,
                            device=device,
                            enable_finetuning=False,
                            finetune_samples=0,
                            finetune_lr=cfg.ZERO_SHOT_FINETUNING_LR,
                            finetune_steps=cfg.ZERO_SHOT_FINETUNING_STEPS
                        )
                        print(f"[Best Epoch Zero-shot] AUC_nf={best_metrics.get('auc_nf', float('nan'))}")
                        print(f"[Best Epoch Zero-shot] Fusion Decoder acc={best_metrics.get('fusion_decoder_acc', 0.0):.4f}")
                        if 'fusion_decoder_acc_raw' in best_metrics:
                            print(f"[Best Epoch Zero-shot] Fusion Decoder acc_raw={best_metrics.get('fusion_decoder_acc_raw', 0.0):.4f}")
                        cm = best_metrics.get("fusion_decoder_confusion_matrix", None)
                        if cm is not None:
                            try:
                                print("[Best Epoch Zero-shot] Confusion Matrix (Fusion Decoder):")
                                print(cm)
                            except Exception:
                                pass
                        cm_raw = best_metrics.get("fusion_decoder_confusion_matrix_raw", None)
                        if cm_raw is not None:
                            try:
                                print("[Best Epoch Zero-shot] Confusion Matrix raw (Fusion Decoder):")
                                print(cm_raw)
                            except Exception:
                                pass
                elif best_ckpt:
                    print(f"[Early Stop] Warning: best_checkpoint_path={best_ckpt} does not exist, using default checkpoint_path={checkpoint_path}")
    
        elif mode == "test":
            print(f"Mode: Test only")
    
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Cannot test: checkpoint not found at {checkpoint_path}")
    
            # 仅加载模型进行测试
            load_model_checkpoint(
                checkpoint_path, signal_encoder, spec_encoder, text_encoder,
                fusion_module, fusion_decoder, optimizer=None, device=device
            )
            print("Model loaded for testing")
    
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
        # ===== 8. 加载checkpoint中的text_prototypes =====
        # 确保使用与模型权重一致的 checkpoint_path（特别是 best_checkpoint_path）
        # 这样 text_prototypes 和模型权重来自同一个 checkpoint，避免错配
        checkpoint_path, checkpoint_exists = load_checkpoint_prototypes(
            checkpoint_path, detector, cfg, device
        )
    
        # ===== 9. 运行评估 =====
        # 域内隔离时只评估对应 Task1，不跑另一域和 Task2（避免无效/误导性数字）
        if use_isolated and isolated_domain == "CWRU":
            run_evaluation(
                detector,
                cwru_test_loader,
                None,
                None,
                hit_loader=None,
                task1_cwru_loader=cwru_test_loader,
                task1_pu_loader=None,
            )
        elif use_isolated and isolated_domain == "PU":
            run_evaluation(
                detector,
                None,
                pu_test_loader,
                None,
                hit_loader=None,
                task1_cwru_loader=None,
                task1_pu_loader=pu_test_loader,
            )
        else:
            run_evaluation(
                detector,
                cwru_test_loader,
                pu_test_loader,
                hust_loader,
                hit_loader=hit_loader,
                task1_cwru_loader=cwru_test_loader if enable_intra_condition else None,
                task1_pu_loader=pu_test_loader if enable_intra_condition else None,
            )
    
        # ===== 10. (Attempt 49) HUST-best checkpoint LP-FT 对比评估 =====
        # Rationale: val_loss-best (E21) ≠ HUST-best (E10). After E10, source domain overfitting
        # degrades target-domain representations. LP-FT on E21 gives 26.18% (Att 48) while E10
        # representations are likely more linearly separable for HUST (Inner R=0.386 vs 0.045).
        # This block loads HUST-best checkpoint and runs an independent HUST LP-FT, then restores
        # the original model state. Zero training risk — purely diagnostic.
        _run_hust_best_lp_ft = False
        hust_best_ckpt_path = None
        hust_best_epoch_num = None
        try:
            _history_exists = isinstance(history, dict)
        except NameError:
            _history_exists = False
        if _history_exists:
            hust_best_ckpt_path = history.get("hust_best_checkpoint_path", None)
            hust_best_epoch_num = history.get("hust_best_epoch", None)
            val_loss_best_ckpt = history.get("best_checkpoint_path", None)
            # Only run if HUST-best checkpoint exists and differs from val_loss-best
            if (hust_best_ckpt_path
                and os.path.exists(hust_best_ckpt_path)
                and hust_best_ckpt_path != val_loss_best_ckpt
                and hust_loader is not None):
                _run_hust_best_lp_ft = True
    
        if _run_hust_best_lp_ft:
            print("\n" + "=" * 80)
            print(f"[Att49 Dual-Checkpoint] HUST-Best LP-FT Comparison (epoch={hust_best_epoch_num})")
            print("=" * 80)
            print(f"[Att49] Loading HUST-best checkpoint: {hust_best_ckpt_path}")
            print(f"[Att49] This is independent from the val_loss-best evaluation above.")
    
            # --- Save current (val_loss-best) full model state ---
            _saved_states = {}
            for name, module in [("signal_encoder", signal_encoder),
                                 ("spec_encoder", spec_encoder),
                                 ("text_encoder", text_encoder),
                                 ("fusion_module", fusion_module),
                                 ("fusion_decoder", fusion_decoder)]:
                _saved_states[name] = {k: v.clone().detach() for k, v in module.state_dict().items()}
    
            try:
                # --- Load HUST-best checkpoint ---
                load_model_checkpoint(
                    hust_best_ckpt_path, signal_encoder, spec_encoder, text_encoder,
                    fusion_module, fusion_decoder, optimizer=None, device=device
                )
                # Load matching text prototypes
                load_checkpoint_prototypes(hust_best_ckpt_path, detector, cfg, device)
    
                # --- Run isolated HUST LP-FT ---
                from utils.evaluation_utils import split_loader_for_target_adapt
                import config as _cfg
    
                _target_finetune_pct = getattr(_cfg, 'TARGET_FINETUNE_PERCENT', 0.05)
                _target_sup_steps = getattr(_cfg, 'TARGET_SUP_STEPS', None)
                _target_sup_lr = getattr(_cfg, 'TARGET_SUP_LR', None)
                _target_sup_nf_weight = getattr(_cfg, 'TARGET_SUP_NF_WEIGHT', 0.5)
                _target_sup_diag_weight = getattr(_cfg, 'TARGET_SUP_DIAG_WEIGHT', 1.0)
    
                detector.snapshot_fusion_state()
    
                hust_ft_loader, hust_ev_loader = split_loader_for_target_adapt(
                    hust_loader, finetune_percent=_target_finetune_pct, seed=42, dataset_name="HUST"
                )
                print(f"[Att49 HUST-Best] Split: {len(hust_ft_loader.dataset)} finetune, "
                      f"{len(hust_ev_loader.dataset)} eval")
    
                # Log finetune label distribution
                try:
                    counts = {}
                    for batch in hust_ft_loader:
                        y = batch.get("label", None)
                        if y is None:
                            continue
                        for v in y.detach().cpu().tolist():
                            counts[int(v)] = counts.get(int(v), 0) + 1
                    counts_line = ", ".join([f"{k}:{counts.get(k, 0)}" for k in sorted(counts.keys())])
                    print(f"[Att49 HUST-Best] Finetune label counts: {counts_line}")
                except Exception:
                    pass
    
                # Skip unsupervised adaptation for HUST (same as standard eval)
                print(f"[Att49 HUST-Best] Stage 1: Skipped (TARGET_SKIP_HUST_UNSUP_ADAPT)")
    
                # Zero-shot baseline on HUST-best checkpoint
                detector.set_eval_nf_threshold(None)
                _zs_result = detector.evaluate_target_domain_accuracy(
                    hust_ev_loader, dataset_name="HUST(hust_best_zs)", diag_num_classes=4
                )
                print(f"[Att49 HUST-Best] Zero-shot diag_acc: {_zs_result['diag_acc']:.4f} "
                      f"({_zs_result['diag_correct']}/{_zs_result['diag_total']})")
    
                # Supervised LP-FT
                print(f"[Att49 HUST-Best] Stage 2: Supervised fine-tuning (5% labeled)")
                detector.supervised_finetune_on_loader(
                    hust_ft_loader, dataset_name="HUST(hust_best)", num_classes_eval=4,
                    num_steps=_target_sup_steps, lr=_target_sup_lr,
                    nf_weight=_target_sup_nf_weight, diag_weight=_target_sup_diag_weight
                )
    
                # Evaluate LP-FT result
                detector.set_eval_nf_threshold(None)
                _ft_result_finetune = detector.evaluate_target_domain_accuracy(
                    hust_ft_loader, dataset_name="HUST(hust_best_ft_split)", diag_num_classes=4
                )
                _ft_result = detector.evaluate_target_domain_accuracy(
                    hust_ev_loader, dataset_name="HUST(hust_best)", diag_num_classes=4
                )
    
                print(f"[Att49 HUST-Best] After LP-FT:")
                print(f"  Finetune split diag_acc: {_ft_result_finetune.get('diag_acc', 0.0):.4f}")
                print(f"  Eval diag_acc: {_ft_result['diag_acc']:.4f} "
                      f"({_ft_result['diag_correct']}/{_ft_result['diag_total']})")
                if 'diag_acc_raw' in _ft_result:
                    print(f"  Eval diag_acc_raw: {_ft_result['diag_acc_raw']:.4f}")
    
                # Restore fusion state from snapshot
                detector.restore_fusion_state()
                print(f"[Att49 HUST-Best] Restored fusion state after LP-FT")
    
            except Exception as e:
                print(f"[Att49 HUST-Best] Error during HUST-best LP-FT: {e}")
                import traceback
                traceback.print_exc()
    
            # --- Restore full model state (val_loss-best) ---
            try:
                for name, module in [("signal_encoder", signal_encoder),
                                     ("spec_encoder", spec_encoder),
                                     ("text_encoder", text_encoder),
                                     ("fusion_module", fusion_module),
                                     ("fusion_decoder", fusion_decoder)]:
                    module.load_state_dict(_saved_states[name])
                # Restore val_loss-best text prototypes
                load_checkpoint_prototypes(checkpoint_path, detector, cfg, device)
                print(f"[Att49] Restored val_loss-best model state")
            except Exception as e:
                print(f"[Att49] Warning: failed to restore val_loss-best state: {e}")
    
            del _saved_states
