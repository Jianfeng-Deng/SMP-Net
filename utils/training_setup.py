"""
训练设置相关工具函数
包括日志设置、数据集加载、DataLoader创建、权重计算等
"""
import datetime
import multiprocessing
import os
import platform
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from MultimodalBearingDataset import MultimodalBearingDataset
from collate import multimodal_collate_fn
from utils.LoadDatasets import load_datasets
from utils.dataPreprocess import dataProcess
from utils.seed_utils import worker_init_fn
from utils.splitData import sample_condition_keys_from_real_distribution


class Tee:
    """同时输出到控制台和文件的类"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(cfg=None):
    """设置日志输出到文件和控制台
    
    Args:
        cfg: 配置对象（Config类实例），如果提供则会将配置参数写入日志文件
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    log_f = open(log_file, 'w', encoding='utf-8')
    
    # 如果提供了配置对象，先写入配置信息
    if cfg is not None:
        log_f.write("=" * 80 + "\n")
        log_f.write("训练配置参数\n")
        log_f.write("=" * 80 + "\n")
        log_f.write(f"日志文件创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("\n")
        
        # 获取Config类的所有属性（排除方法和私有属性）
        config_items = []
        for attr_name in dir(cfg):
            if not attr_name.startswith('_') and not callable(getattr(cfg, attr_name, None)):
                attr_value = getattr(cfg, attr_name, None)
                # 跳过方法
                if not callable(attr_value):
                    config_items.append((attr_name, attr_value))
        
        # 按属性名排序
        config_items.sort(key=lambda x: x[0])
        
        # 按类别分组写入
        current_section = None
        for attr_name, attr_value in config_items:
            # 根据属性名判断类别（简单启发式方法）
            section = None
            if any(keyword in attr_name for keyword in ['PATH', 'SAMPLING', 'WINDOW']):
                section = "数据配置"
            elif any(keyword in attr_name for keyword in ['SEED', 'DEVICE']):
                section = "基础配置"
            elif any(keyword in attr_name for keyword in ['DOMAIN', 'RESIDUAL']):
                section = "域残差配置"
            elif any(keyword in attr_name for keyword in ['FUSION', 'ATTENTION', 'CROSS']):
                section = "融合模块配置"
            elif any(keyword in attr_name for keyword in ['FOCAL', 'LABEL', 'SMOOTHING', 'W_']):
                section = "损失函数配置"
            elif any(keyword in attr_name for keyword in ['TRAIN', 'MODE', 'CHECKPOINT', 'EPOCH', 'BATCH', 'LEARNING']):
                section = "训练配置"
            elif any(keyword in attr_name for keyword in ['MOE', 'EXPERT', 'BAND']):
                section = "MoE配置"
            elif any(keyword in attr_name for keyword in ['BAYESIAN', 'PROTOTYPE']):
                section = "贝叶斯原型配置"
            elif any(keyword in attr_name for keyword in ['FINETUNING', 'DETECTOR', 'ZERO_SHOT']):
                section = "微调配置"
            elif any(keyword in attr_name for keyword in ['CONSISTENCY', 'REGULARIZATION']):
                section = "正则化配置"
            elif any(keyword in attr_name for keyword in ['SCHEDULE', 'EARLY_STOP', 'LR_DECAY']):
                section = "训练策略配置"
            elif any(keyword in attr_name for keyword in ['AMP', 'COMPILE', 'WORKER', 'PREFETCH']):
                section = "性能优化配置"
            elif any(keyword in attr_name for keyword in ['WEIGHT', 'SAMPLER', 'CLASS', 'MONITORING']):
                section = "样本不均衡处理配置"
            elif any(keyword in attr_name for keyword in ['TEST', 'INTERVAL', 'LAMBDA']):
                section = "评估配置"
            elif any(keyword in attr_name for keyword in ['FAULT', 'PROMPT', 'TEMPLATE', 'SEMANTIC']):
                section = "故障标签配置"
            else:
                section = "其他配置"
            
            # 写入分类标题
            if section != current_section:
                if current_section is not None:
                    log_f.write("\n")
                log_f.write(f"--- {section} ---\n")
                current_section = section
            
            # 格式化属性值
            if isinstance(attr_value, dict):
                # 对于字典，格式化输出
                dict_str = str(attr_value)
                if len(dict_str) > 200 or any(isinstance(v, (list, dict)) for v in attr_value.values()):  # 长字典或包含复杂结构，分行显示
                    log_f.write(f"  {attr_name}:\n")
                    for k, v in sorted(attr_value.items()):
                        if isinstance(v, list):
                            if len(v) > 3:
                                log_f.write(f"    {k}: [列表，共{len(v)}项，示例: {v[0] if v else 'N/A'}]\n")
                            else:
                                log_f.write(f"    {k}: {v}\n")
                        elif isinstance(v, dict):
                            if len(str(v)) > 100:
                                log_f.write(f"    {k}: {{字典，共{len(v)}项}}\n")
                            else:
                                log_f.write(f"    {k}: {v}\n")
                        else:
                            log_f.write(f"    {k}: {v}\n")
                else:
                    log_f.write(f"  {attr_name}: {attr_value}\n")
            elif isinstance(attr_value, list):
                if len(attr_value) > 5:
                    # 长列表只显示长度和前几项
                    preview = attr_value[:2] if len(attr_value) > 0 else []
                    log_f.write(f"  {attr_name}: [列表，共{len(attr_value)}项，前2项: {preview}]\n")
                else:
                    log_f.write(f"  {attr_name}: {attr_value}\n")
            else:
                log_f.write(f"  {attr_name}: {attr_value}\n")
        
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write("训练日志开始\n")
        log_f.write("=" * 80 + "\n\n")
        log_f.flush()
    
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    
    print(f"Training log saved to: {log_file}")
    return log_file


def load_and_preprocess_datasets(cfg):
    """加载并预处理所有数据集"""
    cwru_path = cfg.CWRU_PATH
    pu_path = cfg.PU_PATH
    hust_path = cfg.HUST_PATH
    seu_path = cfg.SEU_PATH
    hit_path = getattr(cfg, "HIT_PATH", None) if bool(getattr(cfg, "ENABLE_HIT_DATASET", True)) else None
    _hit_skip_reason = None
    
    # 检查路径是否存在
    for path in [cwru_path, pu_path, hust_path]:
        if not os.path.exists(path):
            print(f"警告：路径不存在: {path}")
    if seu_path is not None and not os.path.exists(seu_path):
        print(f"警告：SEU路径不存在: {seu_path}")
    if not bool(getattr(cfg, "ENABLE_HIT_DATASET", True)):
        _hit_skip_reason = "ENABLE_HIT_DATASET=0"
        hit_path = None
    elif hit_path is None or not str(hit_path).strip():
        _hit_skip_reason = "HIT_PATH is empty"
        hit_path = None
    elif not os.path.exists(hit_path):
        _hit_skip_reason = f"HIT_PATH not found: {hit_path}"
        print(f"警告：HIT路径不存在: {hit_path}")
        hit_path = None
    
    print("=" * 60)
    print("加载数据集")
    print("=" * 60)
    
    # 加载所有数据集
    try:
        (
            cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels,
            seu_data, seu_labels, hit_data, hit_labels,
            cwru_conditions, pu_conditions, hust_conditions, seu_conditions, hit_conditions,
        ) = load_datasets(
            cwru_path, pu_path, hust_path, seu_path=seu_path, hit_path=hit_path, include_conditions=True
        )
        if hit_path is not None and (hit_data is None or hit_labels is None or len(hit_labels) == 0):
            if _hit_skip_reason is None:
                _hit_skip_reason = "hitLoad returned empty dataset (check HIT/raw/healthy|inner|outer/*.npy)"
    except Exception as e:
        print(f"加载数据集失败: {e}")
        raise
    
    # ATT378: bypass LDA — feed raw signals directly to SignalEncoder (1D CNN).
    # ATT379b: skip LDA feature extraction and training entirely (saves ~30s startup).
    DataProcess = dataProcess(cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels)
    cwru_original, pu_original, hust_original = DataProcess._copy_origin_data_()
    print("[ATT379b] LDA fully skipped (no feature extraction or training): raw signals used directly")
    for _dn, _do in [("CWRU", cwru_original), ("PU", pu_original), ("HUST", hust_original)]:
        _lens = [len(s) if hasattr(s, '__len__') else 0 for s in (_do if isinstance(_do, list) else [_do[i] for i in range(min(5, len(_do)))])]
        print(f"  {_dn}: raw signal lengths (first 5) = {_lens}")
    
    # 转换为文本描述
    cwru_texts, pu_texts, hust_texts = DataProcess.data_label_to_prompt(cfg.FAULT_PROMPT_TEMPLATES)
    
    # --- Diffusion Augmentation (ATT85) ---
    # Auto-trigger when training samples are too few (CWRU ~134 → feature collapse)
    if getattr(cfg, 'DIFFUSION_AUGMENT_ENABLED', False):
        try:
            from augmentation.diffusion_augmentor import DiffusionAugmentor
            
            augmentor = DiffusionAugmentor(cfg)
            if augmentor.should_augment(len(cwru_labels), cfg.BATCH_SIZE):
                print(f"[Diffusion-Aug] CWRU samples ({len(cwru_labels)}) below collapse "
                      f"threshold, generating synthetic samples...")
                syn_signals, syn_labels = augmentor.train_and_generate(
                    X_origin=cwru_original,
                    labels=cwru_labels,
                    fs=12000,
                    domain="CWRU",
                )
                if len(syn_labels) > 0:
                    # ATT378: no LDA — merge raw synthetic signals directly
                    syn_texts = DataProcess._labels_to_texts_(
                        syn_labels, cfg.FAULT_PROMPT_TEMPLATES
                    )
                    print(f"[Diffusion-Aug] Merging {len(syn_labels)} synthetic + "
                          f"{len(cwru_labels)} original = {len(cwru_labels)+len(syn_labels)} "
                          f"total CWRU samples")
                    cwru_original = np.concatenate([cwru_original, syn_signals])
                    cwru_labels = np.concatenate([cwru_labels, syn_labels])
                    cwru_texts = cwru_texts + syn_texts
                    
                    # [ATT85] Construct is_synthetic mask (False for real, True for synthetic)
                    is_synthetic = np.concatenate([
                        np.zeros(len(cwru_original) - len(syn_signals), dtype=bool),
                        np.ones(len(syn_signals), dtype=bool)
                    ])
                    
                    # ATT503-A: assign synthetic condition keys from real CWRU distribution.
                    # Fallback to DIFFUSION_SYNTHETIC_CONDITION_KEY only when no real keys exist.
                    if cwru_conditions is not None:
                        _real_cond_list = list(cwru_conditions)
                        if len(_real_cond_list) > 0:
                            syn_conditions = sample_condition_keys_from_real_distribution(
                                _real_cond_list,
                                len(syn_labels),
                                seed=int(getattr(cfg, "RANDOM_SEED", 42)) + 503,
                            )
                        else:
                            _fb = str(
                                getattr(cfg, "DIFFUSION_SYNTHETIC_CONDITION_KEY", "speed_1750")
                            ).strip() or "speed_1750"
                            syn_conditions = [_fb] * len(syn_labels)
                        cwru_conditions = _real_cond_list + syn_conditions
                        _syn_uniq, _syn_cnt = np.unique(
                            np.asarray(syn_conditions, dtype=object), return_counts=True
                        )
                        _syn_dist = ", ".join(
                            f"{k}:{int(c)}" for k, c in zip(_syn_uniq.tolist(), _syn_cnt.tolist())
                        )
                        print(
                            f"[Diffusion-Aug] Synthetic condition_keys sampled from real "
                            f"distribution (n={len(syn_labels)}): {_syn_dist}"
                        )
                else:
                    print("[Diffusion-Aug] No synthetic samples generated, continuing "
                          "with original data.")
                    is_synthetic = None
            else:
                print(f"[Diffusion-Aug] CWRU samples ({len(cwru_labels)}) sufficient, "
                      f"skipping augmentation.")
                is_synthetic = None
        except Exception as e:
            print(f"[Diffusion-Aug] WARNING: Augmentation failed ({e}), "
                  f"continuing with original data.")
            import traceback
            traceback.print_exc()
            is_synthetic = None
    else:
        is_synthetic = None
    
    # ATT378: signals=X_origin (raw) instead of LDA output
    cwru_dataset = MultimodalBearingDataset(
        X_origin=cwru_original, signals=cwru_original, texts=cwru_texts,
        labels=cwru_labels, fs=12000, n_fft=256, domain="CWRU", spec_augment=True,
        condition_keys=cwru_conditions,
        is_synthetic=is_synthetic,
    )
    pu_dataset = MultimodalBearingDataset(
        X_origin=pu_original, signals=pu_original, texts=pu_texts,
        labels=pu_labels, fs=64000, n_fft=256, domain="PU", spec_augment=True,
        condition_keys=pu_conditions,
    )
    hust_dataset = MultimodalBearingDataset(
        X_origin=hust_original, signals=hust_original, texts=hust_texts,
        labels=hust_labels, fs=25600, n_fft=256, domain="HUST", spec_augment=False,
        condition_keys=hust_conditions,
    )
    
    # 处理SEU数据集（如果存在）
    seu_dataset = None
    if seu_data is not None and seu_labels is not None:
        try:
            print(f"[SEU] SEU data shape: {seu_data.shape if hasattr(seu_data, 'shape') else type(seu_data)}")
            print(f"[SEU] SEU labels shape: {seu_labels.shape if hasattr(seu_labels, 'shape') else type(seu_labels)}")
            
            # 确保 seu_data 是正确的格式（numpy array of lists）
            if isinstance(seu_data, np.ndarray) and seu_data.dtype == object:
                seu_data_list = seu_data.tolist()
            elif isinstance(seu_data, np.ndarray) and seu_data.ndim == 2:
                seu_data_list = [seu_data[i] for i in range(len(seu_data))]
            else:
                seu_data_list = seu_data
            
            # ATT378: use raw signal directly (no LDA transform)
            print(f"[SEU] Using raw signals directly (LDA bypassed)")
            
            seu_texts = DataProcess._labels_to_texts_(
                labels=seu_labels,
                FAULT_PROMPT_TEMPLATES=cfg.FAULT_PROMPT_TEMPLATES
            )
            
            seu_dataset = MultimodalBearingDataset(
                X_origin=seu_data,
                signals=seu_data_list,
                texts=seu_texts,
                labels=seu_labels,
                fs=cfg.SAMPLING_RATES["SEU"],
                n_fft=256,
                domain="SEU",
                spec_augment=False,
                condition_keys=seu_conditions,
            )
            print(f"[SEU] Created SEU dataset with {len(seu_dataset)} samples")
        except Exception as e:
            print(f"[SEU] Warning: Failed to process SEU dataset: {e}")
            import traceback
            traceback.print_exc()
            seu_dataset = None

    # 处理HIT数据集（如果存在）
    hit_dataset = None
    if hit_data is not None and hit_labels is not None and len(hit_labels) > 0:
        try:
            print(f"[HIT] HIT data shape: {hit_data.shape if hasattr(hit_data, 'shape') else type(hit_data)}")
            print(f"[HIT] HIT labels shape: {hit_labels.shape if hasattr(hit_labels, 'shape') else type(hit_labels)}")

            if isinstance(hit_data, np.ndarray) and hit_data.dtype == object:
                hit_data_list = hit_data.tolist()
            elif isinstance(hit_data, np.ndarray) and hit_data.ndim == 2:
                hit_data_list = [hit_data[i] for i in range(len(hit_data))]
            else:
                hit_data_list = hit_data

            print("[HIT] Using raw signals directly (LDA bypassed)")

            hit_texts = DataProcess._labels_to_texts_(
                labels=hit_labels,
                FAULT_PROMPT_TEMPLATES=cfg.FAULT_PROMPT_TEMPLATES,
            )

            hit_dataset = MultimodalBearingDataset(
                X_origin=hit_data,
                signals=hit_data_list,
                texts=hit_texts,
                labels=hit_labels,
                fs=cfg.SAMPLING_RATES["HIT"],
                n_fft=256,
                domain="HIT",
                spec_augment=False,
                condition_keys=hit_conditions,
            )
            print(f"[HIT] Created HIT dataset with {len(hit_dataset)} samples")
        except Exception as e:
            print(f"[HIT] Warning: Failed to process HIT dataset: {e}")
            _hit_skip_reason = f"dataset build failed: {e}"
            import traceback
            traceback.print_exc()
            hit_dataset = None
    
    if hit_dataset is None:
        if _hit_skip_reason:
            print(f"[HIT] Dataset unavailable ({_hit_skip_reason}). "
                  f"Remove HIT from STMO_DOMAINS/LOO_DOMAINS or fix HIT_PATH.")
        elif bool(getattr(cfg, "ENABLE_HIT_DATASET", True)):
            print("[HIT] Dataset unavailable (unknown reason). Check logs above for [HIT] warnings.")
    elif hasattr(hit_dataset, "condition_id_to_key"):
        _hit_ng = len(getattr(hit_dataset, "condition_id_to_key", {}) or {})
        print(f"[HIT] Dataset ready: n={len(hit_dataset)} condition_groups={_hit_ng}")
        if _hit_ng < 2:
            print("[HIT] Warning: fewer than 2 condition groups — STMO/LOO folds for HIT will fail "
                  "under strict mode.")
    
    return cwru_dataset, pu_dataset, hust_dataset, seu_dataset, hit_dataset, DataProcess


def compute_class_weights_and_sampler(train_dataset, cfg, device):
    """
    计算类别权重并创建采样器
    
    Returns:
        sampler: WeightedRandomSampler or None
        precomputed_class_weights: torch.Tensor on device
    """
    # 提取所有训练样本的labels（支持 ConcatDataset(Subset, ...) 或 单个 Subset）
    all_labels = []
    if isinstance(train_dataset, Subset):
        subsets_to_iter = [train_dataset]
    elif hasattr(train_dataset, "datasets"):
        subsets_to_iter = train_dataset.datasets
    else:
        subsets_to_iter = [train_dataset]
    for subset in subsets_to_iter:
        if isinstance(subset, Subset):
            base_dataset = subset.dataset
            indices = subset.indices
            if not hasattr(base_dataset, 'labels'):
                raise AttributeError("Base dataset must have 'labels' attribute")
            labels = base_dataset.labels[indices]
            all_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
        elif hasattr(subset, "labels"):
            all_labels.extend(subset.labels.tolist() if hasattr(subset.labels, 'tolist') else list(subset.labels))
        else:
            raise TypeError("train_dataset (or each subset) must be Subset or have 'labels' attribute")
    all_labels = np.array(all_labels)
    
    # 统计类别分布
    class_counts = Counter(all_labels)
    num_classes = len(class_counts)
    total_samples = len(all_labels)
    
    print("Class counts:", class_counts)
    
    # 计算类别权重
    max_class_weight = cfg.MAX_CLASS_WEIGHT
    sampler_weight_scale = cfg.SAMPLER_WEIGHT_SCALE
    use_weighted_sampler = cfg.USE_WEIGHTED_SAMPLER
    
    class_weights = {}
    for cls, count in class_counts.items():
        w = total_samples / (num_classes * count)
        w = float(w) ** 0.5  # sqrt-inv frequency
        class_weights[cls] = min(w, max_class_weight)
    
    print("Class weights:", class_weights)
    print(f"Sampler weight scale: {sampler_weight_scale}, Use weighted sampler: {use_weighted_sampler}")
    
    # 创建采样器
    if use_weighted_sampler:
        sample_weights = np.array([
            1.0 + (class_weights[label] - 1.0) * sampler_weight_scale 
            for label in all_labels
        ])
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
    else:
        sampler = None
    
    # 预计算类别权重（用于loss）
    print("[Performance] Precomputing class_weights...")
    max_label = int(all_labels.max()) if all_labels.size > 0 else -1
    num_classes = max_label + 1
    counts = np.bincount(all_labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    class_weights_array = (1.0 / (counts + 1e-8)) ** 0.5
    class_weights_array = class_weights_array / (class_weights_array.mean() + 1e-8)
    class_weights_array = np.clip(class_weights_array, a_min=0.0, a_max=max_class_weight)
    class_weights_array = class_weights_array / (class_weights_array.mean() + 1e-8)
    precomputed_class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
    
    print(f"[Performance] Class weights precomputed: {dict(zip(range(num_classes), class_weights_array.tolist()))}")
    
    return sampler, precomputed_class_weights


def setup_dataloaders(cfg, train_dataset, cwru_test, pu_test, hust_dataset, sampler, hit_dataset=None):
    """创建所有DataLoader"""
    batch_size = cfg.BATCH_SIZE
    print(f"[DataLoader] Using batch_size={batch_size} (MoE={cfg.USE_MOE})")
    
    # Windows系统需要设置multiprocessing start method
    if platform.system() == 'Windows':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # 设置num_workers
    # Respect the explicit configuration on every platform.  The previous
    # Linux-only auto-expansion to 16 workers created unnecessary CPU processes
    # and made throughput hard to reproduce on the remote server.
    num_workers = max(0, int(getattr(cfg, "NUM_WORKERS", 0)))
    if platform.system() == 'Windows':
        print(f"[DataLoader] Windows detected: Using num_workers={num_workers} (multiprocessing disabled)")
    else:
        print(f"[DataLoader] Using configured num_workers={num_workers} (cpu_count={multiprocessing.cpu_count()})")
    
    # DataLoader配置（当num_workers=0时，prefetch_factor必须为None）
    prefetch_factor = cfg.PREFETCH_FACTOR if num_workers > 0 else None
    persistent_workers = cfg.PERSISTENT_WORKERS and num_workers > 0 and platform.system() != 'Windows'
    
    def create_dataloader(dataset, batch_size, shuffle=False, sampler=None):
        """创建DataLoader的辅助函数"""
        if dataset is None:  # 处理dataset为None的情况
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler if sampler is not None else None,
            shuffle=shuffle if sampler is None else False,
            collate_fn=multimodal_collate_fn,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None  # 设置worker随机种子
        )
    
    train_loader = create_dataloader(train_dataset, batch_size, shuffle=(sampler is None), sampler=sampler)
    cwru_test_loader = create_dataloader(cwru_test, batch_size, shuffle=False)
    pu_test_loader = create_dataloader(pu_test, batch_size, shuffle=False)
    hust_loader = create_dataloader(hust_dataset, batch_size, shuffle=False)
    
    # Keep variable name hit_loader for backward compatibility in call sites; now carries SEU.
    hit_loader = None
    if hit_dataset is not None:
        hit_loader = create_dataloader(hit_dataset, batch_size, shuffle=False)
        print(f"[DataLoader] Created target loader with {len(hit_dataset)} samples")
    
    if persistent_workers:
        print(f"[DataLoader] Using persistent_workers=True and prefetch_factor={prefetch_factor} for better performance")
    
    return train_loader, cwru_test_loader, pu_test_loader, hust_loader, hit_loader
