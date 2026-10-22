import torch


def multimodal_collate_fn(batch):
    batch_size = len(batch)
    
    # ===== signal（优化：预分配和向量化） =====
    signals = [item["signal"] for item in batch]
    # 避免在子进程中访问device属性，先获取第一个tensor的属性
    first_signal = signals[0]
    sig_dtype = first_signal.dtype
    sig_device = first_signal.device if hasattr(first_signal, 'device') else None
    
    sig_lengths = torch.tensor([s.size(0) for s in signals], dtype=torch.long)
    if sig_device is not None:
        sig_lengths = sig_lengths.to(sig_device)
    max_sig_len = sig_lengths.max().item()

    # 预分配tensor，避免多次stack
    padded_signals = torch.zeros(batch_size, max_sig_len, dtype=sig_dtype)
    signal_masks = torch.zeros(batch_size, max_sig_len, dtype=torch.float32)
    if sig_device is not None:
        padded_signals = padded_signals.to(sig_device)
        signal_masks = signal_masks.to(sig_device)
    
    for i, s in enumerate(signals):
        s_len = s.size(0)
        padded_signals[i, :s_len] = s
        signal_masks[i, :s_len] = 1.0

    # =====================================================
    # ===== spectrogram（优化：预分配和向量化）=====
    # =====================================================
    specs = [item["spec"] for item in batch]           # [C, F, T_i]

    C, F = specs[0].size(0), specs[0].size(1)
    first_spec = specs[0]
    spec_dtype = first_spec.dtype
    spec_device = first_spec.device if hasattr(first_spec, 'device') else None
    
    spec_lengths = torch.tensor([s.size(2) for s in specs], dtype=torch.long)
    if spec_device is not None:
        spec_lengths = spec_lengths.to(spec_device)
    max_spec_len = spec_lengths.max().item()

    # 预分配tensor
    padded_specs = torch.zeros(batch_size, C, F, max_spec_len, dtype=spec_dtype)
    spec_masks = torch.zeros(batch_size, max_spec_len, dtype=torch.float32)
    if spec_device is not None:
        padded_specs = padded_specs.to(spec_device)
        spec_masks = spec_masks.to(spec_device)
    
    for i, s in enumerate(specs):
        t_len = s.size(2)
        padded_specs[i, :, :, :t_len] = s
        spec_masks[i, :t_len] = 1.0

    # Optional physical-frequency position axis [1, F, T]
    has_freq_pos = all(("spec_freq_pos" in item and item["spec_freq_pos"] is not None) for item in batch)
    padded_spec_freq_pos = None
    if has_freq_pos:
        spec_freq_list = [item["spec_freq_pos"] for item in batch]
        fp_c, fp_f = spec_freq_list[0].size(0), spec_freq_list[0].size(1)
        padded_spec_freq_pos = torch.zeros(batch_size, fp_c, fp_f, max_spec_len, dtype=spec_dtype)
        if spec_device is not None:
            padded_spec_freq_pos = padded_spec_freq_pos.to(spec_device)
        for i, fp in enumerate(spec_freq_list):
            t_len = fp.size(2)
            padded_spec_freq_pos[i, :, :, :t_len] = fp

    # ===== text / label / domain =====
    texts = [item["text"] for item in batch]
    # 使用列表推导式然后一次性转换为tensor，比循环append快
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    domains = [item["domain"] for item in batch]
    domain_ids = torch.tensor([int(item.get("domain_id", 0)) for item in batch], dtype=torch.long)
    conditions = [item.get("condition", "unknown") for item in batch]

    # ATT377-StatFiLM: stack stat_vec if present
    has_stat_vec = all(("stat_vec" in item and item["stat_vec"] is not None) for item in batch)
    stat_vec_batch = None
    if has_stat_vec:
        stat_vec_batch = torch.stack([item["stat_vec"] for item in batch], dim=0)  # [B, 11] (ATT379b-S2)

    # ATT433: stack phys_features for PhysFeatureProtoModule
    has_phys_feat = all(("phys_features" in item and item["phys_features"] is not None) for item in batch)
    phys_feat_batch = None
    if has_phys_feat:
        phys_feat_batch = torch.stack([item["phys_features"] for item in batch], dim=0)  # [B, 22]

    out = {
        "signal": padded_signals,
        "signal_mask": signal_masks,
        "spec": padded_specs,
        "spec_mask": spec_masks,
        "text": texts,
        "label": labels,
        "domain": domains,
        "domain_id": domain_ids,
        "condition": conditions,
    }
    if padded_spec_freq_pos is not None:
        out["spec_freq_pos"] = padded_spec_freq_pos
    if stat_vec_batch is not None:
        out["stat_vec"] = stat_vec_batch
    if phys_feat_batch is not None:
        out["phys_features"] = phys_feat_batch

    # ATT449-BUGFIX: forward bypass features (wst_spec, env_spec, bpfi_bpfo_ratio)
    has_wst = all(("wst_spec" in item and item["wst_spec"] is not None) for item in batch)
    if has_wst:
        out["wst_spec"] = torch.stack([item["wst_spec"] for item in batch], dim=0)

    has_env = all(("env_spec" in item and item["env_spec"] is not None) for item in batch)
    if has_env:
        out["env_spec"] = torch.stack([item["env_spec"] for item in batch], dim=0)

    has_ratio = all(("bpfi_bpfo_ratio" in item and item["bpfi_bpfo_ratio"] is not None) for item in batch)
    if has_ratio:
        out["bpfi_bpfo_ratio"] = torch.stack([item["bpfi_bpfo_ratio"] for item in batch], dim=0)

    return out