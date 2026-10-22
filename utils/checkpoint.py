import glob
import os
import re

import torch


def _to_cpu_text_prototypes(text_prototypes):
    """
    Ensure {label: tensor[D]} lives on CPU and is detached for safe serialization.
    """
    if text_prototypes is None:
        return None

    cpu_protos = {}
    for k, v in text_prototypes.items():
        if v is None:
            continue
        if not torch.is_tensor(v):
            raise TypeError(f"text_prototypes[{k}] must be a torch.Tensor, got {type(v)}")
        cpu_protos[int(k)] = v.detach().cpu()
    return cpu_protos


def is_checkpoint_readable(checkpoint_path: str, min_bytes: int = 1024) -> bool:
    """Return True if path exists and torch.load can read a full checkpoint zip."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return False
    try:
        if os.path.getsize(checkpoint_path) < int(min_bytes):
            return False
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def resolve_readable_checkpoint(
    primary_path: str,
    *,
    checkpoint_dir: str | None = None,
    checkpoint_tag: str | None = None,
    best_epoch: int | None = None,
    extra_candidates: list[str] | None = None,
) -> str | None:
    """
    Pick the first readable checkpoint from primary_path and fallbacks.
    Used when torch.save partially fails (truncated .pth) but training still records best epoch.
    """
    candidates: list[str] = []
    if primary_path:
        candidates.append(str(primary_path))
    if best_epoch is not None and checkpoint_dir:
        ep = int(best_epoch)
        tag = str(checkpoint_tag or "").strip()
        if tag:
            candidates.extend([
                os.path.join(checkpoint_dir, f"{tag}_epoch_{ep}_pre_best.pth"),
                os.path.join(checkpoint_dir, f"{tag}_epoch_{ep}_best.pth"),
            ])
        candidates.extend([
            os.path.join(checkpoint_dir, f"epoch_{ep}_pre_best.pth"),
            os.path.join(checkpoint_dir, f"epoch_{ep}_best.pth"),
            os.path.join(checkpoint_dir, f"epoch_{ep}_{tag}.pth") if tag else "",
        ])
    if extra_candidates:
        candidates.extend([str(p) for p in extra_candidates if p])
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        tag_glob = f"{checkpoint_tag}_" if checkpoint_tag else ""
        for pattern in (f"{tag_glob}*_pre_best.pth", f"{tag_glob}*_best.pth"):
            paths = glob.glob(os.path.join(checkpoint_dir, pattern))
            paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            candidates.extend(paths)

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if is_checkpoint_readable(path):
            if primary_path and path != primary_path:
                print(
                    f"[Checkpoint] Primary unreadable or missing ({primary_path}); "
                    f"using fallback: {path}"
                )
            return path
    return None


def save_checkpoint(save_path, signal_encoder, spec_encoder, text_encoder, fusion_module, fusion_decoder, optimizer, epoch, loss, text_prototypes=None, ema_manager=None, proto_adapter=None, learnable_proto_module=None, decouple_spec=None, decouple_signal=None, condition_classifier=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cpu_protos = _to_cpu_text_prototypes(text_prototypes)
    
    # Get spec_encoder state_dict (excludes expert_tracker which we handle separately)
    spec_encoder_state = spec_encoder.state_dict()
    
    # 准备保存的字典
    save_dict = {
        "epoch": epoch,
        "loss": loss,
        "signal_encoder": signal_encoder.state_dict(),
        "spec_encoder": spec_encoder_state,
        "text_encoder": text_encoder.state_dict(),
        "fusion_module": fusion_module.state_dict(),
        "fusion_decoder": fusion_decoder.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "text_prototypes": cpu_protos,
    }
    
    # ATT391-S1: save proto adapter state if available
    if proto_adapter is not None and hasattr(proto_adapter, 'state_dict'):
        save_dict["proto_adapter"] = proto_adapter.state_dict()

    # ATT392-S1: save learnable prototype state
    if learnable_proto_module is not None and hasattr(learnable_proto_module, 'state_dict'):
        save_dict["learnable_proto_module"] = learnable_proto_module.state_dict()

    # ATT396: save Feature Decoupler states
    if decouple_spec is not None and hasattr(decouple_spec, 'state_dict'):
        save_dict["decouple_spec"] = decouple_spec.state_dict()
    if decouple_signal is not None and hasattr(decouple_signal, 'state_dict'):
        save_dict["decouple_signal"] = decouple_signal.state_dict()
    if condition_classifier is not None and hasattr(condition_classifier, 'state_dict'):
        save_dict["condition_classifier"] = condition_classifier.state_dict()

    # Save expert_tracker state separately (if it exists)
    if hasattr(spec_encoder, 'expert_tracker') and spec_encoder.expert_tracker is not None:
        try:
            # Call state_dict without destination parameter to get the dict directly
            expert_tracker_state = spec_encoder.expert_tracker.state_dict()
            save_dict["expert_tracker_state"] = expert_tracker_state
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to save expert_tracker state: {e}")
    
    # 如果提供了 EMA manager，保存 EMA 状态
    if ema_manager is not None:
        save_dict["ema_state"] = ema_manager.state_dict()

    # Atomic write: avoid truncated zip when disk is full or process is killed mid-save.
    parent = os.path.dirname(os.path.abspath(save_path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".{os.path.basename(save_path)}.tmp")
    try:
        torch.save(save_dict, tmp_path)
        torch.load(tmp_path, map_location="cpu", weights_only=False)
        os.replace(tmp_path, save_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if cpu_protos is not None:
        ema_info = " with EMA" if ema_manager is not None else ""
        print(f"[Checkpoint] Saved to {save_path} (with text_prototypes: {sorted(cpu_protos.keys())}{ema_info})")
    else:
        ema_info = " with EMA" if ema_manager is not None else ""
        print(f"[Checkpoint] Saved to {save_path} (no text_prototypes{ema_info})")

def load_text_prototypes(checkpoint_path, device="cpu"):
    """
    Load only text_prototypes from a checkpoint (if present), without touching model weights.
    Returns: Dict[int, Tensor[D]] on `device`, or None if not found.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protos = checkpoint.get("text_prototypes", None)
    if protos is None:
        return None

    # move to requested device
    out = {}
    for k, v in protos.items():
        if v is None:
            continue
        out[int(k)] = v.to(device)
    return out


def _filter_state_dict(state_dict, model_state_dict):
    """
    过滤 state_dict，只保留形状匹配的参数
    """
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered[key] = value
            else:
                skipped.append((key, value.shape, model_state_dict[key].shape))
        else:
            skipped.append((key, value.shape, None))
    
    if skipped:
        print(f"[Checkpoint] Warning: Skipped {len(skipped)} mismatched parameters:")
        for key, ckpt_shape, model_shape in skipped[:5]:  # 只显示前5个
            print(f"  {key}: checkpoint shape {ckpt_shape}, model shape {model_shape}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    
    return filtered


def _expand_domain_residual(state_dict, model_state_dict, init_scale=0.0):
    """
    扩展 domain_residual / domain_lora 参数以兼容 checkpoint 域数 < 当前模型域数 的情况
    支持 3->4、4->5 或任意 N->M（M>N）。
    
    Args:
        state_dict: checkpoint中的state_dict
        model_state_dict: 当前模型的state_dict
        init_scale: 新域 domain_residual 的初始化缩放（默认0.0，与DomainFiLM的init_scale一致）
    
    Returns:
        扩展后的state_dict
    """
    expanded = state_dict.copy()
    
    # domain_residual.gamma / .beta：第一维为 num_domains
    gamma_key = "domain_residual.gamma"
    beta_key = "domain_residual.beta"
    
    for key, (default_val, is_gamma) in [(gamma_key, (init_scale, True)), (beta_key, (0.0, False))]:
        if key not in expanded or key not in model_state_dict:
            continue
        ckpt_t = expanded[key]
        model_t = model_state_dict[key]
        ckpt_d, model_d = ckpt_t.shape[0], model_t.shape[0]
        if ckpt_d >= model_d:
            continue
        # 扩展：新域 gamma 用 init_scale，beta 用 0（与 DomainFiLM 一致）
        new_slice = (torch.ones if is_gamma else torch.zeros)(
            model_d - ckpt_d, *ckpt_t.shape[1:],
            device=ckpt_t.device, dtype=ckpt_t.dtype
        )
        if is_gamma:
            new_slice = new_slice * init_scale
        expanded[key] = torch.cat([ckpt_t, new_slice], dim=0)
        print(f"[Checkpoint] Expanded {key} from {ckpt_t.shape} to {expanded[key].shape}")
    
    # domain_lora_fc.lora_A / lora_B：第一维为 num_domains，扩展新域用零（不改变原行为）
    lora_a_key = "domain_lora_fc.lora_A"
    lora_b_key = "domain_lora_fc.lora_B"
    for key in [lora_a_key, lora_b_key]:
        if key not in expanded or key not in model_state_dict:
            continue
        ckpt_t = expanded[key]
        model_t = model_state_dict[key]
        if ckpt_t.shape[0] >= model_t.shape[0]:
            continue
        new_slice = torch.zeros(
            model_t.shape[0] - ckpt_t.shape[0], *ckpt_t.shape[1:],
            device=ckpt_t.device, dtype=ckpt_t.dtype
        )
        expanded[key] = torch.cat([ckpt_t, new_slice], dim=0)
        print(f"[Checkpoint] Expanded {key} from {ckpt_t.shape} to {expanded[key].shape}")
    
    return expanded


def _expand_domain_embedding(state_dict, model_state_dict):
    """
    扩展 domain_embedding 参数以兼容 checkpoint 域数 < 当前模型域数（如 3->4、4->5）。
    
    Args:
        state_dict: checkpoint中的state_dict
        model_state_dict: 当前模型的state_dict
    
    Returns:
        扩展后的state_dict
    """
    expanded = state_dict.copy()
    
    embedding_key = "domain_embedding.weight"
    if embedding_key not in expanded or embedding_key not in model_state_dict:
        return expanded
    
    ckpt_emb = expanded[embedding_key]
    model_emb = model_state_dict[embedding_key]
    ckpt_d, model_d = ckpt_emb.shape[0], model_emb.shape[0]
    if ckpt_d >= model_d:
        return expanded
    
    # 新域用小随机值（与 nn.Embedding 默认初始化类似）
    embedding_dim = ckpt_emb.shape[1]
    new_domain_emb = torch.randn(model_d - ckpt_d, embedding_dim, device=ckpt_emb.device, dtype=ckpt_emb.dtype) * 0.01
    expanded[embedding_key] = torch.cat([ckpt_emb, new_domain_emb], dim=0)
    print(f"[Checkpoint] Expanded {embedding_key} from {ckpt_emb.shape} to {expanded[embedding_key].shape}")
    
    return expanded


_LOO_BEST_BASENAME_RE = re.compile(
    r"^(task1_[a-z]+_loo_\d+)_epoch_(\d+)_(?:best|pre_best)\.pth$",
    re.IGNORECASE,
)


def load_checkpoint(checkpoint_path, signal_encoder, spec_encoder, text_encoder, fusion_module, fusion_decoder, optimizer=None, device="cpu", return_text_prototypes=False, ema_manager=None, decouple_spec=None, decouple_signal=None, condition_classifier=None):
    ckpt_path = str(checkpoint_path or "").strip()
    if not ckpt_path:
        raise RuntimeError("Checkpoint path is empty.")

    if not is_checkpoint_readable(ckpt_path):
        _dir = os.path.dirname(os.path.abspath(ckpt_path)) or None
        _base = os.path.basename(ckpt_path)
        _tag: str | None = None
        _ep: int | None = None
        m = _LOO_BEST_BASENAME_RE.match(_base)
        if m:
            _tag = m.group(1)
            _ep = int(m.group(2))
        _fallback: str | None = None
        if _tag is not None and _dir:
            _fallback = resolve_readable_checkpoint(
                ckpt_path,
                checkpoint_dir=_dir,
                checkpoint_tag=_tag,
                best_epoch=_ep,
            )
        elif _dir and "_best.pth" in _base:
            _pre = os.path.join(_dir, _base.replace("_best.pth", "_pre_best.pth"))
            if is_checkpoint_readable(_pre):
                _fallback = _pre
        if _fallback and is_checkpoint_readable(_fallback):
            if os.path.abspath(_fallback) != os.path.abspath(ckpt_path):
                print(
                    f"[Checkpoint] Primary unreadable/corrupt: {ckpt_path}\n"
                    f"[Checkpoint] Using readable fallback: {_fallback}"
                )
            ckpt_path = _fallback
        else:
            raise RuntimeError(
                f"Checkpoint file is missing or corrupted (truncated/incomplete): {checkpoint_path}. "
                f"Delete it and re-train this LOO fold, or copy a valid *_pre_best.pth to the expected path."
            )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 加载模型参数，处理结构不匹配的情况
    # 扩展 domain_residual / domain_lora_fc 以兼容 checkpoint 域数 < 当前模型域数（如 3->4、4->5）
    signal_encoder_state_dict = checkpoint["signal_encoder"]
    signal_encoder_state_dict = _expand_domain_residual(
        signal_encoder_state_dict,
        signal_encoder.state_dict(),
        init_scale=0.0  # 与 DomainFiLM 的 init_scale 一致
    )
    signal_encoder.load_state_dict(signal_encoder_state_dict, strict=False)
    
    # spec_encoder：可能因 base_channels/num_domains 不同导致 shape 不匹配（如 checkpoint 32 vs 模型 64）
    try:
        spec_encoder.load_state_dict(checkpoint["spec_encoder"], strict=True)
    except RuntimeError:
        # 严格加载失败时，只加载 shape 一致的参数（避免 32→64 等结构不一致报错）
        print(f"[Checkpoint] Warning: SpecEncoder strict load failed (e.g. base_channels or num_domains mismatch), loading only matching shapes...")
        spec_filtered = _filter_state_dict(checkpoint["spec_encoder"], spec_encoder.state_dict())
        spec_encoder.load_state_dict(spec_filtered, strict=False)
        n_loaded = len(spec_filtered)
        n_total = len(spec_encoder.state_dict())
        if n_loaded == 0:
            print(f"[Checkpoint] Warning: No SpecEncoder weights loaded (architecture mismatch, e.g. base_channels 32 vs 64). SpecEncoder uses random init.")
        else:
            print(f"[Checkpoint] SpecEncoder: loaded {n_loaded}/{n_total} matching parameters.")
    
    text_encoder.load_state_dict(checkpoint["text_encoder"], strict=False)
    
    # 扩展 fusion_module 的 domain_embedding 以兼容从3域到4域的变化
    fusion_module_state_dict = checkpoint["fusion_module"]
    fusion_module_state_dict = _expand_domain_embedding(
        fusion_module_state_dict,
        fusion_module.state_dict()
    )
    fusion_module.load_state_dict(fusion_module_state_dict, strict=False)
    
    # 对于 fusion_decoder，先过滤掉形状不匹配的参数
    try:
        fusion_decoder.load_state_dict(checkpoint["fusion_decoder"], strict=False)
    except RuntimeError as e:
        # 如果仍然失败（由于形状不匹配），使用过滤后的 state_dict
        print(f"[Checkpoint] Warning: FusionDecoder loading failed due to shape mismatch, filtering parameters...")
        filtered_state_dict = _filter_state_dict(checkpoint["fusion_decoder"], fusion_decoder.state_dict())
        fusion_decoder.load_state_dict(filtered_state_dict, strict=False)

    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load optimizer state: {e}")

    # 加载 EMA 状态（如果存在）
    if ema_manager is not None and "ema_state" in checkpoint:
        try:
            ema_manager.load_state_dict(checkpoint["ema_state"])
            print(f"[Checkpoint] Loaded EMA state")
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load EMA state: {e}")
    
    # Load expert_tracker state separately (if it exists)
    if hasattr(spec_encoder, 'expert_tracker') and spec_encoder.expert_tracker is not None:
        if "expert_tracker_state" in checkpoint:
            try:
                spec_encoder.expert_tracker.load_state_dict(checkpoint["expert_tracker_state"])
                print(f"[Checkpoint] Loaded expert_tracker state")
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to load expert_tracker state: {e}")
        else:
            print(f"[Checkpoint] No expert_tracker state found in checkpoint (will use default initialization)")

    # ATT396: load Feature Decoupler states
    if decouple_spec is not None and "decouple_spec" in checkpoint:
        try:
            decouple_spec.load_state_dict(checkpoint["decouple_spec"], strict=False)
            print("[Checkpoint] Loaded decouple_spec state")
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load decouple_spec: {e}")
    if decouple_signal is not None and "decouple_signal" in checkpoint:
        try:
            decouple_signal.load_state_dict(checkpoint["decouple_signal"], strict=False)
            print("[Checkpoint] Loaded decouple_signal state")
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load decouple_signal: {e}")
    if condition_classifier is not None and "condition_classifier" in checkpoint:
        try:
            condition_classifier.load_state_dict(checkpoint["condition_classifier"], strict=False)
            print("[Checkpoint] Loaded condition_classifier state")
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load condition_classifier: {e}")

    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"[Checkpoint] Loaded from {ckpt_path} (epoch={start_epoch})")

    if return_text_prototypes:
        # load prototypes from CPU version if present; move to device
        protos = checkpoint.get("text_prototypes", None)
        if protos is not None:
            protos = {int(k): v.to(device) for k, v in protos.items() if v is not None}
        return start_epoch, loss, protos

    return start_epoch, loss