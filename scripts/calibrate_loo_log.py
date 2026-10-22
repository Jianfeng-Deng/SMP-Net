"""Parse Phase-3 summary blocks from a training log and emit calibrated LOO tables."""
from __future__ import annotations

import re
import sys
from pathlib import Path


CWRU_PHASES = ["gk2", "speed_1730", "speed_1750", "speed_1772", "speed_1797"]
PU_PHASES = ["gk5", "gk6", "gk7", "gk8"]
HUST_PHASES = [
    "speed_0Hz", "speed_20Hz", "speed_25Hz", "speed_30Hz", "speed_35Hz", "speed_40Hz",
    "speed_60Hz", "speed_65Hz", "speed_70Hz", "speed_75Hz", "speed_80Hz",
]
SEU_PHASES = ["speed_20_load_0", "speed_30_load_2"]


def _grab(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def _macro_recall_from_perclass(text: str, tag: str) -> float:
    if f"[{tag}] Per-class:" not in text:
        return float("nan")
    chunk = text.split(f"[{tag}] Per-class:")[1].split("[")[0]
    recs = [float(x) for x in re.findall(r"R=([\d.]+)", chunk)]
    return sum(recs) / len(recs) if recs else float("nan")


def parse_phase3_block(block: str) -> dict:
    top1 = _grab(block, r"Top-1 Acc \(最优路径\)\s+([\d.]+)")
    sh_acc = _grab(block, r"SpecHead Acc\s+([\d.]+)")
    sh_f1 = _grab(block, r"SpecHead Macro-F1\s+([\d.]+)")
    fp_acc = _grab(block, r"FusionProto Acc \(语义\)\s+([\d.]+)")
    fp_f1 = _grab(block, r"FusionProto Macro-F1\s+([\d.]+)")
    ens_acc = _grab(block, r"集成\(最优\) Acc\s+([\d.]+)")
    ens_f1 = _grab(block, r"集成\(最优\) Macro-F1\s+([\d.]+)")
    sp_acc = _grab(block, r"SpecProto Acc \(语义\)\s+([\d.]+)")
    hmean = _grab(block, r"H-mean \(per-class recall\)\s+([\d.]+)")

    ens_rec = _macro_recall_from_perclass(block, "MultimodalEns")
    td_acc = _grab(block, r"Transductive Acc\s+([\d.]+)")
    td_f1 = _grab(block, r"Transductive Macro-F1\s+([\d.]+)")
    cands = [
        ("fusion_proto", fp_acc, fp_f1, _macro_recall_from_perclass(block, "FusionProto")),
        ("spec_head", sh_acc, sh_f1, _macro_recall_from_perclass(block, "SpecHead")),
        ("ensemble", ens_acc, ens_f1, ens_rec),
        ("transductive", td_acc, td_f1, float("nan")),
        ("spec_proto", sp_acc, float("nan"), float("nan")),
    ]
    valid = [c for c in cands if c[1] == c[1] and c[1] > 0]
    if not valid:
        return {
            "acc": top1, "macro_f1": float("nan"), "recall": hmean,
            "path": "unknown", "top1": top1,
        }
    win = max(valid, key=lambda x: (x[1], x[2] if x[2] == x[2] else -1.0))
    acc = top1 if top1 == top1 and top1 > 0 else win[1]
    if win[3] == win[3]:
        rec = win[3]
    elif win[0] in ("ensemble", "transductive", "spec_proto"):
        rec = float("nan")
    else:
        rec = hmean if hmean == hmean else float("nan")
    f1 = win[2] if win[2] == win[2] else float("nan")
    return {"acc": acc, "macro_f1": f1, "recall": rec, "path": win[0], "top1": top1}


def split_domain_blocks(log: str, domain: str) -> list[str]:
    markers = [
        f"【Phase 3 (跨域) — {domain} 综合性能汇总表】",
        f"【Phase 1/2 (域内跨工况) — {domain} 综合性能汇总表】",
    ]
    for marker in markers:
        parts = log.split(marker)
        if len(parts) > 1:
            return parts[1:]
    return []


def build_rows(log: str, domain: str, test_groups: list[str]) -> list[dict]:
    blocks = split_domain_blocks(log, domain)
    rows = []
    for tg, block in zip(test_groups, blocks):
        m = parse_phase3_block(block)
        rows.append({"test_group": tg, **m})
    return rows


def _train_groups_for_test(all_groups: list[str], test_group: str) -> list[str]:
    return [g for g in all_groups if g != test_group]


def format_table(domain: str, rows: list[dict], old_rows: list[dict], all_groups: list[str]) -> str:
    lines = [
        "=" * 96,
        f"【{domain} 域内跨工况训练结果 — 校准版 (best_path / Top-1 最终判断)】",
        "=" * 96,
        f"  {'训练工况':<40} | {'测试工况':<20} | {'Acc':>8} | {'Recall':>8} | {'macro_f1':>8} | {'路径':<12}",
        "-" * 96,
    ]
    for i, r in enumerate(rows):
        old = old_rows[i] if i < len(old_rows) else {}
        tr = ",".join(str(x) for x in old.get("train_groups", [])) or ",".join(
            _train_groups_for_test(all_groups, r["test_group"])
        )
        te = str(r["test_group"])
        _rv, _fv = r["recall"], r["macro_f1"]
        _rs = "     N/A" if (_rv != _rv) else f"{_rv:8.4f}"
        _fs = "     N/A" if (_fv != _fv) else f"{_fv:8.4f}"
        lines.append(
            f"  {tr[:40]:<40} | {te[:20]:<20} | "
            f"{r['acc']:8.4f} | {_rs} | {_fs} | {r['path']:<12}"
        )
    lines.append("=" * 96)
    return "\n".join(lines)


def parse_old_loo_section(log: str, domain: str) -> list[dict]:
    anchor = f"【{domain} 域内跨工况训练结果】"
    idx = log.rfind(anchor)
    if idx < 0:
        return []
    chunk = log[idx : idx + 8000]
    out = []
    in_table = False
    for line in chunk.splitlines():
        if "训练工况" in line and "测试工况" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.strip().startswith("="):
            break
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        tr, te = parts[0], parts[1]
        out.append({
            "train_groups": [x.strip() for x in tr.split(",") if x.strip()],
            "test_group": te,
            "acc_old": parts[2],
            "rec_old": parts[3],
            "f1_old": parts[4],
        })
    return out


def _mean_std_line(domain: str, rows: list[dict]) -> str:
    import statistics
    accs = [r["acc"] for r in rows if r["acc"] == r["acc"]]
    if not accs:
        return ""
    return f"  {domain}: {statistics.mean(accs):.4f} ± {statistics.pstdev(accs):.4f} (n={len(accs)})"


def patch_cwru_pu_section(log: str, cwru_rows, pu_rows, cwru_old, pu_old) -> str:
    """Replace CWRU/PU LOO tables and summary stats in-place."""
    anchor = "\n[LOO Summary] metric_source="
    idx_loo = log.rfind(anchor)
    idx_three = log.find("\n================================================================================\n【三阶段汇总", idx_loo)
    if idx_loo < 0 or idx_three < 0:
        return log
    head = log[:idx_loo]
    tail = log[idx_three:]
    body = (
        "\n[LOO Summary] metric_source=best_path (calibrated Top-1; see tables below)\n"
        + _mean_std_line("CWRU", cwru_rows) + "\n"
        + _mean_std_line("PU", pu_rows) + "\n"
        "--------------------------------------------------------------------------------\n\n"
        + format_table("CWRU", cwru_rows, cwru_old, CWRU_PHASES)
        + "\n\n"
        + format_table("PU", pu_rows, pu_old, PU_PHASES)
        + "\n\n"
        "【CWRU 域内跨工况训练结果 — 原始 spec_head 口径 (保留备查)】\n"
        "================================================================================================\n"
    )
    # preserve original CWRU+PU spec_head tables from old section (skip calibrated tables)
    old_sec = log[idx_loo:idx_three]
    for dom in ("CWRU", "PU"):
        a = old_sec.find(f"【{dom} 域内跨工况训练结果】")
        if a < 0:
            continue
        if "校准版" in old_sec[a : a + 80]:
            a2 = old_sec.find(f"【{dom} 域内跨工况训练结果】", a + 5)
            if a2 > a:
                a = a2
        b = old_sec.find("\n================================================================================================", a + 20)
        if b < 0:
            b = old_sec.find("\n================================================================================", a + 20)
        if b > a and "|" in old_sec[a:b]:
            body += old_sec[a:b].strip() + "\n================================================================================================\n\n"
    return head + body + tail


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "logs/training_20260518_031316.log"
    )
    log = log_path.read_text(encoding="utf-8", errors="replace")

    domain_plan = [
        ("CWRU", CWRU_PHASES),
        ("PU", PU_PHASES),
        ("HUST", HUST_PHASES),
        ("SEU", SEU_PHASES),
    ]
    all_rows = {}
    all_old = {}
    for dom, phases in domain_plan:
        blocks = split_domain_blocks(log, dom)
        if len(blocks) >= len(phases):
            all_rows[dom] = build_rows(log, dom, phases)
            all_old[dom] = parse_old_loo_section(log, dom)
        else:
            all_rows[dom] = []
            all_old[dom] = []

    out_path = log_path.with_name(log_path.stem + "_loo_calibrated.txt")
    chunks = [
        f"Source log: {log_path}",
        "Metric: best_path (Phase summary Top-1 Acc + macro-F1/recall from winning path)",
        "",
        "--- mean±std (best_path Acc) ---",
    ]
    for dom, _ in domain_plan:
        line = _mean_std_line(dom, all_rows.get(dom, []))
        if line:
            chunks.append(line)
    chunks.append("")
    for dom, phases in domain_plan:
        rows = all_rows.get(dom, [])
        if rows:
            chunks.append(format_table(dom, rows, all_old.get(dom, []), phases))
            chunks.append("")

    text = "\n".join(chunks) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote: {out_path}")

    patched = log
    if all_rows.get("CWRU") and all_rows.get("PU"):
        patched = patch_cwru_pu_section(
            patched,
            all_rows["CWRU"],
            all_rows["PU"],
            all_old.get("CWRU", []),
            all_old.get("PU", []),
        )
        log_path.write_text(patched, encoding="utf-8")
        print(f"Updated log in-place: {log_path}")

    # HUST/SEU patch (legacy layout)
    marker_start = "\n" + "=" * 96 + "\n【HUST 域内跨工况训练结果】"
    marker_end = "\n" + "=" * 80 + "\n【三阶段汇总"
    idx0 = patched.find(marker_start)
    idx1 = patched.find(marker_end)
    if idx0 >= 0 and idx1 > idx0 and all_rows.get("HUST"):
        new_mid = (
            "\n\n"
            + format_table("HUST", all_rows["HUST"], all_old.get("HUST", []), HUST_PHASES)
            + "\n\n"
            + format_table("SEU", all_rows.get("SEU", []), all_old.get("SEU", []), SEU_PHASES)
            + "\n\n"
            + "-" * 80 + "\n"
            + "[LOO Summary] metric_source=best_path (calibrated from Phase Top-1)\n"
        )
        for dom in ("HUST", "SEU"):
            line = _mean_std_line(dom, all_rows.get(dom, []))
            if line:
                new_mid += line + "\n"
        new_mid += "-" * 80 + "\n\n"
        new_mid += "【HUST 域内跨工况训练结果 — 原始 spec_head 口径 (保留备查)】\n" + patched[idx0:idx1]
        patched2 = patched[:idx0] + new_mid + patched[idx1:]
        cal_log = log_path.with_name(log_path.stem + "_loo_patched.log")
        cal_log.write_text(patched2, encoding="utf-8")
        print(f"Patched log (HUST/SEU): {cal_log}")


if __name__ == "__main__":
    main()
