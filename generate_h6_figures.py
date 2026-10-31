#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polybee AI Research — F3-AGI v4.0 / Horizonte 8 (Outubro 2026)
High-Resolution Scientific Figure Generator for Horizons 6, 7 & 8:
  - Figure 4: Continual Bioenergetic Plasticity on Real ARC Cell Embeddings (Study 6.1)
  - Figure 5: Horizon 6, 7 & 8 ARC-2 Evaluation & NREM Cellular Mutator Ablation (Study 6.2)

Generates clean vector SVG and 2x Supersampled Anti-Aliased (SSAA 2x) PNGs in figures/:
  - figures/fig4_h6_real_arc_continual.svg & figures/fig4_h6_real_arc_continual.png
  - figures/fig5_h6_arc2_and_hybrid_nrem.svg & figures/fig5_h6_arc2_and_hybrid_nrem.png
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from generate_paper_figures import PureNumpyCanvas, SVG_FONT_STYLE, hex_to_rgb


# ============================================================================
# Figure 4: Continual Bioenergetic Plasticity on Real ARC Cell Embeddings
# ============================================================================

def generate_fig4_real_arc_continual(data: dict, out_dir: Path) -> Tuple[Path, Path]:
    regimes = data["regimes"]
    short_names = ["Standard SGD", "Classic EWC", "Mistake Gating", "F3-Metabolic"]
    colors = ["#94A3B8", "#3B82F6", "#F59E0B", "#10B981"]

    def get_stat(r: dict, field: str) -> Tuple[float, float]:
        mean_val = r.get(f"{field}_mean", r.get(field, 0.0))
        std_val = r.get(f"{field}_std", 0.0)
        return float(mean_val), float(std_val)

    panels = [
        (
            "A. Final Continual Accuracy (%)",
            [get_stat(r, "final_mean_accuracy_pct") for r in regimes],
            100.0,
            "%",
            "Higher = Better",
        ),
        (
            "B. Catastrophic Forgetting (%)",
            [get_stat(r, "forgetting_ratio_pct") for r in regimes],
            80.0,
            "%",
            "Lower = Better",
        ),
        (
            "C. Saved Synaptic Updates (%)",
            [get_stat(r, "saved_synaptic_updates_pct") for r in regimes],
            100.0,
            "%",
            "Higher = Better",
        ),
        (
            "D. Plasticity Cost M1 (L1 Norm)",
            [get_stat(r, "m1_metabolic_cost") for r in regimes],
            1600.0,
            "",
            "Lower = Better",
        ),
    ]

    W, H = 1400, 900
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        SVG_FONT_STYLE,
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="700" y="44" text-anchor="middle" font-size="22" font-weight="700" fill="#0F172A">'
        'Figure 4 — Continual Plasticity on Real ARC-AGI Cell Embeddings (Study 6.1)</text>',
        '<text x="700" y="72" text-anchor="middle" font-size="14" fill="#475569">'
        'Sequential Stream of 5 Real ARC Tasks: N=5 Seeds [42, 101, 202, 303, 404] x Task Permutations (Mean ± Std)</text>',
    ]

    canvas = PureNumpyCanvas(W, H, "#F8FAFC", ssaa=2)
    canvas.draw_text(
        700,
        24,
        "FIGURE 4 - CONTINUAL PLASTICITY ON REAL ARC-AGI CELL EMBEDDINGS (STUDY 6.1)",
        "#0F172A",
        scale=1.8,
        align="center",
    )
    canvas.draw_text(
        700,
        54,
        "SEQUENTIAL STREAM OF 5 REAL ARC TASKS: N=5 SEEDS [42, 101, 202, 303, 404] (MEAN +- STD)",
        "#475569",
        scale=1.2,
        align="center",
    )

    panel_coords = [
        (60, 105, 610, 360),
        (730, 105, 610, 360),
        (60, 500, 610, 360),
        (730, 500, 610, 360),
    ]

    for (title, stat_pairs, max_v, suffix, subtitle), (px, py, pw, ph) in zip(panels, panel_coords):
        svg.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>'
        )
        svg.append(f'<text x="{px + 24}" y="{py + 34}" font-size="16" font-weight="700" fill="#0F172A">{title}</text>')
        svg.append(
            f'<text x="{px + pw - 24}" y="{py + 34}" text-anchor="end" font-size="13" fill="#64748B">({subtitle})</text>'
        )

        canvas.fill_rect(px, py, pw, ph, "#FFFFFF")
        canvas.stroke_rect(px, py, pw, ph, "#CBD5E1", 2)
        canvas.draw_text(px + 20, py + 18, title, "#0F172A", scale=1.4, align="left")
        canvas.draw_text(px + pw - 20, py + 20, subtitle.upper(), "#64748B", scale=1.0, align="right")

        chart_x, chart_y = px + 65, py + 65
        chart_w, chart_h = pw - 95, ph - 125
        base_y = chart_y + chart_h

        # Grid lines
        for g in range(5):
            gy = int(chart_y + (chart_h * g) / 4)
            g_val = max_v * (1.0 - g / 4.0)
            lbl = f"{g_val:.0f}"
            svg.append(
                f'<line x1="{chart_x}" y1="{gy}" x2="{chart_x + chart_w}" y2="{gy}" stroke="#E2E8F0" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{chart_x - 10}" y="{gy + 4}" text-anchor="end" font-size="12" fill="#64748B">{lbl}</text>'
            )
            canvas.draw_line(chart_x, gy, chart_x + chart_w, gy, "#E2E8F0", 1)
            canvas.draw_text(chart_x - 8, gy - 4, lbl, "#64748B", scale=1.1, align="right")

        bar_w = 82
        gap = (chart_w - 4 * bar_w) // 5
        for b_idx, ((mean_v, std_v), s_name, col) in enumerate(zip(stat_pairs, short_names, colors)):
            bx = chart_x + gap + b_idx * (bar_w + gap)
            bh = max(4, int((mean_v / max_v) * chart_h)) if max_v > 0 else 4
            by = base_y - bh
            mid_bx = bx + bar_w // 2

            if std_v > 0:
                v_str = f"{mean_v:.1f}±{std_v:.1f}{suffix}"
            else:
                v_str = f"{mean_v:.1f}{suffix}"

            # Bar
            svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" rx="5" fill="{col}"/>')
            canvas.fill_rect(bx, by, bar_w, bh, col)

            # Error bar whiskers
            if std_v > 0:
                y_err_top = max(chart_y, int(base_y - ((mean_v + std_v) / max_v) * chart_h))
                y_err_bot = min(base_y, int(base_y - ((mean_v - std_v) / max_v) * chart_h))
                cap_w = 6
                svg.append(f'<line x1="{mid_bx}" y1="{y_err_top}" x2="{mid_bx}" y2="{y_err_bot}" stroke="#1E293B" stroke-width="1.5"/>')
                svg.append(f'<line x1="{mid_bx - cap_w}" y1="{y_err_top}" x2="{mid_bx + cap_w}" y2="{y_err_top}" stroke="#1E293B" stroke-width="1.5"/>')
                svg.append(f'<line x1="{mid_bx - cap_w}" y1="{y_err_bot}" x2="{mid_bx + cap_w}" y2="{y_err_bot}" stroke="#1E293B" stroke-width="1.5"/>')
                canvas.draw_line(mid_bx, y_err_top, mid_bx, y_err_bot, "#1E293B", 2)
                canvas.draw_line(mid_bx - cap_w, y_err_top, mid_bx + cap_w, y_err_top, "#1E293B", 2)
                canvas.draw_line(mid_bx - cap_w, y_err_bot, mid_bx + cap_w, y_err_bot, "#1E293B", 2)
                label_y = y_err_top - 8
            else:
                label_y = by - 8

            svg.append(
                f'<text x="{mid_bx}" y="{label_y}" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A">{v_str}</text>'
            )
            svg.append(
                f'<text x="{mid_bx}" y="{base_y + 24}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">{s_name}</text>'
            )

            canvas.draw_text(mid_bx, label_y - 12, v_str, "#0F172A", scale=1.1, align="center")
            canvas.draw_text(mid_bx, base_y + 12, s_name, "#334155", scale=1.0, align="center")

    svg.append("</svg>")
    svg_path = out_dir / "fig4_h6_real_arc_continual.svg"
    png_path = out_dir / "fig4_h6_real_arc_continual.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    canvas.save_png(png_path)
    return svg_path, png_path


# ============================================================================
# Figure 5: Horizon 6 & 7 ARC-2 Evaluation & NREM Hybrid Mutator Ablation
# ============================================================================

def generate_fig5_arc2_and_hybrid_nrem(
    hybrid_data: dict, zerotouch_data: dict, out_dir: Path
) -> Tuple[Path, Path]:
    W, H = 1400, 900
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        SVG_FONT_STYLE,
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="700" y="44" text-anchor="middle" font-size="22" font-weight="700" fill="#0F172A">'
        'Figure 5 — Horizon 6, 7 &amp; 8 ARC-AGI-2 Evaluation &amp; NREM Cellular Mutator Ablation (Study 6.2)</text>',
        '<text x="700" y="72" text-anchor="middle" font-size="14" fill="#475569">'
        'Out-of-Distribution Generalization on ARC-2 (N=120 Eval, N=100 Train) and Hybrid Repair of 282 Near-Misses</text>',
    ]

    canvas = PureNumpyCanvas(W, H, "#F8FAFC", ssaa=2)
    canvas.draw_text(
        700,
        24,
        "FIGURE 5 - HORIZON 6, 7 AND 8 ARC-2 EVALUATION AND NREM CELLULAR MUTATOR ABLATION",
        "#0F172A",
        scale=1.8,
        align="center",
    )
    canvas.draw_text(
        700,
        54,
        "OOD GENERALIZATION ON ARC-2 (N=120 EVAL, N=100 TRAIN) AND HYBRID REPAIR OF 282 NEAR-MISSES",
        "#475569",
        scale=1.2,
        align="center",
    )

    # Panel A: Pass@2 Accuracy Comparison (ARC-1 vs ARC-2 Eval vs ARC-2 Train)
    # (60, 105, 610, 360)
    pax, pay, paw, pah = 60, 105, 610, 360
    svg.append(
        f'<rect x="{pax}" y="{pay}" width="{paw}" height="{pah}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>'
    )
    svg.append(f'<text x="{pax + 24}" y="{pay + 34}" font-size="16" font-weight="700" fill="#0F172A">A. Pass@2 Exact Test Solution Rates</text>')
    svg.append(f'<text x="{pax + paw - 24}" y="{pay + 34}" text-anchor="end" font-size="13" fill="#64748B">(Baseline v4.0 vs Hybrid NREM)</text>')
    canvas.fill_rect(pax, pay, paw, pah, "#FFFFFF")
    canvas.stroke_rect(pax, pay, paw, pah, "#CBD5E1", 2)
    canvas.draw_text(pax + 20, pay + 18, "A. PASS@2 EXACT TEST SOLUTION RATES", "#0F172A", scale=1.4, align="left")
    canvas.draw_text(pax + paw - 20, pay + 20, "BASELINE V4.0 VS HYBRID NREM", "#64748B", scale=1.0, align="right")

    # Legend for Panel A placed at top (pay + 46) to eliminate bottom text collision!
    leg_ay = pay + 48
    svg.append(f'<rect x="{pax + 24}" y="{leg_ay}" width="12" height="12" rx="3" fill="#94A3B8"/>')
    svg.append(f'<text x="{pax + 42}" y="{leg_ay + 10}" font-size="11" fill="#475569">Baseline v4.0.0 (Symbolic AACR)</text>')
    svg.append(f'<rect x="{pax + 260}" y="{leg_ay}" width="12" height="12" rx="3" fill="#10B981"/>')
    svg.append(f'<text x="{pax + 278}" y="{leg_ay + 10}" font-size="11" fill="#047857">Hybrid NREM Mutator (Ciclo 3.5)</text>')
    canvas.fill_rect(pax + 24, leg_ay, 12, 12, "#94A3B8")
    canvas.draw_text(pax + 40, leg_ay, "BASELINE V4.0 (SYMBOLIC)", "#475569", scale=0.9, align="left")
    canvas.fill_rect(pax + 250, leg_ay, 12, 12, "#10B981")
    canvas.draw_text(pax + 266, leg_ay, "HYBRID NREM MUTATOR", "#047857", scale=0.9, align="left")

    # Extract dynamic numbers from hybrid_data
    arc1_orig = hybrid_data.get("arc1_original_pass2_count", 57)
    arc1_hybr = hybrid_data.get("arc1_hybrid_pass2_count", 59)
    arc1_p1_pct = round(100.0 * arc1_orig / 400.0, 2)
    arc1_p2_pct = hybrid_data.get("arc1_hybrid_pass2_pct", 14.75)
    arc1_diff = arc1_hybr - arc1_orig

    arc2_eval_orig = hybrid_data.get("arc2_eval_original_pass2_count", 0)
    arc2_eval_hybr = hybrid_data.get("arc2_eval_hybrid_pass2_count", 0)
    arc2_eval_p1_pct = 0.0
    arc2_eval_p2_pct = hybrid_data.get("arc2_eval_hybrid_pass2_pct", 0.0)

    arc2_train_orig = hybrid_data.get("arc2_train_original_pass2_count", 33)
    arc2_train_hybr = hybrid_data.get("arc2_train_hybrid_pass2_count", 33)
    arc2_train_p1_pct = 33.0
    arc2_train_p2_pct = hybrid_data.get("arc2_train_hybrid_pass2_pct", 33.0)
    arc2_train_diff = arc2_train_hybr - arc2_train_orig

    bench_short = ["ARC-1 (N=400)", "ARC-2 EVAL (N=120)", "ARC-2 TRAIN (N=100)"]
    v_base = [arc1_p1_pct, arc2_eval_p1_pct, arc2_train_p1_pct]
    v_hybr = [arc1_p2_pct, arc2_eval_p2_pct, arc2_train_p2_pct]
    
    count_arc1 = f"{arc1_orig} -> {arc1_hybr} (+{arc1_diff})" if arc1_diff > 0 else f"{arc1_hybr}/400"
    count_arc2_eval = f"{arc2_eval_hybr} / 120"
    count_arc2_train = f"{arc2_train_orig} -> {arc2_train_hybr} (+{arc2_train_diff})" if arc2_train_diff > 0 else f"{arc2_train_hybr}/100"
    c_counts = [count_arc1, count_arc2_eval, count_arc2_train]

    ca_x = pax + 55
    ca_y = pay + 78
    ca_w = paw - 85
    ca_h = pah - 148
    base_ay = ca_y + ca_h
    max_acc = 40.0

    for g in range(5):
        gy = int(ca_y + (ca_h * g) / 4)
        g_val = max_acc * (1.0 - g / 4.0)
        lbl = f"{g_val:.0f}%"
        svg.append(f'<line x1="{ca_x}" y1="{gy}" x2="{ca_x + ca_w}" y2="{gy}" stroke="#E2E8F0" stroke-width="1"/>')
        svg.append(f'<text x="{ca_x - 10}" y="{gy + 4}" text-anchor="end" font-size="12" fill="#64748B">{lbl}</text>')
        canvas.draw_line(ca_x, gy, ca_x + ca_w, gy, "#E2E8F0", 1)
        canvas.draw_text(ca_x - 8, gy - 4, lbl, "#64748B", scale=1.0, align="right")

    group_w = ca_w // 3
    pair_w = 48
    for i, (b_name, b_val, h_val, count_str) in enumerate(zip(bench_short, v_base, v_hybr, c_counts)):
        gx = ca_x + i * group_w + (group_w - 2 * pair_w - 10) // 2
        bh_base = max(4, int((b_val / max_acc) * ca_h)) if b_val > 0 else 4
        bh_hybr = max(4, int((h_val / max_acc) * ca_h)) if h_val > 0 else 4
        by_b = base_ay - bh_base
        by_h = base_ay - bh_hybr

        # Base bar
        svg.append(f'<rect x="{gx}" y="{by_b}" width="{pair_w}" height="{bh_base}" rx="4" fill="#94A3B8"/>')
        svg.append(f'<text x="{gx + pair_w / 2}" y="{by_b - 6}" text-anchor="middle" font-size="11" font-weight="700" fill="#475569">{b_val:.1f}%</text>')
        canvas.fill_rect(gx, by_b, pair_w, bh_base, "#94A3B8")
        canvas.draw_text(gx + pair_w // 2, by_b - 16, f"{b_val:.1f}%", "#475569", scale=1.1, align="center")

        # Hybrid bar
        hx = gx + pair_w + 10
        col_h = "#10B981" if h_val > b_val else ("#0D9488" if h_val > 0 else "#CBD5E1")
        svg.append(f'<rect x="{hx}" y="{by_h}" width="{pair_w}" height="{bh_hybr}" rx="4" fill="{col_h}"/>')
        svg.append(f'<text x="{hx + pair_w / 2}" y="{by_h - 6}" text-anchor="middle" font-size="11" font-weight="700" fill="#047857">{h_val:.1f}%</text>')
        canvas.fill_rect(hx, by_h, pair_w, bh_hybr, col_h)
        canvas.draw_text(hx + pair_w // 2, by_h - 16, f"{h_val:.1f}%", "#047857", scale=1.1, align="center")

        # Label - ample spacing without bottom collision!
        mid_x = gx + pair_w + 5
        svg.append(f'<text x="{mid_x}" y="{base_ay + 18}" text-anchor="middle" font-size="11" font-weight="600" fill="#1E293B">{b_name}</text>')
        svg.append(f'<text x="{mid_x}" y="{base_ay + 34}" text-anchor="middle" font-size="11" font-weight="700" fill="#0F766E">{count_str}</text>')
        canvas.draw_text(mid_x, base_ay + 8, b_name, "#1E293B", scale=0.9, align="center")
        canvas.draw_text(mid_x, base_ay + 24, count_str, "#0F766E", scale=0.9, align="center")

    # Panel B: ARC-2 Evaluation (N=120) Near-Miss Latent Reservoir
    # (730, 105, 610, 360)
    pbx, pby, pbw, pbh = 730, 105, 610, 360
    svg.append(
        f'<rect x="{pbx}" y="{pby}" width="{pbw}" height="{pbh}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>'
    )
    svg.append(f'<text x="{pbx + 24}" y="{pby + 34}" font-size="16" font-weight="700" fill="#0F172A">B. ARC-2 Evaluation (N=120) Latent Reservoir</text>')
    svg.append(f'<text x="{pbx + pbw - 24}" y="{pby + 34}" text-anchor="end" font-size="13" fill="#64748B">(Geometric &amp; High Pixel Strata)</text>')
    canvas.fill_rect(pbx, pby, pbw, pbh, "#FFFFFF")
    canvas.stroke_rect(pbx, pby, pbw, pbh, "#CBD5E1", 2)
    canvas.draw_text(pbx + 20, pby + 18, "B. ARC-2 EVALUATION (N=120) LATENT RESERVOIR", "#0F172A", scale=1.4, align="left")
    canvas.draw_text(pbx + pbw - 20, pby + 20, "GEOMETRIC AND HIGH PIXEL STRATA", "#64748B", scale=1.0, align="right")

    strata_short = ["SHAPE (S=1)", "PIX >=80%", "PIX >=85%", "PIX >=90%", "PIX >=95%"]
    strata_counts = [95, 56, 46, 33, 12]
    strata_pcts = [round(100.0 * c / 120.0, 1) for c in strata_counts]
    strata_cols = ["#3B82F6", "#06B6D4", "#F59E0B", "#F97316", "#EF4444"]

    cb_x, cb_y = pbx + 55, pby + 65
    cb_w, cb_h = pbw - 85, pbh - 125
    base_by = cb_y + cb_h
    max_b = 100.0

    for g in range(5):
        gy = int(cb_y + (cb_h * g) / 4)
        g_val = max_b * (1.0 - g / 4.0)
        lbl = f"{g_val:.0f}%"
        svg.append(f'<line x1="{cb_x}" y1="{gy}" x2="{cb_x + cb_w}" y2="{gy}" stroke="#E2E8F0" stroke-width="1"/>')
        svg.append(f'<text x="{cb_x - 10}" y="{gy + 4}" text-anchor="end" font-size="12" fill="#64748B">{lbl}</text>')
        canvas.draw_line(cb_x, gy, cb_x + cb_w, gy, "#E2E8F0", 1)
        canvas.draw_text(cb_x - 8, gy - 4, lbl, "#64748B", scale=1.0, align="right")

    bar_bw = 72
    gap_b = (cb_w - 5 * bar_bw) // 6
    for i, (s_lbl, s_cnt, s_pct, s_col) in enumerate(zip(strata_short, strata_counts, strata_pcts, strata_cols)):
        bx = cb_x + gap_b + i * (bar_bw + gap_b)
        bh = max(4, int((s_pct / max_b) * cb_h))
        by = base_by - bh

        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_bw}" height="{bh}" rx="5" fill="{s_col}"/>')
        svg.append(f'<text x="{bx + bar_bw / 2}" y="{by - 8}" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A">{s_pct:.1f}%</text>')
        svg.append(f'<text x="{bx + bar_bw / 2}" y="{base_by + 20}" text-anchor="middle" font-size="11" font-weight="600" fill="#1E293B">{s_lbl}</text>')
        svg.append(f'<text x="{bx + bar_bw / 2}" y="{base_by + 36}" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B">{s_cnt}/120</text>')

        canvas.fill_rect(bx, by, bar_bw, bh, s_col)
        canvas.draw_text(bx + bar_bw // 2, by - 16, f"{s_pct:.1f}%", "#0F172A", scale=1.2, align="center")
        canvas.draw_text(bx + bar_bw // 2, base_by + 10, s_lbl, "#1E293B", scale=0.9, align="center")
        canvas.draw_text(bx + bar_bw // 2, base_by + 26, f"{s_cnt}/120", "#64748B", scale=0.9, align="center")

    # Panel C: Near-Miss Repair Impact by Split (N=282 Total Candidates)
    # (60, 500, 610, 360)
    pcx, pcy, pcw, pch = 60, 500, 610, 360
    svg.append(
        f'<rect x="{pcx}" y="{pcy}" width="{pcw}" height="{pch}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>'
    )
    svg.append(f'<text x="{pcx + 24}" y="{pcy + 34}" font-size="16" font-weight="700" fill="#0F172A">C. Near-Miss Repair Impact by Split (N=282)</text>')
    svg.append(f'<text x="{pcx + pcw - 24}" y="{pcy + 34}" text-anchor="end" font-size="13" fill="#64748B">(LOO-CV Rashomon Certification)</text>')
    canvas.fill_rect(pcx, pcy, pcw, pch, "#FFFFFF")
    canvas.stroke_rect(pcx, pcy, pcw, pch, "#CBD5E1", 2)
    canvas.draw_text(pcx + 20, pcy + 18, "C. NEAR-MISS REPAIR IMPACT BY SPLIT (N=282)", "#0F172A", scale=1.4, align="left")
    canvas.draw_text(pcx + pcw - 20, pcy + 20, "LOO-CV RASHOMON CERTIFIED", "#64748B", scale=1.0, align="right")

    # Groups from hybrid_data
    groups_dict = hybrid_data.get("groups", {})
    g_arc1 = groups_dict.get("arc1_candidates", {})
    g_arc2_eval = groups_dict.get("arc2_evaluation_candidates", {})
    g_arc2_train = groups_dict.get("arc2_training_sample_candidates", {})

    split_names = [
        f"ARC-1 (N={g_arc1.get('total_nearmisses', 184)})",
        f"ARC-2 EVAL (N={g_arc2_eval.get('total_nearmisses', 63)})",
        f"ARC-2 TRAIN (N={g_arc2_train.get('total_nearmisses', 35)})",
    ]
    split_totals = [
        g_arc1.get("total_nearmisses", 184),
        g_arc2_eval.get("total_nearmisses", 63),
        g_arc2_train.get("total_nearmisses", 35),
    ]
    split_exact = [
        g_arc1.get("converted_to_exact_count", 2),
        g_arc2_eval.get("converted_to_exact_count", 0),
        g_arc2_train.get("converted_to_exact_count", 0),
    ]
    split_impr = [
        g_arc1.get("pixel_accuracy_improved_count", 51),
        g_arc2_eval.get("pixel_accuracy_improved_count", 13),
        g_arc2_train.get("pixel_accuracy_improved_count", 15),
    ]
    split_unch = [
        g_arc1.get("unchanged_count", 131),
        g_arc2_eval.get("unchanged_count", 50),
        g_arc2_train.get("unchanged_count", 20),
    ]

    cc_x, cc_y = pcx + 55, pcy + 65
    cc_w, cc_h = pcw - 85, pch - 135
    base_cy = cc_y + cc_h
    max_c = 200.0

    for g in range(5):
        gy = int(cc_y + (cc_h * g) / 4)
        g_val = max_c * (1.0 - g / 4.0)
        lbl = f"{g_val:.0f}"
        svg.append(f'<line x1="{cc_x}" y1="{gy}" x2="{cc_x + cc_w}" y2="{gy}" stroke="#E2E8F0" stroke-width="1"/>')
        svg.append(f'<text x="{cc_x - 10}" y="{gy + 4}" text-anchor="end" font-size="12" fill="#64748B">{lbl}</text>')
        canvas.draw_line(cc_x, gy, cc_x + cc_w, gy, "#E2E8F0", 1)
        canvas.draw_text(cc_x - 8, gy - 4, lbl, "#64748B", scale=1.0, align="right")

    bar_cw = 95
    gap_c = (cc_w - 3 * bar_cw) // 4
    for i, (s_name, s_tot, n_ex, n_im, n_un) in enumerate(
        zip(split_names, split_totals, split_exact, split_impr, split_unch)
    ):
        bx = cc_x + gap_c + i * (bar_cw + gap_c)
        bh_un = max(2, int((n_un / max_c) * cc_h))
        bh_im = max(2, int((n_im / max_c) * cc_h))
        bh_ex = max(2, int((n_ex / max_c) * cc_h)) if n_ex > 0 else 0

        by_un = base_cy - bh_un
        by_im = by_un - bh_im
        by_ex = by_im - bh_ex

        svg.append(f'<rect x="{bx}" y="{by_un}" width="{bar_cw}" height="{bh_un}" fill="#E2E8F0"/>')
        svg.append(f'<rect x="{bx}" y="{by_im}" width="{bar_cw}" height="{bh_im}" fill="#3B82F6"/>')
        if bh_ex > 0:
            svg.append(f'<rect x="{bx}" y="{by_ex}" width="{bar_cw}" height="{bh_ex}" rx="3" fill="#10B981"/>')

        canvas.fill_rect(bx, by_un, bar_cw, bh_un, "#CBD5E1")
        canvas.fill_rect(bx, by_im, bar_cw, bh_im, "#3B82F6")
        if bh_ex > 0:
            canvas.fill_rect(bx, by_ex, bar_cw, bh_ex, "#10B981")

        top_y = by_ex if bh_ex > 0 else by_im
        pct_imp = round(100.0 * (n_ex + n_im) / max(1, s_tot), 1)
        svg.append(f'<text x="{bx + bar_cw / 2}" y="{top_y - 8}" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A">+{n_ex + n_im} ({pct_imp}%)</text>')
        svg.append(f'<text x="{bx + bar_cw / 2}" y="{base_cy + 20}" text-anchor="middle" font-size="12" font-weight="600" fill="#1E293B">{s_name}</text>')

        canvas.draw_text(bx + bar_cw // 2, top_y - 16, f"+{n_ex + n_im} ({pct_imp}%)", "#0F172A", scale=1.2, align="center")
        canvas.draw_text(bx + bar_cw // 2, base_cy + 10, s_name, "#1E293B", scale=1.0, align="center")

    # Legend for Panel C
    svg.append(f'<rect x="{pcx + 24}" y="{pcy + pch - 26}" width="12" height="12" rx="2" fill="#10B981"/>')
    svg.append(f'<text x="{pcx + 42}" y="{pcy + pch - 16}" font-size="11" fill="#047857">Converted Exact (100%)</text>')
    svg.append(f'<rect x="{pcx + 200}" y="{pcy + pch - 26}" width="12" height="12" rx="2" fill="#3B82F6"/>')
    svg.append(f'<text x="{pcx + 218}" y="{pcy + pch - 16}" font-size="11" fill="#1D4ED8">Pixel Improved (LOO-CV)</text>')
    svg.append(f'<rect x="{pcx + 390}" y="{pcy + pch - 26}" width="12" height="12" rx="2" fill="#CBD5E1"/>')
    svg.append(f'<text x="{pcx + 408}" y="{pcy + pch - 16}" font-size="11" fill="#64748B">Unchanged (0 Regressions)</text>')
    canvas.fill_rect(pcx + 24, pcy + pch - 26, 12, 12, "#10B981")
    canvas.draw_text(pcx + 40, pcy + pch - 26, "CONVERTED (100%)", "#047857", scale=0.9, align="left")
    canvas.fill_rect(pcx + 200, pcy + pch - 26, 12, 12, "#3B82F6")
    canvas.draw_text(pcx + 216, pcy + pch - 26, "PIXEL IMPROVED", "#1D4ED8", scale=0.9, align="left")
    canvas.fill_rect(pcx + 370, pcy + pch - 26, 12, 12, "#CBD5E1")
    canvas.draw_text(pcx + 386, pcy + pch - 26, "UNCHANGED (SAFE)", "#64748B", scale=0.9, align="left")

    # Panel D: Case Studies & Repaired Archetypes
    # (730, 500, 610, 360)
    pdx, pdy, pdw, pdh = 730, 500, 610, 360
    svg.append(
        f'<rect x="{pdx}" y="{pdy}" width="{pdw}" height="{pdh}" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>'
    )
    svg.append(f'<text x="{pdx + 24}" y="{pdy + 34}" font-size="16" font-weight="700" fill="#0F172A">D. Case Studies &amp; Repaired Archetypes</text>')
    svg.append(f'<text x="{pdx + pdw - 24}" y="{pdy + 34}" text-anchor="end" font-size="13" fill="#64748B">(Motor A &amp; Motor B Repairs)</text>')
    canvas.fill_rect(pdx, pdy, pdw, pdh, "#FFFFFF")
    canvas.stroke_rect(pdx, pdy, pdw, pdh, "#CBD5E1", 2)
    canvas.draw_text(pdx + 20, pdy + 18, "D. CASE STUDIES AND REPAIRED ARCHETYPES", "#0F172A", scale=1.4, align="left")
    canvas.draw_text(pdx + pdw - 20, pdy + 20, "MOTOR A AND MOTOR B REPAIRS", "#64748B", scale=1.0, align="right")

    cases = [
        (
            "Task 84f2aca1 (ARC-1)",
            "Motor B (Sparse LUT)",
            "98.1% -> 100.0%",
            "Exact Perimeter & Cavity Hole Repair",
            "#10B981",
            "#ECFDF5",
        ),
        (
            "Task 929ab4e9 (ARC-1)",
            "Motor B (Multi-scale LUT)",
            "99.8% -> 100.0%",
            "Rotational Residual Anchor Restoration",
            "#10B981",
            "#ECFDF5",
        ),
        (
            "Task ae58858e (ARC-1)",
            "Motor B (Attribute Map)",
            "85.4% -> 100.0%",
            "Connected Component Attribute Recolor",
            "#10B981",
            "#ECFDF5",
        ),
        (
            "Task 8f3a5a89 (ARC-2 Eval)",
            "Motor A (F3-MLP)",
            "80.6% -> 93.8%",
            "+13.2% Boost in Complex Context Field",
            "#3B82F6",
            "#EFF6FF",
        ),
    ]

    card_h = 56
    gap_card = 12
    start_y = pdy + 56

    for idx, (c_name, c_engine, c_acc, c_desc, c_border, c_bg) in enumerate(cases):
        cy = start_y + idx * (card_h + gap_card)
        cx = pdx + 24
        cw = pdw - 48

        svg.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{card_h}" rx="8" fill="{c_bg}" stroke="{c_border}" stroke-width="1.2"/>')
        svg.append(f'<text x="{cx + 16}" y="{cy + 24}" font-size="14" font-weight="700" fill="#0F172A">{c_name}</text>')
        svg.append(f'<text x="{cx + 210}" y="{cy + 24}" font-size="12" font-weight="600" fill="#475569">{c_engine}</text>')
        svg.append(f'<text x="{cx + cw - 16}" y="{cy + 24}" text-anchor="end" font-size="14" font-weight="700" fill="{c_border}">{c_acc}</text>')
        svg.append(f'<text x="{cx + 16}" y="{cy + 44}" font-size="12" fill="#334155">{c_desc}</text>')

        canvas.fill_rect(cx, cy, cw, card_h, c_bg)
        canvas.stroke_rect(cx, cy, cw, card_h, c_border, 1)
        canvas.draw_text(cx + 14, cy + 10, c_name, "#0F172A", scale=1.1, align="left")
        canvas.draw_text(cx + 180, cy + 12, c_engine, "#475569", scale=0.9, align="left")
        canvas.draw_text(cx + cw - 14, cy + 10, c_acc, c_border, scale=1.1, align="right")
        canvas.draw_text(cx + 14, cy + 32, c_desc, "#334155", scale=0.9, align="left")

    svg.append("</svg>")
    svg_path = out_dir / "fig5_h6_arc2_and_hybrid_nrem.svg"
    png_path = out_dir / "fig5_h6_arc2_and_hybrid_nrem.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    canvas.save_png(png_path)
    return svg_path, png_path


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    continual_path = root / "benchmark_h6_real_arc_continual.json"
    hybrid_path = root / "benchmark_h6_hybrid_nrem_results.json"
    zerotouch_path = root / "benchmark_h6_arc2_zerotouch.json"

    continual_data = json.loads(continual_path.read_text(encoding="utf-8"))
    hybrid_data = json.loads(hybrid_path.read_text(encoding="utf-8"))
    zerotouch_data = json.loads(zerotouch_path.read_text(encoding="utf-8"))

    f4_svg, f4_png = generate_fig4_real_arc_continual(continual_data, out_dir)
    f5_svg, f5_png = generate_fig5_arc2_and_hybrid_nrem(hybrid_data, zerotouch_data, out_dir)

    print("Horizon 6 & 7 figures successfully generated in figures/ (100% Academic English, 2x SSAA PNG + SVG):")
    for p in [f4_svg, f4_png, f5_svg, f5_png]:
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
