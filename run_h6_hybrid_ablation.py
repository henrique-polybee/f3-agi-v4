#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polybee AI Research — F3-AGI v4.0 / Horizonte 7 (Outubro 2026)
Suíte Experimental de Avaliação e Ablação Híbrida:
  1. Estudo 6.1 — Aprendizado Contínuo Class-Incremental em Células Ativas Reais do ARC-AGI
     (5 tarefas reais sequenciais, 10 cores em pares disjuntos, sem dominação de background)
  2. Estudo 6.2 — Bateria Híbrida Simbólico + NREMCellularResidualLearner (Ciclo 3.5)
     (282 tarefas Near-Miss nos 3 grupos: ARC-1 Hold-Out, ARC-2 Eval e ARC-2 Train)

Opera 100% em CPU com NumPy puro (<400 MB de RAM) e salva os resultados auditáveis em:
  - benchmark_h6_real_arc_continual.json
  - benchmark_h6_hybrid_nrem_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Garante importação limpa a partir da raiz do repositório
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from f3_agi_v4_1_arc2_zerotouch import F3ARC2ZeroTouchSolver, sanitize_grid
from f3_nrem_cellular_mutator import (
    F3MetabolicCellularMLP,
    NREMCellularResidualLearner,
    extract_64d_cell_features,
)


# =============================================================================
# 1. ESTUDO 6.1: APRENDIZADO CONTÍNUO CLASS-INCREMENTAL EM CÉLULAS ATIVAS REAIS
# =============================================================================

def build_real_arc_cell_stream() -> List[Dict[str, Any]]:
    """
    Constrói um fluxo Class-Incremental de 5 tarefas reais sequenciais do ARC-AGI,
    cobrindo as 10 cores em pares disjuntos de classes ativas de transformação,
    sem viés de background dominante:
      - Tarefa 0 (32597951): Classes (0, 1)
      - Tarefa 1 (264363fd): Classes (2, 3)
      - Tarefa 2 (73251a56): Classes (4, 5)
      - Tarefa 3 (3631a71a): Classes (6, 7)
      - Tarefa 4 (83302e8f): Classes (8, 9)
    """
    task_configs = [
        ("32597951", (0, 1)),
        ("264363fd", (2, 3)),
        ("73251a56", (4, 5)),
        ("3631a71a", (6, 7)),
        ("83302e8f", (8, 9)),
    ]
    stream: List[Dict[str, Any]] = []

    for t_idx, (tid, (c1, c2)) in enumerate(task_configs):
        fpath = ROOT_DIR / "ARC-AGI" / "data" / "training" / f"{tid}.json"
        if not fpath.exists():
            fpath = ROOT_DIR / "ARC-AGI-2" / "data" / "training" / f"{tid}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Tarefa {tid} não encontrada em ARC-AGI ou ARC-AGI-2.")

        with open(fpath, "r", encoding="utf-8") as f:
            tdata = json.load(f)

        x_c1: List[np.ndarray] = []
        x_c2: List[np.ndarray] = []

        # Extrai vetores 64D das células reais dos pares de demonstração
        for pair in tdata.get("train", []):
            inp = np.array(pair["input"], dtype=int)
            exp = np.array(pair["output"], dtype=int)
            H, W = exp.shape
            feats = extract_64d_cell_features(inp, exp, c_bg=0)
            for r in range(H):
                for c in range(W):
                    val = int(exp[r, c])
                    if val == c1:
                        x_c1.append(feats[r, c])
                    elif val == c2:
                        x_c2.append(feats[r, c])

        rng = np.random.default_rng(42 + t_idx)
        rng.shuffle(x_c1)
        rng.shuffle(x_c2)

        # Balanceamento estrito entre classes
        n_per_class = min(len(x_c1), len(x_c2), 70)
        train_n = int(n_per_class * 0.70)
        test_n = n_per_class - train_n

        X_tr = np.array(x_c1[:train_n] + x_c2[:train_n], dtype=np.float32)
        y_tr = np.array([c1] * train_n + [c2] * train_n, dtype=np.int64)
        s_ids = np.arange(len(X_tr)) + t_idx * 10000

        # Embaralha para treino estocástico
        idx_tr = np.arange(len(X_tr))
        rng.shuffle(idx_tr)
        X_tr, y_tr, s_ids = X_tr[idx_tr], y_tr[idx_tr], s_ids[idx_tr]

        X_te = np.array(x_c1[train_n:n_per_class] + x_c2[train_n:n_per_class], dtype=np.float32)
        y_te = np.array([c1] * test_n + [c2] * test_n, dtype=np.int64)

        stream.append({
            "task_index": t_idx,
            "task_id": tid,
            "classes": (c1, c2),
            "X_train": X_tr,
            "y_train": y_tr,
            "sample_ids": s_ids,
            "X_test": X_te,
            "y_test": y_te,
            "n_train_cells": len(X_tr),
            "n_test_cells": len(X_te),
        })

    return stream


def run_real_arc_continual_regime(
    regime_name: str, stream: List[Dict[str, Any]], epochs: int = 12, lr: float = 0.045
) -> Dict[str, Any]:
    """Executa um dos 4 regimes contínuos no fluxo Class-Incremental de 5 tarefas com células ativas reais."""
    t0 = time.perf_counter()
    n_tasks = len(stream)
    net = F3MetabolicCellularMLP(
        input_dim=64,
        hidden_dim=128,
        num_classes=10,  # 10 cores do ARC
        learning_rate=lr,
        mistake_margin=0.15,
        syngap_gamma=30.0,
        lambda_anc=3000.0,
        seed=42,
    )

    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    ewc_consolidated: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []

    for t_idx, task in enumerate(stream):
        X_tr = task["X_train"]
        y_tr = task["y_train"]
        s_ids = task["sample_ids"]

        for epoch in range(epochs):
            for i in range(len(X_tr)):
                x = X_tr[i]
                y = int(y_tr[i])
                sid = int(s_ids[i])

                if regime_name == "ewc":
                    # EWC clássico (Kirkpatrick et al., 2017) com penalidade quadrática calibrada
                    net.total_forward_steps += 1
                    probs, cache = net.forward_single(x)
                    net.total_synaptic_updates += 1
                    grads = net.backward_single(probs, y, cache)
                    for k in net.param_names:
                        W = getattr(net, k)
                        g = grads[k]
                        ewc_grad = np.zeros_like(g)
                        for f_old, w_old in ewc_consolidated:
                            ewc_grad += 2000.0 * f_old[k] * (W - w_old[k])
                        delta = -lr * (g + ewc_grad)
                        setattr(net, k, W + delta)
                        net.m1_metabolic_cost += float(np.sum(np.abs(delta)))

                elif regime_name in ("sgd", "mistake_gating", "f3_metabolic"):
                    net.train_step(x, y, sid, regime=regime_name)

        # Consolidação pós-tarefa
        if regime_name == "ewc":
            f_task = {k: np.zeros_like(getattr(net, k)) for k in net.param_names}
            for i in range(len(X_tr)):
                probs, cache = net.forward_single(X_tr[i])
                g = net.backward_single(probs, int(y_tr[i]), cache)
                for k in net.param_names:
                    f_task[k] += (g[k] ** 2) / len(X_tr)
            w_task = {k: getattr(net, k).copy() for k in net.param_names}
            ewc_consolidated.append((f_task, w_task))

        elif regime_name == "f3_metabolic":
            net.consolidate_task_nrem((X_tr, y_tr))

        # Avalia acurácia em todas as tarefas vistas até agora
        for eval_idx in range(t_idx + 1):
            t_eval = stream[eval_idx]
            X_te = t_eval["X_test"]
            y_te = t_eval["y_te" if "y_te" in t_eval else "y_test"]
            correct = 0
            for k in range(len(X_te)):
                p, _ = net.forward_single(X_te[k])
                if int(np.argmax(p)) == int(y_te[k]):
                    correct += 1
            acc_matrix[t_idx, eval_idx] = 100.0 * correct / max(1, len(X_te))

    elapsed_s = time.perf_counter() - t0
    final_accs = acc_matrix[n_tasks - 1, :]
    mean_final_acc = round(float(np.mean(final_accs)), 2)

    # Forgetting ratio padrão (GEM, Lopez-Paz & Ranzato, 2017)
    forgetting_list = [
        float(acc_matrix[j, j] - acc_matrix[n_tasks - 1, j])
        for j in range(n_tasks - 1)
    ]
    forgetting_ratio = round(max(0.0, float(np.mean(forgetting_list))), 2)

    saved_updates_pct = round(
        100.0 * net.total_skipped_updates / max(1, net.total_forward_steps),
        2,
    )

    return {
        "regime_name": regime_name,
        "final_mean_accuracy_pct": mean_final_acc,
        "forgetting_ratio_pct": forgetting_ratio,
        "saved_synaptic_updates_pct": saved_updates_pct,
        "total_forward_steps": net.total_forward_steps,
        "total_synaptic_updates": net.total_synaptic_updates,
        "m1_metabolic_cost": round(net.m1_metabolic_cost, 2),
        "elapsed_sec": round(elapsed_s, 3),
        "task_accuracies": [round(float(a), 2) for a in final_accs],
        "accuracy_matrix": acc_matrix.tolist(),
    }


def execute_study_6_1() -> Dict[str, Any]:
    """Executa o Estudo 6.1 comparando os 4 regimes de plasticidade contínua."""
    print("\n" + "=" * 78)
    print("🔬 ESTUDO 6.1 — APRENDIZADO CONTÍNUO CLASS-INCREMENTAL EM CÉLULAS ATIVAS DO ARC")
    print("   Fluxo: 5 Tarefas Reais (10 Cores Disjuntas, Células de Transformação Balanceadas)")
    print("   Espaço de Entrada: 64D Cell Features | Zero Replay Buffer | 100% CPU NumPy")
    print("=" * 78)

    stream = build_real_arc_cell_stream()
    total_cells = sum(t["n_train_cells"] for t in stream)
    print(f"📦 Fluxo carregado com sucesso: {len(stream)} tarefas, {total_cells} células ativas de treino.")

    regimes = ["sgd", "ewc", "mistake_gating", "f3_metabolic"]
    regime_labels = {
        "sgd": "Standard SGD / Adam Baseline",
        "ewc": "Classic EWC (Fisher Quadratic Penalty)",
        "mistake_gating": "Pure Memorized Mistake Gating",
        "f3_metabolic": "F3-Metabolic-Optim (SynGAP + Mistake Gating + Tensor M)",
    }

    results: List[Dict[str, Any]] = []
    for reg in regimes:
        print(f"\n▶ Executando regime: {regime_labels[reg]}...")
        res = run_real_arc_continual_regime(reg, stream, epochs=12, lr=0.045)
        res["display_name"] = regime_labels[reg]
        results.append(res)
        print(
            f"   ✓ Acurácia Final: {res['final_mean_accuracy_pct']:.2f}% | "
            f"Esquecimento: {res['forgetting_ratio_pct']:.2f}% | "
            f"Updates Poupados: {res['saved_synaptic_updates_pct']:.2f}% | "
            f"Custo M1: {res['m1_metabolic_cost']:.1f} | "
            f"Tempo: {res['elapsed_sec']:.2f}s"
        )

    out_payload = {
        "study": "Estudo 6.1 — Continual Learning on Real ARC-AGI 64D Cell Vectors",
        "benchmark_dataset": "Real ARC-AGI 5-Task Sequential Stream",
        "tasks": [t["task_id"] for t in stream],
        "total_train_cells": total_cells,
        "regimes": results,
    }

    out_file = ROOT_DIR / "benchmark_h6_real_arc_continual.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Resultados do Estudo 6.1 salvos em: {out_file}")

    return out_payload


# =============================================================================
# 2. ESTUDO 6.2: BATERIA HÍBRIDA SIMBÓLICO + NREM NOS 282 NEAR-MISSES
# =============================================================================

def execute_study_6_2() -> Dict[str, Any]:
    """
    Executa a avaliação híbrida (v4.0.0 + NREMCellularResidualLearner Ciclo 3.5)
    sobre os 3 grupos de tarefas de h6_nearmiss_candidates.json.
    """
    print("\n" + "=" * 78)
    print("🔬 ESTUDO 6.2 — BATERIA HÍBRIDA (v4.0.0 + NREMCellularResidualLearner Ciclo 3.5)")
    print("   Alvo: 282 Tarefas Near-Miss Mapeadas (ARC-1 Hold-Out, ARC-2 Eval e ARC-2 Train)")
    print("=" * 78)

    candidates_path = ROOT_DIR / "h6_nearmiss_candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Arquivo de candidatos não encontrado: {candidates_path}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_data = json.load(f)

    solver = F3ARC2ZeroTouchSolver(max_cycles=3, max_time_per_task=5.0)
    learner = NREMCellularResidualLearner(c_bg=0)

    groups = [
        ("arc1_candidates", "ARC-1 Public Hold-Out (N=400 Split)", "ARC-AGI/data/evaluation"),
        ("arc2_evaluation_candidates", "ARC-2 Public Evaluation (N=120 Split)", "ARC-AGI-2/data/evaluation"),
        ("arc2_training_sample_candidates", "ARC-2 Training Sample (N=100 Split)", "ARC-AGI-2/data/training"),
    ]

    group_results: Dict[str, Any] = {}
    total_nearmiss_evaluated = 0
    total_nearmiss_converted = 0
    total_pixel_improved = 0

    for cand_key, group_label, base_dir in groups:
        tasks_meta = candidates_data.get(cand_key, [])
        print(f"\n▶ Avaliando Grupo: {group_label} ({len(tasks_meta)} near-misses em '{base_dir}')...")

        converted_exact: List[Dict[str, Any]] = []
        improved_pixel: List[Dict[str, Any]] = []
        unchanged: List[str] = []

        for idx, t_entry in enumerate(tasks_meta, 1):
            tid = t_entry["task_id"]
            fpath = ROOT_DIR / base_dir / f"{tid}.json"
            if not fpath.exists():
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                tdata = json.load(f)

            total_nearmiss_evaluated += 1
            # 1. Executa motor v4.0.0 base
            base_res = solver.solve_task(tid, tdata)
            base_acc = base_res["best_test_pixel_acc"]
            base_p1 = base_res["pass_at_1"]
            base_p2 = base_res["pass_at_2"]

            # Define função âncora simbólica segura
            winning_code = f"def transform(grid: np.ndarray) -> np.ndarray:\n    return {base_res['winning_expr']}"

            def anchor_call(g: np.ndarray) -> Optional[np.ndarray]:
                return solver._execute_code_on_grid(winning_code, g)

            # 2. Aciona o NREMCellularResidualLearner (Ciclo 3.5 com Motor B expandido)
            mut_res = learner.solve_residual(tdata, anchor_call)

            new_p1 = base_p1
            new_p2 = base_p2
            new_acc = base_acc

            if mut_res.get("patch_found") and mut_res.get("patched_test_preds"):
                test_pairs = tdata.get("test", [])
                p1_ok = True
                p2_ok = True
                test_accs = []

                for tp, patched_grid in zip(test_pairs, mut_res["patched_test_preds"]):
                    if "output" not in tp:
                        continue
                    gt = np.array(tp["output"], dtype=int)
                    arr_patched = np.array(patched_grid, dtype=int)
                    arr_anchor = np.array(base_res["predictions"][0]["attempt_1"], dtype=int)

                    # Se atingiu 100% de treino, patched é attempt_1 e anchor é attempt_2
                    if mut_res.get("perfect_train"):
                        a1, a2 = arr_patched, arr_anchor
                    else:
                        a1, a2 = arr_anchor, arr_patched

                    a1_match = bool(a1.shape == gt.shape and np.array_equal(a1, gt))
                    a2_match = bool(a2.shape == gt.shape and np.array_equal(a2, gt))

                    if not a1_match:
                        p1_ok = False
                    if not (a1_match or a2_match):
                        p2_ok = False

                    acc = float(np.mean(a1 == gt)) if a1.shape == gt.shape else 0.0
                    acc_2 = float(np.mean(a2 == gt)) if a2.shape == gt.shape else 0.0
                    test_accs.append(max(acc, acc_2))

                if test_accs:
                    new_acc = round(100.0 * float(np.mean(test_accs)), 2)
                    new_p1 = p1_ok
                    new_p2 = p2_ok

            if new_p2 and not base_p2:
                total_nearmiss_converted += 1
                converted_exact.append({
                    "task_id": tid,
                    "engine": mut_res.get("engine"),
                    "rule_type": mut_res.get("rule_type"),
                    "base_test_acc": base_acc,
                    "new_test_acc": new_acc,
                    "winning_expr": base_res["winning_expr"],
                })
                print(f"   🏆 [RESGATE 100%] Task {tid} convertida com sucesso via {mut_res.get('rule_type')}!")
            elif new_acc > base_acc:
                total_pixel_improved += 1
                improved_pixel.append({
                    "task_id": tid,
                    "engine": mut_res.get("engine"),
                    "rule_type": mut_res.get("rule_type"),
                    "base_test_acc": base_acc,
                    "new_test_acc": new_acc,
                })
                print(f"   🎯 [PRECISÃO ELEVADA] Task {tid}: {base_acc:.1f}% -> {new_acc:.1f}% via {mut_res.get('rule_type')}")
            else:
                unchanged.append(tid)

        group_results[cand_key] = {
            "group_label": group_label,
            "total_nearmisses": len(tasks_meta),
            "converted_to_exact_count": len(converted_exact),
            "pixel_accuracy_improved_count": len(improved_pixel),
            "unchanged_count": len(unchanged),
            "converted_tasks": converted_exact,
            "improved_tasks": improved_pixel,
        }

        print(
            f"   ✓ Resultado do Grupo: {len(converted_exact)} convertidas para 100% | "
            f"{len(improved_pixel)} com precisão de pixels elevada."
        )

    # Consolida os saltos globais
    arc1_converted = group_results["arc1_candidates"]["converted_to_exact_count"]
    arc2_eval_converted = group_results["arc2_evaluation_candidates"]["converted_to_exact_count"]
    arc2_train_converted = group_results["arc2_training_sample_candidates"]["converted_to_exact_count"]

    summary_payload = {
        "study": "Estudo 6.2 — Hybrid Symbolic + NREMCellularResidualLearner on Near-Misses",
        "total_nearmiss_evaluated": total_nearmiss_evaluated,
        "total_nearmiss_converted_to_exact": total_nearmiss_converted,
        "total_pixel_improved": total_pixel_improved,
        "arc1_original_pass2_count": 57,
        "arc1_hybrid_pass2_count": 57 + arc1_converted,
        "arc1_hybrid_pass2_pct": round(100.0 * (57 + arc1_converted) / 400.0, 2),
        "arc2_eval_original_pass2_count": 0,
        "arc2_eval_hybrid_pass2_count": arc2_eval_converted,
        "arc2_eval_hybrid_pass2_pct": round(100.0 * arc2_eval_converted / 120.0, 2),
        "arc2_train_original_pass2_count": 33,
        "arc2_train_hybrid_pass2_count": 33 + arc2_train_converted,
        "arc2_train_hybrid_pass2_pct": round(100.0 * (33 + arc2_train_converted) / 100.0, 2),
        "groups": group_results,
    }

    out_file = ROOT_DIR / "benchmark_h6_hybrid_nrem_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Resultados do Estudo 6.2 salvos em: {out_file}")

    return summary_payload


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main() -> None:
    t_start = time.perf_counter()
    print("🚀 INICIANDO BATERIA EXPERIMENTAL HORIZONTE 7 (F3-AGI v4.0)")
    print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Diretório Raiz: {ROOT_DIR}")

    # Estudo 6.1
    s61_results = execute_study_6_1()

    # Estudo 6.2
    s62_results = execute_study_6_2()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print("🏁 BATERIA EXPERIMENTAL HORIZONTE 7 CONCLUÍDA COM SUCESSO!")
    print(f"   Tempo Total: {total_time:.2f}s ({total_time / 60.0:.2f} min)")
    print(f"   ARC-1 Pass@2 Híbrido: {s62_results['arc1_hybrid_pass2_count']}/400 ({s62_results['arc1_hybrid_pass2_pct']}%)")
    print(f"   ARC-2 Eval Pass@2 Híbrido: {s62_results['arc2_eval_hybrid_pass2_count']}/120 ({s62_results['arc2_eval_hybrid_pass2_pct']}%)")
    print(f"   ARC-2 Train Pass@2 Híbrido: {s62_results['arc2_train_hybrid_pass2_count']}/100 ({s62_results['arc2_train_hybrid_pass2_pct']}%)")
    print(f"   Tarefas Convertidas para 100%: {s62_results['total_nearmiss_converted_to_exact']}")
    print(f"   Tarefas com Acurácia de Pixels Elevada: {s62_results['total_pixel_improved']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
