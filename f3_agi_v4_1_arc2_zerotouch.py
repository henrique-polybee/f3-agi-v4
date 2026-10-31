#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polybee AI Research — F3-AGI v4.0 (Outubro 2026)
Horizonte 6 — Fase 6A: Avaliação Zero-Touch Out-of-Distribution no ARC-AGI-2

Importa estritamente o motor neuro-simbólico congelado da release v4.0.0:
  - f3_agi_v4_0_core.py (VisualCortexV40, CompositionalProgramSynthesizer, RealistV40, ResidualPatchSynthesizer)
  - f3_learned_library.py (Biblioteca paramétrica congelada com 30 operadores)
SEM MODIFICAR NEM ADICIONAR NENHUM OPERADOR, medindo a generalização out-of-distribution pura.

Baterias Executadas:
  1. Avaliação Completa no Conjunto Público ARC-AGI-2/data/evaluation (todas as N = 120 tarefas)
  2. Avaliação Amostral no Conjunto ARC-AGI-2/data/training (primeiras N = 100 tarefas lexicográficas)
  3. Mapeamento de Near-Misses (ARC-AGI-1 full400 + ARC-AGI-2 eval/train) para o Mutador Neural NREM (Fase 6B)
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Garante importação limpa a partir da raiz do repositório
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from f3_agi_v4_0_core import (
    CompositionalProgramSynthesizer,
    F3Primitives,
    HypothesisV40,
    RealistV40,
    ResidualPatchSynthesizer,
    VisualCortexV40,
    center_of_mass,
    find_objects,
    label,
    load_learned_library_scope,
)
from f3_learned_library import F3LearnedOperators


def sanitize_grid(grid: Any, fallback_input: np.ndarray) -> List[List[int]]:
    """Garante conformidade com o schema do ARC Prize (2D int list, 1<=H,W<=30, cores 0..9)."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        arr = np.asarray(fallback_input, dtype=int)
    else:
        arr = grid.astype(int)
    h, w = arr.shape
    if h < 1 or w < 1 or h > 30 or w > 30:
        arr = np.asarray(fallback_input, dtype=int)[:30, :30]
    arr = np.clip(arr, 0, 9)
    return arr.tolist()


def extract_expression_from_code(code: str, fallback: str = "") -> str:
    """Extrai a expressão funcional Python ou resumo simbólico do código da hipótese."""
    if not code:
        return fallback or "identity(grid)"
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    ret_lines = [ln for ln in lines if ln.startswith("return ")]
    if ret_lines:
        return ret_lines[-1][len("return "):].strip()
    return fallback or lines[-1]


class F3ARC2ZeroTouchSolver:
    """
    Executor Zero-Touch para o benchmark ARC-AGI-2.
    Opera 100% em CPU com o motor neuro-simbólico congelado da v4.0.0.
    """

    def __init__(self, max_cycles: int = 3, max_time_per_task: float = 15.0):
        self.max_cycles = max_cycles
        self.max_time_per_task = max_time_per_task
        self.cortex = VisualCortexV40()
        self.realist = RealistV40()
        self.synthesizer = CompositionalProgramSynthesizer(self.realist)

    def _execute_code_on_grid(self, code: str, test_inp: np.ndarray) -> Optional[np.ndarray]:
        """Executa com segurança o código sintetizado sobre um grid de teste."""
        try:
            scope: Dict[str, Any] = {
                "np": np,
                "F3Primitives": F3Primitives,
                "F3LearnedOperators": F3LearnedOperators,
                "find_objects": find_objects,
                "label": label,
                "center_of_mass": center_of_mass,
            }
            scope.update(load_learned_library_scope())
            local_ns: Dict[str, Any] = {}
            exec(code, scope, local_ns)
            fn = local_ns.get("transform") or local_ns.get("transform_grid")
            if callable(fn):
                res = np.asarray(fn(test_inp.copy()), dtype=int)
                if res.ndim == 2:
                    h, w = res.shape
                    if 1 <= h <= 30 and 1 <= w <= 30:
                        return np.clip(res, 0, 9)
        except Exception:
            pass
        return None

    def solve_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa os 3 Ciclos do AACR (Acúmulo D1, Amplificação D2 e ResidualPatch D3)
        para resolver a tarefa e gerar previsões attempt_1 e attempt_2.
        """
        t0 = time.perf_counter()
        train_pairs = task_data.get("train", [])
        test_pairs = task_data.get("test", [])

        analysis = self.cortex.analyze_task(train_pairs)
        scene_desc = self.cortex.format_for_prompt(analysis, train_pairs)
        ctx = {
            "task_id": task_id,
            "scene_description": scene_desc,
            "analysis": analysis,
            "train_examples": train_pairs,
        }

        all_evaluated: List[HypothesisV40] = []
        best_overall: Optional[HypothesisV40] = None

        for cycle_num in range(1, self.max_cycles + 1):
            elapsed_so_far = time.perf_counter() - t0
            rem_budget = self.max_time_per_task - elapsed_so_far
            if rem_budget <= 0.05:
                break
            # Orçamento adaptativo por ciclo
            cycle_budget = min(rem_budget, self.max_time_per_task * (0.55 if cycle_num == 1 else 0.35))
            hyps = self.synthesizer.synthesize_candidates(
                ctx,
                cycle=cycle_num,
                best_previous=best_overall,
                n_hypotheses=8,
                max_time_sec=max(0.15, cycle_budget),
            )
            evals = self.realist.evaluate_all(hyps, train_pairs)
            evals.sort(key=lambda h: (h.score, -h.free_energy), reverse=True)
            all_evaluated.extend(evals)

            if evals and (best_overall is None or evals[0].score > best_overall.score):
                best_overall = evals[0]
                best_overall.solved_in_cycle = cycle_num

            if best_overall is not None and best_overall.score >= 1.0:
                break

        # Ranking unificado sem duplicatas
        all_evaluated.sort(key=lambda h: (h.score, -h.free_energy), reverse=True)
        unique_ranked: List[HypothesisV40] = []
        seen_codes = set()
        for h in all_evaluated:
            if h.code not in seen_codes:
                seen_codes.add(h.code)
                unique_ranked.append(h)

        # Candidatos do beam cache caso necessário
        for op_tag, expr, thought, sc in self.synthesizer.beam_cache.get(task_id, []):
            code = f"def transform(grid: np.ndarray) -> np.ndarray:\n    return {expr}"
            if code not in seen_codes:
                seen_codes.add(code)
                h_extra = HypothesisV40(code=code, thought=f"[Beam | {op_tag}] {thought}", temperature=0.2)
                self.realist.evaluate(h_extra, train_pairs)
                unique_ranked.append(h_extra)

        unique_ranked.sort(key=lambda h: (h.score, -h.free_energy), reverse=True)

        # Pool alternativo para attempt_2
        runner_up_pool = sorted(
            unique_ranked[1:],
            key=lambda h: (1 if h.score >= 1.0 else 0, h.shape_accuracy, h.pixel_accuracy, h.score),
            reverse=True,
        )

        top1 = unique_ranked[0] if unique_ranked else HypothesisV40(code="", thought="identity fallback", temperature=0.1)
        top2_selected: Optional[HypothesisV40] = None

        predictions: List[Dict[str, List[List[int]]]] = []
        p1_exact_all = True
        p2_exact_all = True
        s1_acc_list: List[float] = []
        s2_acc_list: List[float] = []

        for tp in test_pairs:
            test_inp = np.array(tp["input"], dtype=int)
            pred_1_arr: Optional[np.ndarray] = None
            pred_2_arr: Optional[np.ndarray] = None

            for h_cand in unique_ranked:
                out_arr = self._execute_code_on_grid(h_cand.code, test_inp)
                if out_arr is not None:
                    pred_1_arr = out_arr
                    break

            if pred_1_arr is not None:
                for h_cand in runner_up_pool:
                    out_arr = self._execute_code_on_grid(h_cand.code, test_inp)
                    if out_arr is not None and not np.array_equal(out_arr, pred_1_arr):
                        pred_2_arr = out_arr
                        if top2_selected is None:
                            top2_selected = h_cand
                        break

            if pred_1_arr is None:
                pred_1_arr = test_inp.copy()
            if pred_2_arr is None:
                pred_2_arr = np.fliplr(pred_1_arr)
                if np.array_equal(pred_2_arr, pred_1_arr):
                    pred_2_arr = np.rot90(pred_1_arr)

            p1_clean = sanitize_grid(pred_1_arr, test_inp)
            p2_clean = sanitize_grid(pred_2_arr, test_inp)
            predictions.append({"attempt_1": p1_clean, "attempt_2": p2_clean})

            # Avaliação contra ground truth caso disponível
            if "output" in tp:
                gt = np.array(tp["output"], dtype=int)
                a1 = np.array(p1_clean, dtype=int)
                a2 = np.array(p2_clean, dtype=int)

                a1_match = bool(a1.shape == gt.shape and np.array_equal(a1, gt))
                a2_match = bool(a2.shape == gt.shape and np.array_equal(a2, gt))

                if not a1_match:
                    p1_exact_all = False
                if not (a1_match or a2_match):
                    p2_exact_all = False

                acc1 = float(np.mean(a1 == gt)) if a1.shape == gt.shape else 0.0
                acc2 = float(np.mean(a2 == gt)) if a2.shape == gt.shape else 0.0
                s1_acc_list.append(acc1)
                s2_acc_list.append(acc2)
            else:
                p1_exact_all = False
                p2_exact_all = False

        elapsed_sec = time.perf_counter() - t0
        has_gt = all("output" in tp for tp in test_pairs) and len(test_pairs) > 0

        if has_gt:
            acc_attempt_1 = round(100.0 * float(np.mean(s1_acc_list)), 2)
            acc_attempt_2 = round(100.0 * float(np.mean(s2_acc_list)), 2)
            best_test_pixel_acc = round(max(acc_attempt_1, acc_attempt_2), 2)
        else:
            acc_attempt_1 = 0.0
            acc_attempt_2 = 0.0
            best_test_pixel_acc = 0.0
            p1_exact_all = False
            p2_exact_all = False

        used_residual = bool(
            "ResidualPatch" in (top1.thought or "")
            or "apply_residual_patch" in (top1.code or "")
            or getattr(top1, "solved_in_cycle", 0) == 3
        )
        winning_expr = extract_expression_from_code(top1.code, fallback=top1.thought)

        return {
            "task_id": task_id,
            "train_solved": bool(top1.score >= 1.0),
            "best_train_score": round(float(top1.score), 4),
            "shape_match": bool(getattr(top1, "shape_accuracy", 0.0) >= 1.0),
            "train_pixel_acc": round(float(getattr(top1, "pixel_accuracy", 0.0)) * 100.0, 2),
            "train_object_acc": round(float(getattr(top1, "object_accuracy", 0.0)) * 100.0, 2),
            "pass_at_1": p1_exact_all,
            "pass_at_2": p2_exact_all,
            "test_pixel_acc_attempt_1_pct": acc_attempt_1,
            "test_pixel_acc_attempt_2_pct": acc_attempt_2,
            "best_test_pixel_acc": best_test_pixel_acc,
            "winning_expr": winning_expr,
            "top1_thought": top1.thought,
            "top2_thought": top2_selected.thought if top2_selected else "",
            "used_residual_patch": used_residual,
            "solved_in_cycle": getattr(top1, "solved_in_cycle", None) if top1.score >= 1.0 else None,
            "latency_s": round(elapsed_sec, 4),
            "predictions": predictions,
        }


def evaluate_directory(
    dir_path: Path,
    limit: int = 0,
    max_time_per_task: float = 15.0,
    split_name: str = "ARC-AGI-2 Evaluation",
) -> Dict[str, Any]:
    """Executa a bateria Zero-Touch sobre um diretório de arquivos JSON de tarefas."""
    files = sorted(dir_path.glob("*.json"))
    if limit > 0:
        files = files[:limit]

    print(f"\n{'='*78}")
    print(f"🚀 INICIANDO AVALIAÇÃO ZERO-TOUCH: {split_name} (N = {len(files)})")
    print(f"   Diretório: {dir_path}")
    print(f"   Timeout Máximo: {max_time_per_task:.1f}s/tarefa | Modo: 100% CPU / <400MB RAM")
    print(f"{'='*78}")

    solver = F3ARC2ZeroTouchSolver(max_cycles=3, max_time_per_task=max_time_per_task)
    results: List[Dict[str, Any]] = []

    t_start = time.perf_counter()
    pass1_count = 0
    pass2_count = 0
    train_solved_count = 0
    shape_match_count = 0
    ge85_count = 0
    ge90_count = 0
    solved_tasks: List[Dict[str, Any]] = []

    for idx, fpath in enumerate(files, 1):
        tid = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                tdata = json.load(f)
        except Exception as e:
            print(f"   [{idx:03d}/{len(files):03d}] ❌ Erro ao ler {fpath}: {e}")
            continue

        res = solver.solve_task(tid, tdata)
        results.append(res)

        if res["pass_at_1"]:
            pass1_count += 1
        if res["pass_at_2"]:
            pass2_count += 1
            solved_tasks.append({
                "task_id": tid,
                "pass_at_1": res["pass_at_1"],
                "pass_at_2": res["pass_at_2"],
                "best_train_score": res["best_train_score"],
                "best_test_pixel_acc": res["best_test_pixel_acc"],
                "winning_expr": res["winning_expr"],
                "used_residual_patch": res["used_residual_patch"],
                "latency_s": res["latency_s"],
            })

        if res["train_solved"]:
            train_solved_count += 1
        if res["shape_match"]:
            shape_match_count += 1
        if res["best_test_pixel_acc"] >= 85.0:
            ge85_count += 1
        if res["best_test_pixel_acc"] >= 90.0:
            ge90_count += 1

        status_icon = "🏆" if res["pass_at_2"] else ("🎯" if res["best_test_pixel_acc"] >= 85.0 else "·")
        print(
            f"   [{idx:03d}/{len(files):03d}] {status_icon} Task {tid} | "
            f"TrainScore: {res['best_train_score']:.3f} | "
            f"Shape: {'OK' if res['shape_match'] else '--'} | "
            f"TestAcc: {res['best_test_pixel_acc']:.1f}% | "
            f"pass@1: {res['pass_at_1']} | pass@2: {res['pass_at_2']} | "
            f"{res['latency_s']:.2f}s"
        )

    total_time = time.perf_counter() - t_start
    n_total = max(1, len(files))

    pass1_pct = round(100.0 * pass1_count / n_total, 2)
    pass2_pct = round(100.0 * pass2_count / n_total, 2)
    rescue_count = pass2_count - pass1_count
    train_solved_pct = round(100.0 * train_solved_count / n_total, 2)
    transfer_rate_pct = round(100.0 * pass2_count / max(1, train_solved_count), 2)
    ge85_pct = round(100.0 * ge85_count / n_total, 2)
    ge90_pct = round(100.0 * ge90_count / n_total, 2)
    mean_lat = round(total_time / n_total, 3)

    summary = {
        "split_name": split_name,
        "directory": str(dir_path),
        "total_tasks": len(files),
        "pass_at_1_count": pass1_count,
        "pass_at_1_pct": pass1_pct,
        "pass_at_2_count": pass2_count,
        "pass_at_2_pct": pass2_pct,
        "attempt_2_rescues": rescue_count,
        "train_solved_count": train_solved_count,
        "train_solved_pct": train_solved_pct,
        "transfer_rate_pct": transfer_rate_pct,
        "shape_match_count": shape_match_count,
        "shape_match_pct": round(100.0 * shape_match_count / n_total, 2),
        "high_precision_ge_85_count": ge85_count,
        "high_precision_ge_85_pct": ge85_pct,
        "high_precision_ge_90_count": ge90_count,
        "high_precision_ge_90_pct": ge90_pct,
        "total_time_s": round(total_time, 2),
        "mean_latency_s": mean_lat,
        "solved_tasks": solved_tasks,
        "tasks": results,
    }

    print(f"\n📊 RESUMO DO SPLIT: {split_name}")
    print(f"   • pass@1: {pass1_count}/{n_total} ({pass1_pct:.2f}%)")
    print(f"   • pass@2: {pass2_count}/{n_total} ({pass2_pct:.2f}%) [Resgates via attempt_2: +{rescue_count}]")
    print(f"   • Train Solved: {train_solved_count}/{n_total} ({train_solved_pct:.2f}%)")
    print(f"   • Transferência (Train -> Test Solved): {transfer_rate_pct:.2f}%")
    print(f"   • Test Pixel Acc >= 85%: {ge85_count}/{n_total} ({ge85_pct:.2f}%)")
    print(f"   • Test Pixel Acc >= 90%: {ge90_count}/{n_total} ({ge90_pct:.2f}%)")
    print(f"   • Latência Média: {mean_lat:.2f}s/tarefa | Tempo Total: {total_time:.2f}s")
    print(f"{'='*78}\n")

    return summary


def extract_near_miss_candidates(
    arc1_eval_file: Path,
    arc2_eval_summary: Dict[str, Any],
    arc2_train_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Identifica e compila a lista exata de tarefas Near-Miss:
    Critério canônico: (S_shape == 1.0 e A_struct in [0.80, 1.0)) ou (best_test_pixel_acc >= 85.0% e < 100.0%).
    """
    arc1_candidates: List[Dict[str, Any]] = []
    if arc1_eval_file.exists():
        try:
            with open(arc1_eval_file, "r", encoding="utf-8") as f:
                h5_data = json.load(f)
            for t in h5_data.get("tasks", []):
                t_score = float(t.get("top1_train_score", 0.0))
                test_best = float(t.get("test_pixel_acc_best_of_2_pct", 0.0))
                pass_2 = bool(t.get("pass_at_2", False))
                is_near = (not pass_2) and ((0.80 <= t_score < 1.0) or (85.0 <= test_best < 100.0))
                if is_near:
                    arc1_candidates.append({
                        "dataset": "ARC-AGI-1",
                        "split": "evaluation",
                        "task_id": t.get("task_id"),
                        "train_score": round(t_score, 4),
                        "test_pixel_acc_best_pct": round(test_best, 2),
                        "top1_thought": t.get("top1_thought", "")[:120],
                    })
        except Exception as e:
            print(f"⚠️ Aviso ao carregar {arc1_eval_file}: {e}")

    arc2_eval_candidates: List[Dict[str, Any]] = []
    for t in arc2_eval_summary.get("tasks", []):
        t_score = float(t.get("best_train_score", 0.0))
        shape_ok = bool(t.get("shape_match", False))
        test_best = float(t.get("best_test_pixel_acc", 0.0))
        pass_2 = bool(t.get("pass_at_2", False))
        is_near = (not pass_2) and ((shape_ok and 0.80 <= t_score < 1.0) or (85.0 <= test_best < 100.0))
        if is_near:
            arc2_eval_candidates.append({
                "dataset": "ARC-AGI-2",
                "split": "evaluation",
                "task_id": t.get("task_id"),
                "shape_match": shape_ok,
                "train_score": round(t_score, 4),
                "test_pixel_acc_best_pct": round(test_best, 2),
                "winning_expr": t.get("winning_expr", "")[:120],
                "top1_thought": t.get("top1_thought", "")[:120],
                "used_residual_patch": t.get("used_residual_patch", False),
            })

    arc2_train_candidates: List[Dict[str, Any]] = []
    for t in arc2_train_summary.get("tasks", []):
        t_score = float(t.get("best_train_score", 0.0))
        shape_ok = bool(t.get("shape_match", False))
        test_best = float(t.get("best_test_pixel_acc", 0.0))
        pass_2 = bool(t.get("pass_at_2", False))
        is_near = (not pass_2) and ((shape_ok and 0.80 <= t_score < 1.0) or (85.0 <= test_best < 100.0))
        if is_near:
            arc2_train_candidates.append({
                "dataset": "ARC-AGI-2",
                "split": "training_sample_100",
                "task_id": t.get("task_id"),
                "shape_match": shape_ok,
                "train_score": round(t_score, 4),
                "test_pixel_acc_best_pct": round(test_best, 2),
                "winning_expr": t.get("winning_expr", "")[:120],
                "top1_thought": t.get("top1_thought", "")[:120],
                "used_residual_patch": t.get("used_residual_patch", False),
            })

    total_candidates = len(arc1_candidates) + len(arc2_eval_candidates) + len(arc2_train_candidates)
    return {
        "nearmiss_definition": "(shape_match == True and 0.80 <= train_score < 1.00) or (85.0% <= best_test_pixel_acc < 100.0%)",
        "total_nearmiss_candidates": total_candidates,
        "arc1_full400_count": len(arc1_candidates),
        "arc2_evaluation_120_count": len(arc2_eval_candidates),
        "arc2_training_100_count": len(arc2_train_candidates),
        "arc1_candidates": arc1_candidates,
        "arc2_evaluation_candidates": arc2_eval_candidates,
        "arc2_training_sample_candidates": arc2_train_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polybee F3-AGI v4.0 — Zero-Touch ARC-AGI-2 Benchmark & Near-Miss Candidate Extractor"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="ARC-AGI-2/data/evaluation",
        help="Diretório de avaliação oficial do ARC-AGI-2",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="ARC-AGI-2/data/training",
        help="Diretório de treinamento oficial do ARC-AGI-2",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=120,
        help="Quantidade de tarefas para o split de avaliação (padrão: 120 = todas)",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=100,
        help="Quantidade de tarefas para o split de treinamento (padrão: 100)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=15.0,
        help="Timeout seguro por tarefa em segundos (padrão: 15.0s)",
    )
    parser.add_argument(
        "--out-results",
        type=str,
        default="benchmark_h6_arc2_zerotouch.json",
        help="Arquivo JSON para telemetria auditável dos resultados",
    )
    parser.add_argument(
        "--out-nearmiss",
        type=str,
        default="h6_nearmiss_candidates.json",
        help="Arquivo JSON para exportação dos candidatos Near-Miss",
    )
    args = parser.parse_args()

    eval_path = ROOT_DIR / args.eval_dir if not Path(args.eval_dir).is_absolute() else Path(args.eval_dir)
    train_path = ROOT_DIR / args.train_dir if not Path(args.train_dir).is_absolute() else Path(args.train_dir)
    arc1_h5_path = ROOT_DIR / "benchmark_h5_full400_eval.json"

    if not eval_path.is_dir():
        raise FileNotFoundError(f"Diretório ARC-AGI-2 de avaliação não encontrado: {eval_path}")
    if not train_path.is_dir():
        raise FileNotFoundError(f"Diretório ARC-AGI-2 de treinamento não encontrado: {train_path}")

    print("=" * 78)
    print("🐝 POLYBEE AI RESEARCH — F3-AGI v4.0 (RELEASE CONGELADA v4.0.0)")
    print("🔬 HORIZONTE 6 — FASE 6A: AVALIAÇÃO ZERO-TOUCH OUT-OF-DISTRIBUTION NO ARC-AGI-2")
    print(f"   Motor: 30 Operadores Paramétricos Congelados + ResidualPatchSynthesizer")
    print(f"   Hardware: 100% CPU Local / Zero Chamadas Externas / < 400 MB RAM")
    print("=" * 78)

    # 1. Avaliação Completa no Conjunto Público ARC-AGI-2/data/evaluation (N = 120)
    eval_summary = evaluate_directory(
        eval_path,
        limit=args.eval_limit,
        max_time_per_task=args.max_time,
        split_name="ARC-AGI-2 Public Evaluation (Full N=120)",
    )

    # 2. Avaliação Amostral no Conjunto ARC-AGI-2/data/training (primeiras N = 100)
    train_summary = evaluate_directory(
        train_path,
        limit=args.train_limit,
        max_time_per_task=args.max_time,
        split_name="ARC-AGI-2 Training Calibration Sample (N=100)",
    )

    # 3. Compilação de Resultados Auditáveis
    payload = {
        "benchmark": "Polybee F3-AGI v4.0 — Zero-Touch ARC-AGI-2 Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "frozen_release_tag": "v4.0.0",
        "frozen_doi": "10.5281/zenodo.23816665",
        "evaluation_summary": eval_summary,
        "training_sample_summary": train_summary,
    }

    out_results_path = ROOT_DIR / args.out_results
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"💾 Telemetria completa salva em: {out_results_path}")

    # 4. Mapeamento de Near-Misses para a Fase 6B
    nearmiss_data = extract_near_miss_candidates(arc1_h5_path, eval_summary, train_summary)
    out_nearmiss_path = ROOT_DIR / args.out_nearmiss
    with open(out_nearmiss_path, "w", encoding="utf-8") as f:
        json.dump(nearmiss_data, f, indent=2, ensure_ascii=False)
    print(f"🎯 Candidatos Near-Miss salvos em: {out_nearmiss_path} ({nearmiss_data['total_nearmiss_candidates']} tarefas)")

    # 5. Tabela Comparativa Final
    print("\n" + "=" * 78)
    print("📋 TABELA COMPARATIVA OFICIAL — ZERO-TOUCH ARC-AGI-2 (MOTOR CONGELADO v4.0.0)")
    print("=" * 78)
    print(f"{'Métrica':<42} | {'Evaluation (N=120)':<17} | {'Training (N=100)':<17}")
    print("-" * 78)
    print(f"{'Tarefas Avaliadas':<42} | {eval_summary['total_tasks']:<17} | {train_summary['total_tasks']:<17}")
    print(f"{'pass@1 (Attempt 1)':<42} | {eval_summary['pass_at_1_count']:>3} ({eval_summary['pass_at_1_pct']:>5.2f}%)     | {train_summary['pass_at_1_count']:>3} ({train_summary['pass_at_1_pct']:>5.2f}%)")
    print(f"{'pass@2 (Attempt 1 or 2)':<42} | {eval_summary['pass_at_2_count']:>3} ({eval_summary['pass_at_2_pct']:>5.2f}%)     | {train_summary['pass_at_2_count']:>3} ({train_summary['pass_at_2_pct']:>5.2f}%)")
    print(f"{'Resgates Exclusivos via Attempt 2':<42} | +{eval_summary['attempt_2_rescues']:<16} | +{train_summary['attempt_2_rescues']:<16}")
    print(f"{'Train Solved (A_struct == 1.0)':<42} | {eval_summary['train_solved_count']:>3} ({eval_summary['train_solved_pct']:>5.2f}%)     | {train_summary['train_solved_count']:>3} ({train_summary['train_solved_pct']:>5.2f}%)")
    print(f"{'Taxa Transferência (Train -> Test)':<42} | {eval_summary['transfer_rate_pct']:>5.2f}%            | {train_summary['transfer_rate_pct']:>5.2f}%")
    print(f"{'Shape Match Exato (S_shape == 1.0)':<42} | {eval_summary['shape_match_count']:>3} ({eval_summary['shape_match_pct']:>5.2f}%)     | {train_summary['shape_match_count']:>3} ({train_summary['shape_match_pct']:>5.2f}%)")
    print(f"{'Alta Precisão de Pixels (>= 85%)':<42} | {eval_summary['high_precision_ge_85_count']:>3} ({eval_summary['high_precision_ge_85_pct']:>5.2f}%)     | {train_summary['high_precision_ge_85_count']:>3} ({train_summary['high_precision_ge_85_pct']:>5.2f}%)")
    print(f"{'Alta Precisão de Pixels (>= 90%)':<42} | {eval_summary['high_precision_ge_90_count']:>3} ({eval_summary['high_precision_ge_90_pct']:>5.2f}%)     | {train_summary['high_precision_ge_90_count']:>3} ({train_summary['high_precision_ge_90_pct']:>5.2f}%)")
    print(f"{'Latência Média por Tarefa (CPU)':<42} | {eval_summary['mean_latency_s']:>6.3f}s           | {train_summary['mean_latency_s']:>6.3f}s")
    print(f"{'Tempo Total de Execução':<42} | {eval_summary['total_time_s']:>6.2f}s           | {train_summary['total_time_s']:>6.2f}s")
    print("=" * 78)

    print("\n🏆 TAREFAS RESOLVIDAS NO SPLIT EVALUATION (ARC-AGI-2/data/evaluation):")
    if eval_summary["solved_tasks"]:
        for st in eval_summary["solved_tasks"]:
            res_str = " [ResidualPatch]" if st["used_residual_patch"] else ""
            print(f"   • Task {st['task_id']}{res_str} ({st['latency_s']:.2f}s): {st['winning_expr']}")
    else:
        print("   (Nenhuma tarefa resolvida com 100% de match no teste)")

    print("\n🏆 TAREFAS RESOLVIDAS NO SPLIT TRAINING (ARC-AGI-2/data/training, Amostra N=100):")
    if train_summary["solved_tasks"]:
        for st in train_summary["solved_tasks"]:
            res_str = " [ResidualPatch]" if st["used_residual_patch"] else ""
            print(f"   • Task {st['task_id']}{res_str} ({st['latency_s']:.2f}s): {st['winning_expr']}")
    else:
        print("   (Nenhuma tarefa resolvida com 100% de match no teste)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
