#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polybee AI Research — F3-AGI v4.0 (Outubro 2026)
Pacote Autossuficiente de Submissão Competitiva Kaggle (ARC Prize 2026)

Opera 100% offline (sem internet / sem chamadas de API externas), utilizando
exclusivamente CPU + NumPy (< 400 MB RAM):
  - F3Primitives (Core Knowledge Geométrico e Topológico)
  - F3LearnedOperators (Operadores Algébricos Paramétricos estilo SOAR/DreamCoder)
  - VisualCortexV40 (Extração de Invariantes + Embedding Isomórfico SIGReg-JEPA 512D)
  - CompositionalProgramSynthesizer (Busca em Feixe Profundidade 1..3 com Early-Reject Vetorizado)
  - ResidualPatchSynthesizer (Fechamento do Ciclo 3 sobre Máscaras de Erro Residual)
  - RealistV40 (Avaliação Estrutural Multi-Objetivo + Energia Livre F_F3)

Gera o arquivo oficial `submission.json` no formato exigido pelo Kaggle ARC Prize:
  {
    "<task_id>": [
      {
        "attempt_1": [[int, ...], ...],
        "attempt_2": [[int, ...], ...]
      }
    ]
  }
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Garante importação limpa a partir da raiz do projeto ou do mesmo diretório no Kaggle
ROOT_DIR = Path(__file__).resolve().parent.parent
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


class F3KaggleOfflineSolver:
    """
    Motor de síntese 100% offline para o Kaggle ARC Prize 2026.
    Produz duas hipóteses diversas (attempt_1 e attempt_2) maximizando a métrica oficial pass@2.
    """

    def __init__(self, max_cycles: int = 3, max_time_per_task: float = 4.0):
        self.max_cycles = max_cycles
        self.max_time_per_task = max_time_per_task
        self.cortex = VisualCortexV40()
        self.realist = RealistV40()
        self.synthesizer = CompositionalProgramSynthesizer(self.realist)

    @staticmethod
    def _sanitize_grid(grid: Any, fallback_input: np.ndarray) -> List[List[int]]:
        """Garante conformidade estrita com o schema do Kaggle (2D int list, 1<=H,W<=30, cores 0..9)."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            arr = np.asarray(fallback_input, dtype=int)
        else:
            arr = grid.astype(int)
        h, w = arr.shape
        if h < 1 or w < 1 or h > 30 or w > 30:
            arr = np.asarray(fallback_input, dtype=int)[:30, :30]
        arr = np.clip(arr, 0, 9)
        return arr.tolist()

    def _execute_code_on_grid(self, code: str, test_inp: np.ndarray) -> Optional[np.ndarray]:
        """Executa o programa sintetizado (`def transform(grid)`) em um grid de teste isolado."""
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

    def solve_task_pass_at_2(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa os 3 Ciclos do AACR offline (Acúmulo, Amplificação e Reparo Residual)
        e seleciona duas hipóteses distintas (`attempt_1` e `attempt_2`) para cada teste.
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

        # Ordena todos os candidatos únicos avaliados ao longo dos ciclos
        all_evaluated.sort(key=lambda h: (h.score, -h.free_energy), reverse=True)
        unique_ranked: List[HypothesisV40] = []
        seen_codes = set()
        for h in all_evaluated:
            if h.code not in seen_codes:
                seen_codes.add(h.code)
                unique_ranked.append(h)

        # Também adiciona candidatos diversos do beam_cache caso unique_ranked tenha poucas opções
        for op_tag, expr, thought, sc in self.synthesizer.beam_cache.get(task_id, []):
            code = f"def transform(grid: np.ndarray) -> np.ndarray:\n    return {expr}"
            if code not in seen_codes:
                seen_codes.add(code)
                h_extra = HypothesisV40(code=code, thought=f"[Beam | {op_tag}] {thought}", temperature=0.2)
                self.realist.evaluate(h_extra, train_pairs)
                unique_ranked.append(h_extra)

        unique_ranked.sort(key=lambda h: (h.score, -h.free_energy), reverse=True)
        # Ranking complementar para attempt_2: prioriza 2ª hipótese 100%-treino ou maior pixel_accuracy bruta
        runner_up_pool = sorted(
            unique_ranked[1:],
            key=lambda h: (1 if h.score >= 1.0 else 0, h.shape_accuracy, h.pixel_accuracy, h.score),
            reverse=True,
        )

        predictions_for_task: List[Dict[str, List[List[int]]]] = []
        top2_selected: Optional[HypothesisV40] = None

        for test_pair in test_pairs:
            test_inp = np.array(test_pair["input"], dtype=int)
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

            predictions_for_task.append({
                "attempt_1": self._sanitize_grid(pred_1_arr, test_inp),
                "attempt_2": self._sanitize_grid(pred_2_arr, test_inp),
            })

        elapsed_s = time.perf_counter() - t0
        top1 = unique_ranked[0] if unique_ranked else HypothesisV40(code="", thought="fallback", temperature=0.1)
        top2 = top2_selected or (unique_ranked[1] if len(unique_ranked) > 1 else top1)

        return {
            "task_id": task_id,
            "predictions": predictions_for_task,
            "top1_train_score": round(float(top1.score), 4),
            "top2_train_score": round(float(top2.score), 4),
            "top1_thought": top1.thought,
            "top2_thought": top2.thought,
            "elapsed_sec": round(elapsed_s, 4),
        }


def validate_kaggle_submission_schema(submission: Dict[str, Any], expected_task_ids: List[str]) -> Tuple[bool, str]:
    """Valida se o dicionário `submission.json` obedece estritamente às regras do Kaggle ARC Prize."""
    if not isinstance(submission, dict):
        return False, "Root object must be a JSON dictionary."
    for tid in expected_task_ids:
        if tid not in submission:
            return False, f"Missing task_id '{tid}' in submission.json."
        preds = submission[tid]
        if not isinstance(preds, list) or len(preds) < 1:
            return False, f"Task '{tid}' must map to a non-empty list of test predictions."
        for idx, entry in enumerate(preds):
            if not isinstance(entry, dict):
                return False, f"Task '{tid}' test[{idx}] must be a dict."
            for key in ("attempt_1", "attempt_2"):
                if key not in entry:
                    return False, f"Task '{tid}' test[{idx}] missing required key '{key}'."
                grid = entry[key]
                if not isinstance(grid, list) or not (1 <= len(grid) <= 30):
                    return False, f"Task '{tid}' {key} invalid row count."
                w = len(grid[0]) if isinstance(grid[0], list) else -1
                if not (1 <= w <= 30):
                    return False, f"Task '{tid}' {key} invalid col count."
                for row in grid:
                    if not isinstance(row, list) or len(row) != w:
                        return False, f"Task '{tid}' {key} non-rectangular grid."
                    for val in row:
                        if not isinstance(val, int) or not (0 <= val <= 9):
                            return False, f"Task '{tid}' {key} invalid pixel value '{val}'."
    return True, "Kaggle ARC Prize 2026 Schema Validated 100% OK"


def load_challenges_from_source(source_path: Path, limit: int = 10) -> Dict[str, Dict[str, Any]]:
    """Carrega tarefas de um arquivo JSON consolidado do Kaggle ou de um diretório de JSONs do ARC-AGI."""
    if source_path.is_file():
        data = json.loads(source_path.read_text(encoding="utf-8"))
        items = sorted(data.items())[:limit] if limit > 0 else sorted(data.items())
        return dict(items)

    files = sorted(source_path.glob("*.json"))
    if limit > 0:
        files = files[:limit]
    challenges: Dict[str, Dict[str, Any]] = {}
    for f in files:
        challenges[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return challenges


def main() -> None:
    parser = argparse.ArgumentParser(description="Polybee F3-AGI v4.0 Offline Kaggle Submission Runner")
    parser.add_argument("--input", type=str, default="ARC-AGI/data/evaluation", help="Diretório ARC ou JSON Kaggle")
    parser.add_argument("--output", type=str, default="kaggle_submission/submission.json", help="Arquivo submission.json")
    parser.add_argument("--limit", type=int, default=10, help="Limite de tarefas (0 = todas)")
    parser.add_argument("--max-time", type=float, default=4.0, help="Timeout máximo por tarefa em segundos")
    parser.add_argument("--verify-submission", action="store_true", help="Executa verificação e avalia pass@1 vs pass@2")
    args = parser.parse_args()

    source_path = ROOT_DIR / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_path = ROOT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = 10 if args.verify_submission and args.limit <= 0 else args.limit
    challenges = load_challenges_from_source(source_path, limit=limit)

    solver = F3KaggleOfflineSolver(max_cycles=3, max_time_per_task=args.max_time)
    submission: Dict[str, List[Dict[str, List[List[int]]]]] = {}
    verification_rows: List[Dict[str, Any]] = []

    t_start = time.perf_counter()
    pass1_hits = 0
    pass2_only_hits = 0
    pass_at_2_total = 0

    for tid, task_data in challenges.items():
        res = solver.solve_task_pass_at_2(tid, task_data)
        submission[tid] = res["predictions"]

        # Se os dados de avaliação possuírem ground-truth de teste, calcula pass@1 e pass@2
        test_pairs = task_data.get("test", [])
        p1_ok = True
        p2_ok = True
        has_gt = all("output" in tp for tp in test_pairs) and len(test_pairs) > 0

        if has_gt:
            s1_list, s2_list = [], []
            for tp, pred_dict in zip(test_pairs, res["predictions"]):
                gt = np.array(tp["output"], dtype=int)
                a1 = np.array(pred_dict["attempt_1"], dtype=int)
                a2 = np.array(pred_dict["attempt_2"], dtype=int)
                if not np.array_equal(a1, gt):
                    p1_ok = False
                if not (np.array_equal(a1, gt) or np.array_equal(a2, gt)):
                    p2_ok = False
                s1 = float(np.mean(a1 == gt)) if a1.shape == gt.shape else 0.0
                s2 = float(np.mean(a2 == gt)) if a2.shape == gt.shape else 0.0
                s1_list.append(s1)
                s2_list.append(s2)
            test_acc_1 = round(100.0 * float(np.mean(s1_list)), 2)
            test_acc_2 = round(100.0 * float(np.mean(s2_list)), 2)
            test_acc_best2 = round(max(test_acc_1, test_acc_2), 2)
        else:
            p1_ok = False
            p2_ok = False
            test_acc_1 = 0.0
            test_acc_2 = 0.0
            test_acc_best2 = 0.0

        if p1_ok:
            pass1_hits += 1
        if p2_ok and not p1_ok:
            pass2_only_hits += 1
        if p2_ok:
            pass_at_2_total += 1

        verification_rows.append({
            "task_id": tid,
            "top1_train_score": res["top1_train_score"],
            "top2_train_score": res["top2_train_score"],
            "test_pixel_acc_attempt_1_pct": test_acc_1,
            "test_pixel_acc_attempt_2_pct": test_acc_2,
            "test_pixel_acc_best_of_2_pct": test_acc_best2,
            "pass_at_1": p1_ok,
            "pass_at_2": p2_ok,
            "elapsed_sec": res["elapsed_sec"],
            "top1_thought": res["top1_thought"],
            "top2_thought": res["top2_thought"],
        })

    total_elapsed = time.perf_counter() - t_start
    out_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")

    # Também salva em submission.json na raiz caso exigido pelo ambiente Kaggle
    root_sub_path = ROOT_DIR / "submission.json"
    root_sub_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")

    schema_ok, schema_msg = validate_kaggle_submission_schema(submission, list(challenges.keys()))
    n_tasks = max(1, len(challenges))
    summary = {
        "schema_valid": schema_ok,
        "schema_message": schema_msg,
        "submission_file": str(out_path),
        "total_tasks": len(challenges),
        "pass_at_1_count": pass1_hits,
        "pass_at_1_pct": round(100.0 * pass1_hits / n_tasks, 2),
        "attempt_2_rescue_count": pass2_only_hits,
        "pass_at_2_count": pass_at_2_total,
        "pass_at_2_pct": round(100.0 * pass_at_2_total / n_tasks, 2),
        "total_elapsed_sec": round(total_elapsed, 3),
        "mean_elapsed_sec": round(total_elapsed / n_tasks, 3),
        "tasks": verification_rows,
    }
    verify_json_path = out_path.parent / "verification_report.json"
    verify_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(challenges) >= 400:
        full_sub_path = out_path.parent / "submission_full400.json"
        full_rep_path = out_path.parent / "verification_report_full400.json"
        if out_path.resolve() != full_sub_path.resolve():
            full_sub_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        summary_400 = dict(summary)
        summary_400["submission_file"] = str(full_sub_path)
        full_rep_path.write_text(json.dumps(summary_400, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("POLYBEE F3-AGI v4.0 — KAGGLE ARC PRIZE OFFLINE SUBMISSION VERIFICATION")
    print("=" * 78)
    print(f"Arquivo Gerado   : {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"Validação Schema : {schema_msg}")
    print(f"Tarefas Testadas : {len(challenges)} ({args.input})")
    print(f"Score pass@1     : {pass1_hits}/{len(challenges)} ({summary['pass_at_1_pct']:.2f}%)")
    print(f"Score pass@2     : {pass_at_2_total}/{len(challenges)} ({summary['pass_at_2_pct']:.2f}%) [+{pass2_only_hits} via attempt_2]")
    print(f"Tempo Total      : {total_elapsed:.2f}s ({summary['mean_elapsed_sec']:.2f}s/tarefa)")
    print("=" * 78)


if __name__ == "__main__":
    main()
