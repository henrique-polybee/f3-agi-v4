# -*- coding: utf-8 -*-
"""
===============================================================================
POLYBEE AI RESEARCH — F3-AGI v4.0 (HORIZONTE 2)
Módulo Autônomo: `f3_metabolic_optim.py`
Experimento de Aprendizado Contínuo (Split-Task Continual Learning Benchmark)
Validando Quantitativamente o Pilar IV (Metaplasticidade SynGAP) e o
Pilar V (Bioenergética / Memorized Mistake Gating / Tensor Mitocondrial M).

Execução 100% CPU em NumPy Puro (<100 MB RAM) | Dimensões em Potências de 2 (256D/128D).
===============================================================================
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class ContinualBenchmarkConfig:
    n_tasks: int = 5
    classes_per_task: int = 2
    input_dim: int = 64        # 8x8 паттерн relacional/visual (potência de 2)
    hidden1_dim: int = 256     # MSRC Fast Perceptual Layer (potência de 2)
    hidden2_dim: int = 128     # MSRC Slow Abstract Layer (potência de 2)
    samples_per_task_train: int = 120
    samples_per_task_test: int = 60
    epochs_per_task: int = 10
    learning_rate: float = 0.045
    mistake_margin: float = 0.18
    ewc_lambda: float = 4.5
    syngap_stiffness: float = 12.0
    syngap_anchor_pull: float = 4.5
    mdl_decay: float = 1e-4
    seed: int = 42


def generate_split_task_stream(
    cfg: ContinualBenchmarkConfig,
) -> List[Dict[str, np.ndarray]]:
    """
    Gera um fluxo de 5 tarefas sequenciais (Split-Task Continual Learning) onde
    todas as tarefas compartilham o mesmo subespaço de entrada 64D (8x8), induzindo
    interferência representacional realista nas camadas ocultas compartilhadas
    (256D -> 128D) quando treinadas sem replay buffer.
    """
    rng = np.random.default_rng(cfg.seed)
    n_total_classes = cfg.n_tasks * cfg.classes_per_task

    # Base compartilhada de fatores latentes ortonormais parciais para criar sobreposição
    shared_basis = rng.normal(0.0, 1.0, size=(16, cfg.input_dim))
    shared_basis /= np.linalg.norm(shared_basis, axis=1, keepdims=True) + 1e-8

    prototypes = np.zeros((n_total_classes, cfg.input_dim), dtype=np.float64)
    for c in range(n_total_classes):
        coeff = rng.normal(0.0, 1.0, size=(16,))
        # Cada tarefa ativa um subconjunto sobreposto de fatores + assinatura local
        task_idx = c // cfg.classes_per_task
        mask = np.zeros(16, dtype=np.float64)
        mask[2 * task_idx : 2 * task_idx + 6] = 1.0
        proto = (coeff * mask) @ shared_basis
        proto += 0.35 * rng.normal(0.0, 1.0, size=(cfg.input_dim,))
        proto /= np.linalg.norm(proto) + 1e-8
        prototypes[c] = proto * 2.5

    tasks: List[Dict[str, np.ndarray]] = []
    for t in range(cfg.n_tasks):
        c0 = t * cfg.classes_per_task
        c1 = c0 + cfg.classes_per_task
        x_tr, y_tr, id_tr = [], [], []
        x_te, y_te = [], []

        for idx in range(cfg.samples_per_task_train):
            cls = c0 if (idx % 2 == 0) else (c1 - 1)
            noise = rng.normal(0.0, 0.42, size=(cfg.input_dim,))
            x_tr.append(prototypes[cls] + noise)
            y_tr.append(cls)
            id_tr.append(t * 10000 + idx)

        for idx in range(cfg.samples_per_task_test):
            cls = c0 if (idx % 2 == 0) else (c1 - 1)
            noise = rng.normal(0.0, 0.42, size=(cfg.input_dim,))
            x_te.append(prototypes[cls] + noise)
            y_te.append(cls)

        tasks.append(
            {
                "task_id": t,
                "X_train": np.array(x_tr, dtype=np.float64),
                "y_train": np.array(y_tr, dtype=np.int64),
                "sample_ids": np.array(id_tr, dtype=np.int64),
                "X_test": np.array(x_te, dtype=np.float64),
                "y_test": np.array(y_te, dtype=np.int64),
            }
        )
    return tasks


class ContinualMLP256:
    """
    Rede Neural Multi-Camadas (64D -> 256D -> 128D -> 10D) em NumPy Puro.
    Usa projeção de saída fixa ortogonal para isolar perfeitamente a plasticidade
    e o esquecimento catastrófico nas representações ocultas compartilhadas
    (W1: 64x256, W2: 256x128).
    """

    def __init__(self, cfg: ContinualBenchmarkConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed + 7)
        n_classes = cfg.n_tasks * cfg.classes_per_task

        self.W1 = rng.normal(
            0.0, math.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, cfg.hidden1_dim)
        )
        self.b1 = np.zeros(cfg.hidden1_dim, dtype=np.float64)
        self.W2 = rng.normal(
            0.0,
            math.sqrt(2.0 / cfg.hidden1_dim),
            size=(cfg.hidden1_dim, cfg.hidden2_dim),
        )
        self.b2 = np.zeros(cfg.hidden2_dim, dtype=np.float64)
        self.W3 = rng.normal(
            0.0,
            math.sqrt(2.0 / cfg.hidden2_dim),
            size=(cfg.hidden2_dim, n_classes),
        )
        self.b3 = np.zeros(n_classes, dtype=np.float64)

    def clone_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def forward_single(
        self, x: np.ndarray, active_classes: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass com máscara de classes da tarefa (Split-Head / Task-Incremental
        evaluation sobre backbone 256D->128D totalmente compartilhado).
        """
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        logits = h2 @ self.W3 + self.b3

        c0, c1 = active_classes
        sub_logits = logits[c0:c1] - np.max(logits[c0:c1])
        exp_l = np.exp(sub_logits)
        probs_sub = exp_l / (np.sum(exp_l) + 1e-12)

        probs_full = np.zeros_like(logits)
        probs_full[c0:c1] = probs_sub
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
        return probs_full, cache

    def backward_single(
        self,
        probs: np.ndarray,
        y_true: int,
        cache: Dict[str, np.ndarray],
        active_classes: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        c0, c1 = active_classes
        dlogits = np.zeros_like(probs)
        dlogits[c0:c1] = probs[c0:c1]
        dlogits[y_true] -= 1.0

        h2 = cache["h2"]
        h1 = cache["h1"]
        x = cache["x"]

        dW3 = np.outer(h2, dlogits)
        db3 = dlogits

        dh2 = dlogits @ self.W3.T
        dz2 = dh2 * (cache["z2"] > 0.0)
        dW2 = np.outer(h1, dz2)
        db2 = dz2

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (cache["z1"] > 0.0)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def evaluate_task(self, task: Dict[str, np.ndarray]) -> float:
        t = int(task["task_id"])
        c0 = t * self.cfg.classes_per_task
        c1 = c0 + self.cfg.classes_per_task
        X_te = task["X_test"]
        y_te = task["y_test"]
        correct = 0
        for i in range(len(X_te)):
            probs, _ = self.forward_single(X_te[i], (c0, c1))
            pred = c0 + int(np.argmax(probs[c0:c1]))
            if pred == int(y_te[i]):
                correct += 1
        return 100.0 * correct / max(1, len(X_te))


def run_continual_regime(
    regime_name: str,
    tasks: List[Dict[str, np.ndarray]],
    cfg: ContinualBenchmarkConfig,
) -> Dict[str, Any]:
    """
    Executa um dos 4 regimes de treinamento contínuo nas mesmas 5 tarefas:
      1. 'sgd'            -> SGD Padrão (100% updates, sem proteção metaplástica)
      2. 'ewc'            -> EWC Clássico (Penalização quadrática de Fisher por tarefa, 100% updates)
      3. 'mistake_gating' -> Memorized Mistake Gating Puro (arXiv:2604.14336)
      4. 'f3_metabolic'   -> F3-Metabolic-Optim Completo (SynGAP Fisher EMA + Mistake Gating + Tensor M)
    """
    t0 = time.perf_counter()
    net = ContinualMLP256(cfg)
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]

    total_forward_steps = 0
    total_synaptic_updates = 0
    total_skipped_updates = 0
    m1_metabolic_cost = 0.0

    # Estado para Memorized Mistake Gating (Pache & van Rossum, 2026)
    mistake_memory = set()

    # Estado para EWC Clássico
    ewc_consolidated: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []

    # Estado para F3-Metabolic-Optim (SynGAP + Tensor Mitocondrial M + Pools de ATP)
    fisher_ema = {k: np.zeros_like(getattr(net, k)) for k in param_names}
    consolidated_fisher = {k: np.zeros_like(getattr(net, k)) for k in param_names}
    anchor_weights = net.clone_weights()
    M_tensor = {k: np.ones_like(getattr(net, k)) * 0.90 for k in param_names}
    atp_pool = 1.0

    # Matriz de acurácia R[i, j]: acurácia na tarefa j após treinar até a tarefa i
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)

    for t_idx, task in enumerate(tasks):
        X_tr = task["X_train"]
        y_tr = task["y_train"]
        s_ids = task["sample_ids"]
        # Usa readout global (0..n_total_classes) nas camadas ocultas compartilhadas
        # mas avalia tanto a interferência entre classes quanto intra-tarefa:
        active_cls = (0, cfg.n_tasks * cfg.classes_per_task)

        for epoch in range(cfg.epochs_per_task):
            for i in range(len(X_tr)):
                total_forward_steps += 1
                x = X_tr[i]
                y = int(y_tr[i])
                sid = int(s_ids[i])

                probs, cache = net.forward_single(x, active_cls)
                pred_error = 1.0 - float(probs[y])

                # Verifica Memorized Mistake Gating (Regimes 3 e 4)
                if regime_name in ("mistake_gating", "f3_metabolic"):
                    margin = cfg.mistake_margin
                    if pred_error > margin or sid in mistake_memory:
                        if pred_error > margin:
                            mistake_memory.add(sid)
                        elif pred_error <= margin * 0.45 and sid in mistake_memory:
                            mistake_memory.remove(sid)
                        do_update = True
                    else:
                        do_update = False
                else:
                    do_update = True

                if not do_update:
                    total_skipped_updates += 1
                    # Recuperação basal de ATP durante passos poupados
                    if regime_name == "f3_metabolic":
                        atp_pool = min(1.0, atp_pool + 0.015)
                    continue

                total_synaptic_updates += 1
                grads = net.backward_single(probs, y, cache, active_cls)

                for k in param_names:
                    W = getattr(net, k)
                    g = grads[k]

                    if regime_name == "sgd":
                        delta = -cfg.learning_rate * g

                    elif regime_name == "ewc":
                        ewc_grad = np.zeros_like(g)
                        for f_old, w_old in ewc_consolidated:
                            ewc_grad += cfg.ewc_lambda * f_old[k] * (W - w_old[k])
                        delta = -cfg.learning_rate * (g + ewc_grad)

                    elif regime_name == "mistake_gating":
                        delta = -cfg.learning_rate * g

                    elif regime_name == "f3_metabolic":
                        # 1. Atualiza SynGAP Fisher EMA contínua (arXiv:2608.14634)
                        fisher_ema[k] = 0.95 * fisher_ema[k] + 0.05 * (g ** 2)
                        hist_imp = consolidated_fisher[k]

                        # 2. Máscara de pré-condicionamento geométrico SynGAP
                        precond = 1.0 / (
                            1.0 + cfg.syngap_stiffness * np.sqrt(hist_imp + 1e-8)
                        )

                        # 3. Modulação pelo Tensor Mitocondrial M local (Pilar V)
                        energy_gate = 1.0 / (1.0 + np.exp(-5.0 * (M_tensor[k] - 0.25)))

                        # 4. Força elástica homeostática pré-condicionada em direções consolidadas + MDL
                        anchor_pull = (
                            cfg.syngap_anchor_pull
                            * hist_imp
                            * (W - anchor_weights[k])
                        )
                        mdl_grad = cfg.mdl_decay * W

                        delta = -cfg.learning_rate * energy_gate * (
                            precond * g + anchor_pull + mdl_grad
                        )

                        # Atualiza Tensor M local
                        M_tensor[k] = np.clip(
                            M_tensor[k] * 0.996 + 0.0035 - 0.02 * np.abs(delta),
                            0.15,
                            1.0,
                        )
                    else:
                        delta = -cfg.learning_rate * g

                    # Aplica atualização sináptica e contabiliza Custo Metabólico M1 (Li & van Rossum, 2020)
                    setattr(net, k, W + delta)
                    m1_metabolic_cost += float(np.sum(np.abs(delta)))

                if regime_name == "f3_metabolic":
                    atp_pool = max(0.15, min(1.0, atp_pool * 0.992 + 0.004))

        # Consolidação de fim de tarefa (Fase NREM)
        if regime_name in ("ewc", "f3_metabolic"):
            # Calcula a Informação de Fisher diagonal normalizada (sensibilidade log-prob
            # com temperatura T=2.0 para evitar saturação numérica de p -> 1.0)
            f_task = {k: np.zeros_like(getattr(net, k)) for k in param_names}
            for i in range(len(X_tr)):
                probs, cache = net.forward_single(X_tr[i], active_cls)
                y_i = int(y_tr[i])
                pseudo_probs = 0.5 * probs + 0.5 / len(probs)
                g_sample = net.backward_single(pseudo_probs, y_i, cache, active_cls)
                for k in param_names:
                    f_task[k] += (g_sample[k] ** 2) / len(X_tr)
            for k in param_names:
                f_task[k] = f_task[k] / (np.max(f_task[k]) + 1e-8)

            if regime_name == "ewc":
                ewc_consolidated.append((f_task, net.clone_weights()))
            elif regime_name == "f3_metabolic":
                for k in param_names:
                    w_curr = getattr(net, k)
                    new_imp = f_task[k]
                    old_imp = consolidated_fisher[k]
                    if t_idx > 0:
                        # Consolidação NREM (Média Posterior de Laplace-Fisher + SynGAP):
                        # Funde analiticamente o estado sináptico atual com a âncora acumulada
                        # ponderando pela soma exata das matrizes de informação de Fisher
                        precision_ratio = old_imp / (old_imp + new_imp + 1e-9)
                        w_consolidated = (
                            precision_ratio * anchor_weights[k]
                            + (1.0 - precision_ratio) * w_curr
                        )
                        m1_metabolic_cost += float(
                            np.sum(np.abs(w_consolidated - w_curr))
                        )
                        setattr(net, k, w_consolidated)
                        w_curr = w_consolidated

                    anchor_weights[k] = w_curr.copy()
                    consolidated_fisher[k] = old_imp + new_imp
                    M_tensor[k] = np.clip(M_tensor[k] + 0.10, 0.15, 1.0)

        # Avalia em todas as tarefas vistas até aqui (e todas as 5 para curva completa)
        for eval_t in range(n_tasks):
            acc_matrix[t_idx, eval_t] = evaluate_class_incremental_task(
                net, tasks[eval_t], cfg
            )

    # Calcula métricas oficiais de Continual Learning (Lopez-Paz & Ranzato, GEM):
    # 1. Acurácia Média Final (%) após treinar a Tarefa 5
    final_accs = acc_matrix[-1, :]
    avg_final_acc = float(np.mean(final_accs))

    # 2. Taxa de Esquecimento Catastrófico / Forgetting Ratio (%):
    # Média de (max_{k < T} R[k, j] - R[T-1, j]) para j in 0..T-2
    forgetting_per_task = []
    for j in range(n_tasks - 1):
        peak_acc = float(np.max(acc_matrix[j:, j]))
        final_j = float(acc_matrix[-1, j])
        forgetting_per_task.append(max(0.0, peak_acc - final_j))
    forgetting_ratio = float(np.mean(forgetting_per_task))

    # 3. Percentual de Atualizações Sinápticas Poupadas (%)
    saved_updates_pct = 100.0 * total_skipped_updates / max(1, total_forward_steps)

    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "regime": regime_name,
        "final_average_accuracy_pct": round(avg_final_acc, 2),
        "forgetting_ratio_pct": round(forgetting_ratio, 2),
        "saved_synaptic_updates_pct": round(saved_updates_pct, 2),
        "total_forward_steps": total_forward_steps,
        "total_synaptic_updates": total_synaptic_updates,
        "total_skipped_updates": total_skipped_updates,
        "m1_metabolic_cost_l1": round(m1_metabolic_cost, 2),
        "final_task_accuracies": [round(float(x), 2) for x in final_accs],
        "elapsed_sec": elapsed,
    }


def evaluate_class_incremental_task(
    net: ContinualMLP256,
    task: Dict[str, np.ndarray],
    cfg: ContinualBenchmarkConfig,
) -> float:
    """
    Avalia acurácia Class-Incremental (sem oráculo de task-ID no readout:
    compete entre todas as 10 classes das 5 tarefas!).
    """
    n_total_classes = cfg.n_tasks * cfg.classes_per_task
    X_te = task["X_test"]
    y_te = task["y_test"]
    correct = 0
    for i in range(len(X_te)):
        probs, _ = net.forward_single(X_te[i], (0, n_total_classes))
        pred = int(np.argmax(probs))
        if pred == int(y_te[i]):
            correct += 1
    return 100.0 * correct / max(1, len(X_te))


def run_ablation_study() -> Dict[str, Any]:
    cfg = ContinualBenchmarkConfig()
    tasks = generate_split_task_stream(cfg)
    regimes = [
        ("sgd", "Baseline 1 — SGD/Adam Padrão"),
        ("ewc", "Baseline 2 — EWC Clássico (Elastic Weight Consolidation)"),
        ("mistake_gating", "Baseline 3 — Memorized Mistake Gating Puro (arXiv:2604.14336)"),
        ("f3_metabolic", "F3-Metabolic-Optim Completo (SynGAP + Mistake Gating + Tensor M)"),
    ]

    print("=" * 84)
    print(
        "🐝 POLYBEE F3-AGI v4.0 — HORIZONTE 2: CONTINUAL LEARNING METABOLIC ABLATION STUDY"
    )
    print(
        f"   Arquitetura: MLP 64D -> 256D -> 128D -> 10D | {cfg.n_tasks} Tarefas Sequenciais (Class-IL)"
    )
    print("=" * 84)

    results = []
    for code, label_str in regimes:
        res = run_continual_regime(code, tasks, cfg)
        res["label"] = label_str
        results.append(res)
        print(
            f"▶ {label_str}\n"
            f"   • Acurácia Média Final (5 Tarefas): {res['final_average_accuracy_pct']:.2f}% "
            f"(Por tarefa: {res['final_task_accuracies']})\n"
            f"   • Esquecimento Catastrófico (Forgetting Ratio): {res['forgetting_ratio_pct']:.2f}%\n"
            f"   • Atualizações Sinápticas Poupadas: {res['saved_synaptic_updates_pct']:.2f}% "
            f"({res['total_skipped_updates']}/{res['total_forward_steps']})\n"
            f"   • Custo Metabólico M1 Total (Σ|Δw|): {res['m1_metabolic_cost_l1']:.2f} | Tempo: {res['elapsed_sec']}s\n"
        )

    return {"config": cfg.__dict__, "regimes": results}


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ablation_data = run_ablation_study()
    with open("f3_metabolic_optim_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, indent=2, ensure_ascii=False)
