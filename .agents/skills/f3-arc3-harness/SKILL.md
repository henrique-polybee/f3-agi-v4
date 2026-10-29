---
name: f3-arc3-harness
description: >-
  Executa, audita e estende o harness interativo do F3-AGI v4.0 para o benchmark ARC-AGI-3
  (grades 64x64 baseadas em turnos, sem instruções, avaliadas por RHAE).
  Ative esta skill quando o usuário pedir para rodar ambientes interativos do ARC-AGI-3,
  ajustar o DeadSignatureTracker, o modelo de mundo AlgebraicWorldModelSLS (512D SIGReg-JEPA)
  ou otimizar a Eficiência Relativa de Ação Humana (RHAE).
---

# Skill: F3-AGI v4.0 Interactive Harness para ARC-AGI-3

Esta skill define o protocolo de operação e evolução do agente interativo `F3ARC3InteractiveAgent` em `f3_agi_v4_arc3_runner.py` para o benchmark **ARC-AGI-3**.

## 1. Componentes do Harness Interativo

1. **`DeadSignatureTracker` (NumPy, Zero GPU)**:
   - Implementa a memória de *Dead-Signature* (Reki & Forge) acoplada ao *Memorized Mistake Gating* (Pache & van Rossum, 2026).
   - Hash rápida de frames (`blake2b` de 8 bytes) que bloqueia pares `(state_hash, action)` improdutivos e assinaturas de objetos `(color, size)` cujos cliques falharam $\ge 2$ vezes sem jamais alterar o mundo.
2. **`AlgebraicWorldModelSLS` (SLS 512D SIGReg-JEPA + Notação Algébrica + Evicção)**:
   - Codifica cada grade $64 \times 64$ em um vetor latente isotrópico $\mathbf{z}_t \in \mathbb{R}^{512}$ e calcula o **Valor Epistêmico de Cobertura de Estados** (`compute_state_epistemic_value` via `PairDist` — Proposição 6 de Arnez & Gomez-Villa, 2026).
   - Comprime o frame em uma linha de **Notação Algébrica Compacta** (`L1|T2|Objs:[...]|Changed:... <= Act:...`) e mantém janela deslizante de `max_history_turns=8` (*Infinite Play via Eviction*).
3. **Seleção de Ação por Energia Livre Esperada ($G_{F3}$)**:
   - **Sistema 1 (Rápido / NumPy)**: Explora cliques em objetos raros/pequenos quando há bloqueio causal e navega até o portal/meta minimizando distância pragmática + risco de colisão.
   - **Sistema 2 (Lento / Antigravity LLM)**: Acionado sob demanda para planejar filas de 1 a 4 ações (`action_queue`) sobre a Notação Algébrica Compacta, economizando até 80% de chamadas de modelo e maximizando o score **RHAE**.

## 2. Comandos de Execução e Verificação

- **Rodar a suíte interativa local ARC-AGI-3 (Meta: 3/3 níveis vencidos e RHAE $\ge 80\%$)**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend mock --suite arc3
  ```
- **Rodar com Planejador Sistema 2 via Antigravity (`gemini-4-argon`)**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --suite arc3
  ```
- **Rodar a bateria completa (Estático ARC-1/2 + Interativo ARC-3)**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend mock --suite both
  ```
