---
name: f3-aacr-solver
description: >-
  Executa o Ciclo Criativo AACR v4.0 (Acúmulo, Amplificação, Colapso e Recuperação SOAR)
  do framework Polybee F3-AGI v4.0 para resolver tarefas do ARC-AGI-1 e ARC-AGI-2
  usando exclusivamente o Google Antigravity (Agente Nativo, Subagents, SDK ou CLI).
  Ative esta skill quando o usuário pedir para analisar, resolver ou depurar tarefas estáticas
  do ARC-AGI ou gerar traços evolutivos para o buffer NREM (soar_nrem_traces.json).
---

# Skill: F3-AGI v4.0 AACR Neuro-Symbolic Solver (Antigravity Native)

Esta skill orienta o Agente Antigravity (e seus Subagents) a executar o **Ciclo AACR v4.0** completo sobre qualquer tarefa do **ARC-AGI-1** ou **ARC-AGI-2**, atuando diretamente como o módulo **DreamerV40 (Sistema 1/2)** acoplado ao verificador **RealistV40**, ao **VisualCortexV40** e ao **MetabolicControllerV40**.

## 1. Fluxo de Execução Passo a Passo

### Passo 1: Extração do Grafo de Cena Relacional (`VisualCortexV40`)
Para inspecionar uma tarefa JSON do ARC-AGI e gerar o prompt estruturado com invariantes topológicos e todas as matrizes de treino (sem truncamento):

```powershell
python -c "import json; from f3_agi_v4_0_core import VisualCortexV40, F3Primitives; t = json.load(open('CAMINHO_DA_TAREFA.json')); vc = VisualCortexV40(); a = vc.analyze_task(t['train']); print(F3Primitives.get_tool_docs()); print(vc.format_for_prompt(a, t['train']))"
```

### Passo 2: Fase 1 (Acúmulo) e Fase 2 (Amplificação Evolutiva)
1. Gere de **2 a 4 hipóteses candidatas** na forma de funções Python `def transform(grid: np.ndarray) -> np.ndarray:` utilizando prioritariamente as 12 primitivas seguras de `F3Primitives`:
   - `F3Primitives.get_background_color(grid)`
   - `F3Primitives.crop_to_content(grid, background_color=-1)`
   - `F3Primitives.get_objects_by_color(grid, color)`
   - `F3Primitives.get_connected_objects(grid, ignore_bg=True)`
   - `F3Primitives.get_largest_object(grid, ignore_bg=True)`
   - `F3Primitives.replace_color(grid, old, new)`
   - `F3Primitives.resize_grid(grid, target_h, target_w, fill_value=0)`
   - `F3Primitives.scale_up(grid, factor_h, factor_w)`
   - `F3Primitives.downsample_block(grid, target_h, target_w)`
   - `F3Primitives.tile_grid(grid, reps_h, reps_w)`
   - `F3Primitives.split_and_overlay(grid, axis=0, n_parts=2, op='or', sep_size=1)`
   - `F3Primitives.detect_symmetries(grid)`
2. Avalie cada hipótese com `RealistV40.evaluate(h, train_pairs)` para obter:
   - `h.score` (0.0 a 1.0),
   - `h.free_energy` (Energia Livre Estrutural),
   - `h.shape_accuracy`, `h.pixel_accuracy`, `h.object_accuracy`, `h.mdl_cost`, e
   - `h.residual_feedback` (coordenadas e pares exatos que falharam).
3. Se `h.score < 1.0`, execute a **Fase 2 (Amplificação)**: use o código de maior score (`best_previous`) e seu `residual_feedback` para realizar mutação dirigida até zerar o erro residual em todos os pares de treino (`h.score == 1.0`).
4. Para tarefas complexas em lote, invoque subagentes paralelos via `invoke_subagent` (com `TypeName="self"` e `Model="inherit"`) para explorar ramos de hipóteses concorrentes (simulando busca em árvore no Global Workspace).

### Passo 3: Fase 3 (Colapso) e Fase 4 (Recuperação NREM / SOAR)
Quando uma hipótese atingir `score == 1.0` nos pares de treino:
1. Valide na matriz de `test` da tarefa.
2. Registre o traço evolutivo (`initial_best` -> `residual_feedback` -> `winner`) no `NREMConsolidationBuffer` filtrado por **Memorized Mistake Gating** e salve em `soar_nrem_traces.json`.

## 2. Execução via Linha de Comando Unificada

- **Teste de verificação instantâneo (5 tarefas do benchmark de regressão)**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend mock --suite arc1
  ```
- **Execução com Antigravity SDK / CLI / IPC (`gemini-4-argon`)**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --suite arc1
  ```
- **Execução em um arquivo JSON específico do ARC-AGI**:
  ```powershell
  python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --task-file caminho/para/tarefa.json --export-soar soar_nrem_traces.json
  ```
