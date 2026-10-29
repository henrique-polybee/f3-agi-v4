# Diretrizes Arquiteturais e Científicas — Polybee F3-AGI v4.0 (Outubro 2026)

Este workspace contém a implementação oficial do **Framework F3-AGI v4.0** da **Polybee AI Research**, configurado para operar **exclusivamente com o ecossistema Google Antigravity** em máquina local de 16GB de RAM (CPU-first, `<400 MB` de RAM).

---

## 1. Arquitetura de Cinco Níveis do F3-AGI v4.0

Toda evolução de código neste repositório deve preservar os cinco níveis validados na *Avaliação Científica e Roadmap F3-AGI (Outubro 2026)*:

1. **Nível 1 — Multi-Scale Recursive Core (MSRC)**:
   - Inspirado em *HRM* (Wang et al., 2025/2026) e *TRM* (Jolicoeur-Martineau, 2025).
   - Recorrência em duas escalas temporais (módulo rápido perceptivo + módulo lento abstrato) com dimensões padronizadas em potências de 2 (`256D` ou `512D`).
2. **Nível 1.5 — Metabolic & MDL Controller (`MetabolicControllerV40`)**:
   - Une o Tensor Mitocondrial (`M_tensor`), Pools de ATP Virtual (*SNEP*, Burroni et al., 2017), **Memorized Mistake Gating** (Pache & van Rossum, jul. 2026 — `arXiv:2604.14336`) e penalidade de **Comprimento Mínimo de Descrição (MDL)** (*CompressARC*, Liao & Gu, 2025 — `arXiv:2512.06104`).
   - Atualizações sinápticas e consolidação NREM só ocorrem se o erro de predição violar `mistake_margin` ou se a amostra constar em `mistake_memory`.
3. **Nível 2 — SynGAP + Titans Memory**:
   - Implementa o pré-condicionamento geométrico metaplástico via média móvel exponencial da Informação de Fisher diagonal (**SynGAP**, Aguilar et al., jul. 2026 — `arXiv:2608.14634`) modulado pelo Tensor Mitocondrial `M`.
4. **Nível 3 — SIGReg-JEPA + Active Inference (`VisualCortexV40`, `DreamerV40`, `RealistV40`)**:
   - Unifica modelos de mundo latentes isotrópicos (**SIGReg-JEPA**, Arnez & Gomez-Villa, jul. 2026 — `arXiv:2607.13612`) com Inferência Ativa e síntese evolutiva de programas (**SOAR**, Pourcel et al., 2025 — `arXiv:2507.14172`).
   - Minimiza a **Energia Livre Variacional Bioenergeticamente Restrita** ($F_{F3}$) na síntese estática (ARC-AGI-1/2) e a **Energia Livre Esperada Completa** ($G_{F3}$, incluindo Valor Pragmático + Valor Epistêmico de Cobertura de Estados via `PairDist` + Custo Energético de Ação $c_{\text{ATP}}$) na exploração interativa (ARC-AGI-3).
5. **Nível 4 — Continuous Global Workspace (`AlgebraicWorldModelSLS` + Diagnóstico Metabólico + `NREMConsolidationBuffer`)**:
   - Espaço Latente Compartilhado (`SLS 512D`), Notação Algébrica Compacta estilo *Astra*, evicção contínua de contexto estilo *The Duck* (Tufa Labs) e auditoria homeostática contra **Rigidez**, **Mania** e **Depressão (Dark Room)**.

---

## 2. Regras de Poda Científica (Navalha de Occam)

- **Proibido usar $\phi = 1,618$ ( Fibonacci / Razão Áurea)** para dimensionar tensores ou camadas neurais. Use sempre potências de 2 (`256D`, `512D`) e recorrência multi-escala.
- **Proibido acoplar Redes Espiculadas (SNNs LIF)** ou cálculo de **Expoente de Hurst via R/S em janelas curtas** em tarefas cognitivas abstratas (ARC-AGI). Use sempre **SynGAP** (Fisher EMA) + **Memorized Mistake Gating** + surpresa variacional.
- **Proibido truncar exemplos de treinamento** (`train_examples[:2]` é um bug extinto do v3.9.5). `VisualCortexV40` e `RealistV40` devem sempre inspecionar **todos** os pares de demonstração da tarefa.
- **Proibido avaliar hipóteses apenas por acurácia bruta de pixels**. Toda avaliação no `RealistV40` deve combinar:
  1. `shape_accuracy` (divergência de dimensão),
  2. `pixel_accuracy` ponderada entre *foreground* e *background* (evitando a armadilha de pintar tudo da cor de fundo),
  3. `object_accuracy` (Jaccard de cores + similaridade de contagem de componentes conectados),
  4. `mdl_cost` (penalidade de concisão Kolmogorov/MDL), e
  5. `residual_feedback` (diff estruturado com shapes e coordenadas exatas de erro injetado no ciclo seguinte do AACR via `best_previous`).

---

## 3. Política de Execução Exclusiva com Google Antigravity

- **Ambiente Local Leve (`<400 MB` RAM)**:
  - Não adicione dependências pesadas de GPU local (`torch` com CUDA, `bitsandbytes`, `transformers` pesados) ao fluxo padrão do notebook de 16GB de RAM.
  - O motor local ([f3_agi_v4_0_core.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_0_core.py) e [f3_agi_v4_arc3_runner.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_arc3_runner.py)) deve rodar 100% em CPU apenas com `numpy` (e `scipy` opcional, pois o core já possui fallback em NumPy puro para `label`, `find_objects` e `center_of_mass`).
- **Backends Autorizados**:
  - Use exclusivamente o ecossistema **Google Antigravity** (`--backend antigravity` com modelos `gemini-4-argon`, `gemini-3.8-flash`, `claude-sonnet-5.5` ou `claude-opus-5.5`, `--backend gemini_api` ou `--backend mock` para testes determinísticos).
  - Não utilize nem reintroduza integrações com OpenAI Codex CLI (`@openai/codex`).
  - Quando o usuário pedir para resolver tarefas do ARC-AGI diretamente aqui no chat do Antigravity, utilize as skills `.agents/skills/f3-aacr-solver` e `.agents/skills/f3-arc3-harness` (ou lance Subagents via `invoke_subagent`) operando sobre a bridge `.antigravity_ipc/` ou diretamente via script Python.
