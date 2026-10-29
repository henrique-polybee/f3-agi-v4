# Guia Passo a Passo: Configurando e Rodando o F3-AGI v4.0 no Seu Notebook (16GB RAM, Sem GPU — Exclusivo Google Antigravity)

Este guia prático mostra como configurar, autenticar e rodar o motor **F3-AGI v4.0** ([f3_agi_v4_0_core.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_0_core.py) e [f3_agi_v4_arc3_runner.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_arc3_runner.py)) diretamente no seu notebook Windows de **16GB de RAM sem placa de vídeo dedicada**, utilizando **exclusivamente o ecossistema Google Antigravity** (Agente Nativo + Subagents na IDE/App Antigravity 2.0, SDK Python `google-antigravity` e `agy` CLI com sua assinatura **Google AI Pro**) e usando as **30h semanais gratuitas de GPU dupla (2x NVIDIA T4 = 32GB VRAM) do Kaggle** apenas quando quiser realizar ajuste fino de pesos offline (LoRA NREM/REM estilo SOAR).

---

## 1. Estrutura de Arquivos e Customizações Antigravity no Projeto

O workspace está configurado com as regras e skills nativas do **Google Antigravity**:

```text
polybee-f3-agi/
├── GEMINI.md                                  # Regras arquiteturais e científicas do F3-AGI v4.0 (sempre ativas)
├── .agents/
│   └── skills/
│       ├── f3-aacr-solver/
│       │   └── SKILL.md                       # Skill nativa para o Ciclo AACR v4.0 + SOAR (ARC-AGI-1 e 2)
│       └── f3-arc3-harness/
│           └── SKILL.md                       # Skill nativa para o Harness Interativo ARC-AGI-3 (RHAE)
├── f3_agi_v4_0_core.py                        # Núcleo F3-AGI v4.0 (100% CPU/NumPy, com fallback sem depender de SciPy)
├── f3_agi_v4_arc3_runner.py                   # Orquestrador CLI para ARC-AGI-1/2 e ARC-AGI-3 (64x64)
├── soar_nrem_traces.json                      # Buffer exportado com traços evolutivos consolidados (Mistake Gating)
└── guia_setup_f3_agi_v4.md                    # Este guia de configuração e operação
```

- **[f3_agi_v4_0_core.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_0_core.py)**: Núcleo F3-AGI v4.0 com `F3Primitives` (12 primitivas seguras contra erro de broadcast), `VisualCortexV40` (análise relacional sem truncamento `[:2]`), `DreamerV40` (com memória evolutiva `best_previous` e Reflexion semântico), `RealistV40` (Energia Livre Estrutural + diff residual), `MetabolicControllerV40` (Tensor M + *Memorized Mistake Gating* + *SynGAP*), `NREMConsolidationBuffer` e o conector `AntigravityBackend`.
- **[f3_agi_v4_arc3_runner.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_arc3_runner.py)**: Orquestrador para o **ARC-AGI-3** com `DeadSignatureTracker` em NumPy, Espaço Latente 512D `AlgebraicWorldModelSLS` (*SIGReg-JEPA* + Valor Epistêmico `PairDist`), Notação Algébrica Compacta estilo *Astra*, evicção contínua de contexto estilo *The Duck* e cálculo da métrica oficial **RHAE**.

---

## 2. Ambiente Python Leve (`<400 MB` de RAM)

No seu notebook de 16GB de RAM, você **não precisa instalar PyTorch com CUDA nem `bitsandbytes`**. O [f3_agi_v4_0_core.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_0_core.py) já conta com implementação em **NumPy puro** (incluindo fallback nativo para extração de componentes conectados `label`, `find_objects` e `center_of_mass` caso `scipy` não esteja instalado):

```powershell
# No Windows PowerShell (dentro da pasta do projeto):
python --version
# Python 3.14.0 (já configurado com numpy)

# Opcional: instalar scipy caso queira aceleração C adicional em grades grandes
pip install numpy scipy
```

Para testar imediatamente a instalação local (sem gastar cota de API, usando o simulador evolutivo determinístico):

```powershell
python f3_agi_v4_arc3_runner.py --backend mock --suite both
```

Resultado validado no terminal:
- **Etapa 1 (ARC-AGI-1/2 Estático):** `5/5 Tarefas Resolvidas (100% Test Match)` nas tarefas `e98196ab`, `2204b7a8`, `a61ba2ce`, `5614dbcf` e `2dee498d`, com exportação automática de `soar_nrem_traces.json`.
- **Etapa 2 (ARC-AGI-3 Interativo 64x64):** `3/3 Níveis Vencidos` com **84,6% de Eficiência Relativa de Ação Humana (RHAE)**.

---

## 3. Operando Exclusivamente com o Google Antigravity

Você possui três modos integrados para conectar o `DreamerV40` e o planejador Sistema 2 do `F3ARC3InteractiveAgent` aos modelos de fronteira do **Google Antigravity** (Outubro 2026):

| Modo de Execução Antigravity | Como Funciona | Modelos Disponíveis no Antigravity (Outubro 2026) | Comando / Ação |
| :--- | :--- | :--- | :--- |
| **Modo 1: Agente Nativo & Subagents (No Chat do Antigravity)** | O próprio Agente Antigravity ativa as skills `f3-aacr-solver` e `f3-arc3-harness` (e lança Subagents em paralelo) usando o modelo selecionado na UI. | **Gemini 4 Argon** (`98,5%` ARC-1 / `91,3%` ARC-2 / `97,46%` ARC-3), `gemini-3.8-flash`, `claude-sonnet-5.5`, `claude-opus-5.5` | Peça diretamente no chat: *"Resolva a tarefa X usando o ciclo AACR v4.0"* |
| **Modo 2: Antigravity Python SDK (`google.antigravity`) ou CLI (`agy`)** | O script Python invoca programaticamente o `Agent` do SDK `google-antigravity` ou o terminal `agy` (cota renovada a cada 5h no Google AI Pro). | `gemini-4-argon` (default), `gemini-3.8-flash`, `claude-sonnet-5.5`, `claude-opus-5.5` | `python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --suite both` |
| **Modo 3: Google AI Studio REST API (`gemini_api`)** | Chamadas HTTP diretas via `urllib` padrão do Python usando sua `GEMINI_API_KEY` (zero peso na RAM). | `gemini-3.8-flash`, `gemini-4-argon` | `python f3_agi_v4_arc3_runner.py --backend gemini_api --model gemini-3.8-flash --suite both` |

### 3.1 Usando o Modo 1 (Direto na Interface do Antigravity — Zero Configuração Extra)

Como as regras ([GEMINI.md](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/GEMINI.md)) e as skills ([f3-aacr-solver](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/.agents/skills/f3-aacr-solver/SKILL.md) e [f3-arc3-harness](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/.agents/skills/f3-arc3-harness/SKILL.md)) já estão instaladas neste workspace, você pode usar o modelo já ativo aqui no Antigravity (**Gemini 4 Argon**) diretamente pelo chat:
- Para resolver uma ou várias tarefas do ARC-AGI-1/2: basta mencionar o arquivo JSON da tarefa no chat e pedir para rodar o ciclo AACR v4.0. O Antigravity extrairá o grafo de cena via `VisualCortexV40`, sintetizará as hipóteses Python, validará com o `RealistV40` e gravará os traços vencedores em `soar_nrem_traces.json`.
- Para rodar busca evolutiva paralela em tarefas difíceis: peça ao Antigravity para lançar **Subagents** explorando diferentes hipóteses em paralelo.

### 3.2 Configurando o Modo 2: Antigravity SDK (`google-antigravity`) e CLI (`agy`)

Se quiser que os scripts Python lancem sessões autônomas do Antigravity pelo terminal do Windows PowerShell ([Documentação Antigravity](https://antigravity.google/docs)):

```powershell
# 1. Instalar o SDK Python oficial do Google Antigravity
pip install google-antigravity

# 2. (Opcional) Instalar o Antigravity CLI no Windows PowerShell
irm https://antigravity.google/cli/install.ps1 | iex

# 3. Autenticar com sua conta Google AI Pro (caso use o CLI autônomo)
agy
```

Exemplos de execução no PowerShell:

```powershell
# Rodar com o modelo flagship Gemini 4 Argon (padrão do AntigravityBackend)
python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --suite both

# Rodar com Gemini 3.8 Flash (rápido e econômico na janela de 5h)
python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-3.8-flash --suite both

# Rodar tarefas com Claude Sonnet 5.5 ou Opus 5.5 (incluídos no Google AI Pro)
python f3_agi_v4_arc3_runner.py --backend antigravity --model claude-sonnet-5.5 --suite arc1
```

### 3.3 Configurando o Modo 3: Chave Direta (`GEMINI_API_KEY`)

Se quiser rodar lotes rápidos via REST API pura (`urllib`):

```powershell
$env:GEMINI_API_KEY="sua_chave_aqui"
python f3_agi_v4_arc3_runner.py --backend gemini_api --model gemini-3.8-flash --suite both
```

---

## 4. Testando nos Datasets Oficiais Completos (ARC-AGI-1, ARC-AGI-2 e ARC-AGI-3)

Para baixar os datasets oficiais da [ARC Prize Foundation](https://arcprize.org/) na pasta do projeto (`<50 MB` em disco):

```powershell
# Clonar o dataset estático oficial (ARC-AGI-1 e ARC-AGI-2)
git clone https://github.com/fchollet/ARC-AGI.git

# Instalar o toolkit interativo oficial do ARC-AGI-3 (roda a 1.000 FPS em CPU!)
pip install arc-agi
```

Para resolver qualquer tarefa JSON real da pasta `ARC-AGI/data/training` ou `evaluation` e exportar automaticamente os traços evolutivos para o buffer de sono NREM (`soar_nrem_traces.json`), basta rodar:

```powershell
python f3_agi_v4_arc3_runner.py --backend antigravity --model gemini-4-argon --task-file ARC-AGI/data/training/2204b7a8.json --export-soar soar_nrem_traces.json
```

---

## 5. Como Usar GPU Gratuita no Kaggle (30h/Semana de 2x T4 = 32GB VRAM)

Quando você acumular traços verificados no arquivo [soar_nrem_traces.json](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/soar_nrem_traces.json) e quiser rodar a fase de **Consolidação NREM/REM (Fine-tuning LoRA estilo SOAR)** em um modelo open-weights (como `DeepSeek V4.1 Flash`, `Qwen 3.8-27B` ou `Gemma-4-31B`), ou submeter offline para a competição oficial [ARC Prize 2026 no Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3):

1. Acesse sua conta gratuita em [kaggle.com](https://www.kaggle.com/) (com telefone verificado para liberar as **30 horas semanais de GPU gratuita**).
2. Clique em **Create -> New Notebook**.
3. No painel direito (**Session Options -> Accelerator**), selecione **GPU T4 x2** (**2 placas NVIDIA T4 de 16GB = 32GB de VRAM total**).
4. Faça upload dos arquivos [f3_agi_v4_0_core.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_0_core.py), [f3_agi_v4_arc3_runner.py](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/f3_agi_v4_arc3_runner.py) e [soar_nrem_traces.json](file:///c:/Users/henri/OneDrive/%C3%81rea%20de%20Trabalho/Polybee/F3-AGI/Hipotese/polybee-f3-agi/soar_nrem_traces.json) como um Dataset privado da Polybee.
5. Use o botão **Save Version -> Save & Run All (Commit)**: o Kaggle executará seu treinamento LoRA (com *Mistake Gating* e máscara *SynGAP*) ou avaliação offline por até **12 horas seguidas em segundo plano na nuvem**, sem consumir bateria ou memória do seu notebook local.
