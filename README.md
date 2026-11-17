# Detecção de Fraude em Cartão de Crédito

Projeto de Machine Learning desenvolvido para identificar transações fraudulentas em uma base real altamente desbalanceada, priorizando métricas adequadas ao negócio e explicabilidade das decisões com SHAP.

## 1. O Problema e o Impacto do Desbalanceamento
Na base analisada (284.807 transações originais), **99,83% são transações legítimas** (284.315) e apenas **0,17% são fraudes** (492). Nesse cenário, a acurácia é enganosa: um modelo que classificasse todas as operações como normais atingiria 99,83% de acurácia sem detectar nenhuma fraude.

Por isso, a avaliação foi direcionada exclusivamente para a **classe de fraude (`Class = 1`)** e para a **Curva Precision-Recall (PR-AUC)**:
- **Recall:** Mede a capacidade de barrar fraudes reais (evitando prejuízo direto com estornos).
- **Precisão:** Mede quantos bloqueios eram realmente fraudes (evitando atrito ao bloquear clientes legítimos).
- **F1-Score:** Equilibra Precisão e Recall.

## 2. Preparação dos Dados
- **Carregamento direto:** Leitura via URL pública sem armazenar arquivos pesados (>100 MB) no repositório.
- **Limpeza:** Remoção de **1.081 linhas duplicadas** (incluindo 19 fraudes repetidas), restando **283.726 transações únicas** (473 fraudes).
- **Transformação Logarítmica:** Aplicação de `np.log1p()` na variável `Amount` para suavizar a forte assimetria dos valores financeiros (onde muitas fraudes ocorrem em valores baixos de teste).
- **Prevenção de Vazamento (*Data Leakage*):** Divisão estratificada (`stratify=y`, 80% treino / 20% teste) realizada antes de qualquer escalonamento. O `StandardScaler` foi ajustado exclusivamente nos dados de treino e encapsulado em um `Pipeline` na etapa de validação cruzada.

## 3. Comparação dos Modelos (Limiar Padrão 0,5)

| Modelo | Precisão (Fraude) | Recall (Fraude) | F1-Score (Fraude) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Regressão Logística (Baseline) | 84,85% | 58,95% | 0,6957 | 0,7010 |
| Random Forest (`class_weight='balanced'`) | 88,75% | 74,74% | 0,8114 | 0,7473 |
| **XGBoost (`scale_pos_weight`)** | **97,37%** | **77,89%** | **0,8655** | **0,8251** |

O **XGBoost** apresentou o melhor equilíbrio inicial, capturando quase 78% das fraudes com mais de 97% de precisão.

## 4. Otimização, Limiar de Decisão e Explicabilidade (SHAP)
- **Ajuste de Hiperparâmetros:** Uso de `Pipeline` + `GridSearchCV` com `scoring='recall'` no XGBoost (`learning_rate=0.05`, `max_depth=3`, `n_estimators=100`), elevando o Recall no teste para **84,21%** no limiar 0,5 (Precisão de 10,17%).
- **Redução do Limiar (0,5 → 0,3):** Ao reduzir o corte de probabilidade para **0,3**, o **Recall subiu para 87,37%** (capturando 83 das 95 fraudes do conjunto de teste), enquanto a Precisão caiu para **4,54%**, evidenciando o trade-off prático entre máxima proteção contra perdas e aumento de verificações preventivas.
- **Explicabilidade com SHAP:**
  - **Visão Global (*Beeswarm*):** As variáveis **`V14`**, **`V4`**, **`V10`** e **`V12`** foram as mais determinantes[cite: 1]. Valores baixos/negativos em `V14`, `V10` e `V12` e valores altos em `V4` empurram fortemente a previsão para fraude[cite: 1].
  - **Visão Individual (*Waterfall*):** Em uma fraude real analisada (probabilidade prevista de **99,44%**), a queda acentuada em `V14` (`-4.004`, impacto `+2.11`) e `V10` (`-3.295`, impacto `+1.04`) justificou o bloqueio[cite: 2].

## 5. Diferenciais Implementados
- Remoção explícita de registros duplicados antes do split para evitar superestimação das métricas.
- Encapsulamento do `StandardScaler` dentro de um `Pipeline` durante o `GridSearchCV` para eliminar qualquer vazamento de dados entre *folds*.
- Inclusão da métrica **PR-AUC** na tabela comparativa e análise crítica do custo operacional de otimizar exclusivamente para Recall.
