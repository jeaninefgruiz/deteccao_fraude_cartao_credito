# Detecção de Fraude em Cartão de Crédito

Projeto de machine learning para identificar transações fraudulentas em uma base pública muito desbalanceada. A avaliação usa métricas da classe de fraude e PR-AUC, e as decisões do modelo são analisadas com SHAP.

O código está no notebook `deteccao_fraude_cartao_credito.ipynb`, que abre no Colab pelo botão no início do arquivo.

## Problema

A base tem 284.807 transações, das quais 284.315 (99,83%) são legítimas e 492 (0,17%) são fraudes. Nesse cenário a acurácia engana: um modelo que classifica tudo como legítimo acerta 99,83% e não detecta nenhuma fraude.

A avaliação foca na classe de fraude (`Class = 1`) e na curva Precision-Recall (PR-AUC):

- Recall: proporção das fraudes reais que o modelo detecta (evita prejuízo direto com estornos).
- Precisão: proporção dos bloqueios que eram de fato fraude (evita bloquear clientes legítimos).
- F1-score: média harmônica entre precisão e recall.

## Preparação dos dados

- A base é lida de uma URL pública, sem armazenar o arquivo (mais de 100 MB) no repositório.
- Foram removidas 1.081 linhas duplicadas (19 eram fraudes), restando 283.726 transações únicas, com 473 fraudes.
- A variável `Amount` recebeu transformação `np.log1p` para reduzir a assimetria dos valores.
- A divisão em 80% treino e 20% teste é estratificada e feita antes de qualquer escalonamento. O `StandardScaler` é ajustado só no treino e, na validação cruzada, fica dentro de um `Pipeline`, o que evita vazamento de dados entre treino e validação.

## Comparação dos modelos (limiar 0,5)

| Modelo | Precisão (fraude) | Recall (fraude) | F1-score (fraude) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Regressão Logística (baseline) | 84,85% | 58,95% | 0,6957 | 0,7010 |
| Random Forest (`class_weight='balanced'`) | 88,75% | 74,74% | 0,8114 | 0,7473 |
| XGBoost (`scale_pos_weight`) | 97,37% | 77,89% | 0,8655 | 0,8251 |

O XGBoost teve o melhor resultado, com cerca de 78% das fraudes detectadas e precisão acima de 97%. O Random Forest foi treinado em configuração reduzida (30 árvores, profundidade máxima 8, 30% das amostras por árvore) para rodar mais rápido no Colab, então a comparação não tem o mesmo esforço de ajuste entre os modelos.

## Ajuste, limiar de decisão e explicabilidade

- Ajuste de hiperparâmetros: `GridSearchCV` com `Pipeline`, otimizando recall no XGBoost (melhor combinação: `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`). No teste, com limiar 0,5, o recall foi de 84,21% e a precisão caiu para 10,17%.
- Limiar: reduzir de 0,5 para 0,3 elevou o recall para 87,37% (83 das 95 fraudes do teste) e baixou a precisão para 4,54%. A escolha depende do custo de uma fraude não detectada em relação ao custo de verificar um alerta falso.
- SHAP, visão global (beeswarm): `V14`, `V4`, `V10` e `V12` foram as variáveis mais influentes. Valores baixos em `V14`, `V10` e `V12` e valores altos em `V4` aumentam a probabilidade de fraude.
- SHAP, visão individual (waterfall): em uma fraude real do conjunto de teste, com probabilidade prevista de 99,44%, os valores de `V14` (-4,004; contribuição de +2,11) e `V10` (-3,295; contribuição de +1,04) contribuíram para a previsão.

## Limitações

- As variáveis `V1` a `V28` são componentes anonimizados. O SHAP indica quais componentes pesam na decisão, não o motivo de negócio.
- O SHAP foi calculado sobre as 1.000 primeiras linhas do conjunto de teste, uma amostra com poucas fraudes (o teste inteiro tem 95).
- Otimizar por recall reduz muito a precisão. Otimizar por PR-AUC ou F1 seria uma alternativa a testar.
- Os resultados vêm de uma única divisão treino/teste (`random_state=42`).

## Tecnologias

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib e seaborn.
