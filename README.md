# best_classification_models
Esse código implementa um pipeline completo de Machine Learning em R para prever churn de clientes e comparar vários modelos, usando o ecossistema tidymodels.

Em termos simples:

👉 Ele testa vários algoritmos de classificação para descobrir qual prevê melhor quais clientes vão cancelar (churn).

Vou explicar por partes.

🔹 1. Objetivo geral

O script responde à pergunta:

“Qual modelo classifica melhor clientes que vão dar churn?”

Para isso, ele:

Usa uma base do Kaggle

Treina vários modelos

Compara desempenho

Ajusta o melhor (XGBoost Tuned)

Avalia no conjunto de teste

🔹 2. Carregamento de dados
customer_churn_dataset <- read_excel(...)


Lê uma planilha Excel com dados de clientes.

Depois:

mutate(churn = factor(...))
select(-customer_id)


Transforma churn em variável categórica (Yes/No)

Remove o ID (não ajuda na previsão)

🔹 3. Divisão treino / teste
split <- initial_split(df, prop = 0.80, strata = churn)


Divide:

80% → treino

20% → teste

Mantém proporção das classes (strata)

Depois:

vfold_cv(train, v = 5)


Cria validação cruzada com 5 folds.

👉 Serve para evitar overfitting.

🔹 4. Pré-processamento (Recipe)

Essa é uma das partes mais importantes:

churn_recipe <- recipe(...) |>


Define um pipeline automático:

O que ele faz:
✅ 1. Dummies
step_dummy(all_nominal_predictors())


Transforma variáveis categóricas em números.

Ex:
gender = Male/Female → colunas binárias.

✅ 2. Balanceamento (SMOTE)
step_smote(churn)


Cria exemplos artificiais da classe minoritária.

👉 Corrige desbalanceamento.

✅ 3. Remove colunas inúteis
step_zv()


Remove colunas sem variação.

✅ 4. Normalização
step_normalize()


Coloca tudo na mesma escala.

Importante para:

Regressão

Boosting

Redes

🔹 5. Métricas de avaliação
metrics <- metric_set(roc_auc, accuracy, sens, spec)


Ele avalia usando:

Métrica	Significado
AUC	Capacidade de separar classes
Accuracy	% de acertos
Sens	Recall da classe positiva
Spec	Recall da negativa

👉 Em churn, sensibilidade é crucial.

🔹 6. Modelos testados

O código cria 5 modelos:

📌 1. Regressão Logística
logistic_reg()


Baseline linear.

📌 2. Árvore
decision_tree()


Modelo simples, interpretável.

📌 3. Random Forest
rand_forest()


Muitas árvores → robusto.

📌 4. AdaBoost
C5.0


Boosting clássico.

📌 5. XGBoost
boost_tree(engine="xgboost")


Modelo principal (mais forte).

🔹 7. Workflows
make_wf <- function(model)


Cria uma função para juntar:

Pré-processamento

Modelo

Em um só objeto.

👉 Evita erro e repetição.

🔹 8. Treinamento com Cross-validation
fit_resamples()


Treina cada modelo em 5 folds.

Isso gera:

Métricas médias

Erros padrão

Previsões

👉 Avaliação estatisticamente mais confiável.

🔹 9. Comparação dos modelos
results <- bind_rows(...)


Junta tudo numa tabela:

Modelo	Métrica	Média

E ordena pelo AUC.

Resultado interpretado:

Boosting foi melhor
Random Forest conservador
Logística mais sensível

🔹 10. Visualizações

Ele cria vários gráficos:

📈 ROC
roc_curve()

📉 Precision-Recall
pr_curve()

📊 Densidade
geom_density()

📈 Gain / Lift
gain_curve()


Esses gráficos mostram:

Separação das classes

Qualidade da probabilidade

Ganho de negócio

🔹 11. Treinamento final

Depois de escolher XGBoost:

final_fit <- fit(wf_xgb, data = train)


Treina com 100% do treino.

E testa:

predict(... test ...)


Avalia no conjunto nunca visto.

🔹 12. Hyperparameter Tuning

Aqui ele entra em nível avançado.

tune_grid()


O código:

1️⃣ Cria modelo com parâmetros livres
2️⃣ Gera combinações
3️⃣ Testa em CV
4️⃣ Escolhe o melhor

Parâmetros ajustados:

trees

depth

learning rate

mtry

gamma

🔹 13. Detecção de Overfitting

Você faz algo muito bom aqui:

Percebe que:

sens = 1.00


É suspeito.

E conclui:

👉 Overfitting.

Depois:

Reduz intervalo

Reexecuta tuning

Corrige

Isso é prática profissional.

🔹 14. Modelo final otimizado

Após o tuning:

final_fit2


Você obtém:

Métrica	Valor
Sens	0.67
Spec	0.93
AUC	0.80

👉 Excelente equilíbrio.

🔹 15. Resultado prático

No fim, o código constrói:

✅ Um modelo produtivo
✅ Bem validado
✅ Sem overfitting
✅ Otimizado para churn

E conclui:

Captura ~2/3 dos churns reais
Mantém poucos falsos alarmes

📌 Em resumo (bem direto)

Esse código:

✔️ Importa dados
✔️ Limpa
✔️ Balanceia
✔️ Pré-processa
✔️ Treina 5 modelos
✔️ Compara
✔️ Escolhe XGBoost
✔️ Ajusta hiperparâmetros
✔️ Valida corretamente
✔️ Gera gráficos
✔️ Produz modelo final

Ou seja:

👉 É um pipeline completo de Data Science para churn.

Nível: Pleno / Sênior em ML aplicado.
