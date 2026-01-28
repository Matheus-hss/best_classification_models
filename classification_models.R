###################### QUAL MODELO CLASSIFICA MELHOR ?🤖💻 ###################
#Utilizando uma base de Churn de clientes do Kaggle vamos utilizar 5 modelos
#para avaliar qual prevê melhor a classe correta, os modelos são:
#Regressão logística
#Árvore de decisão
#Random Forest
#AdaBoost
#XGBoost
#
#---------------------------------------------------------------------------#

#bibliotecas
library(readxl)
library(tidyverse)
library(tidymodels)
library(themis)
library(ranger)
library(xgboost)
library(gbm)
library(adabag)
library(vip)
library(C50) #pacote para modelo adaboost
library(lightgbm) #modelo lightgbm sera construido fora do tidymodels


customer_churn_dataset <- read_excel("C:/Users/m-hen/Downloads/customer_churn_dataset.xlsx", 
                                     col_types = c("numeric", "numeric", "numeric", 
                                                   "numeric", "text", "text", "text", 
                                                   "text", "text", "numeric", "text"))
View(customer_churn_dataset)
str(customer_churn_dataset)

#Ajuste da base de dados, transformando a coluna Churn em fator e retirando ID
set.seed(123)
df <- customer_churn_dataset |> 
  mutate(
    churn = factor(churn, levels = c("Yes", "No"))
  ) |> 
  select(-customer_id)

#Fazendo a separação entre treino e teste, 80% treino e 20% teste
split <- initial_split(df, prop = 0.80, strata = churn)
train <- training(split)
test <- testing(split)

#Criando fold para validação cruzada
folds <- vfold_cv(train, v = 5, strata = churn)

#Criação do recipe (pipeline de pré-processamento) do tidymodels
churn_recipe <- recipe(churn ~ ., data = train) |>  #Define alvo e preditores
  step_dummy(all_nominal_predictors()) |>  #Converte categóricas em dummies (one-hot encoding)
    step_smote(churn) |>  #Balancea as categorias "No" e "Yes" em Churn com a técnica SMOTE
      step_zv(all_predictors()) |>  #Remove colunas sem variância
        step_normalize(all_numeric_predictors()) #Padroniza/Normaliza variáveis numéricas
#Isso coloca todas as variáveis na mesma escala, o que é essencial para modelos sensíveis a magnitude

#Verificando o balanceamento da base treino
train |> 
  count(churn) |> 
    mutate(
      prop = n/sum(n)
    )
#Desbalanceamento moderado: 65,8% clientes que não deram churn / 34,2% clientes que deram churn
#Apliquei a técnica SMOTE (Synthetic Minority Oversampling Technique)
#Ela cria novas observações sintéticas da classe minoritária usando interpolação entre vizinhos próximos

#Métricas que serão utilizadas
metrics <- metric_set(roc_auc, accuracy, sens, spec)
#roc_auc → mede a capacidade do modelo de separar as classes
#accuracy → porcentagem de previsões corretas
#sens (sensibilidade / recall) → capacidade de detectar a classe positiva (“Yes”)
#spec (especificidade) → capacidade de detectar a classe negativa (“No”)

#Especificando os modelos dentro do tidymodels

#Regressão Logistica
log_model <- logistic_reg() |> 
  set_engine("glm") |> 
    set_mode("classification")

#Árvore de decissão
tree_model <- decision_tree() |> 
  set_engine("rpart") |> 
    set_mode("classification")

#Random Forest
rf_model <- rand_forest(trees = 500) |> 
  set_engine("ranger", importance = "impurity") |> 
    set_mode("classification")

#AdaBoost
ada_c50_model <- decision_tree() |> 
  set_engine("C5.0", trials = 100) |>    # trials = boosting rounds
  set_mode("classification")

#XGBoost
xgb_model <- boost_tree(
  trees = 800,
  tree_depth = 6,
  learn_rate = 0.05,
  mtry = 5,
  loss_reduction = 0,
  sample_size = 0.8
) |> 
  set_engine("xgboost") |> 
    set_mode("classification")

#Workflows
make_wf <- function(model) {
  workflow() |> 
    add_model(model) |> 
      add_recipe(churn_recipe)
} #Função para rodar os modelos de maneira mais rapida

wf_log <- make_wf(log_model)
wf_tree <- make_wf(tree_model)
wf_rf <- make_wf(rf_model)
wf_ada <- make_wf(ada_c50_model)
wf_xgb <- make_wf(xgb_model)

#Treinamento
res_log <- fit_resamples(wf_log, folds, metrics = metrics, control = control_resamples(save_pred = TRUE))
res_tree <- fit_resamples(wf_tree, folds, metrics = metrics, control = control_resamples(save_pred = TRUE))
res_rf <- fit_resamples(wf_rf, folds, metrics = metrics, control = control_resamples(save_pred = TRUE))
res_ada <- fit_resamples(wf_ada, folds, metrics = metrics, control = control_resamples(save_pred = TRUE))
res_xgb <- fit_resamples(wf_xgb, folds, metrics = metrics, control = control_resamples(save_pred = TRUE))

#Previsões

pred_log <- collect_predictions(res_log)
pred_tree <- collect_predictions(res_tree)
pred_rf   <- collect_predictions(res_rf)
pred_ada  <- collect_predictions(res_ada)
pred_xgb  <- collect_predictions(res_xgb)

#Matriz de confusão
conf_mat(pred_rf, truth = churn, estimate = .pred_class)
conf_mat(pred_rf, truth = churn, estimate = .pred_class) |> 
  autoplot(type = "heatmap")


#Para ver as outras matrizes basta alterar "pred_log"

#Métricas
results <- bind_rows(
  collect_metrics(res_log) |> mutate(model = "Regressão Logistica"),
  collect_metrics(res_tree) |> mutate(model = "Arvore de Decisão"),
  collect_metrics(res_rf) |> mutate(model = "Random Forest"),
  collect_metrics(res_ada) |> mutate(model = "AdaBoost"),
  collect_metrics(res_xgb) |> mutate(model = "XGBoost")
) |> 
  select(model, .metric, mean, std_err) |> 
  arrange(desc(.metric == "roc_auc"), desc(mean))

results
#“Os modelos baseados em boosting (XGBoost e AdaBoost) apresentaram o melhor desempenho discriminatório, com ROC AUC de aproximadamente 0,81. 
#O Random Forest apresentou maior acurácia e especificidade, sendo mais conservador na classificação. 
#A Regressão Logística apresentou maior sensibilidade, sendo mais eficiente na identificação de clientes propensos ao churn, porém com maior taxa de falsos positivos. 
#Assim, a escolha do modelo dependeria do trade-off entre recall e precisão desejado pelo negócio.”

#Visualizações
#1 - Curva ROC (como estamos trabalhando com dados desbalanceados a curva roc nesse caso não interessa muito)
roc_curve(pred_log, truth = churn, .pred_Yes) |> autoplot()


#2 - Curva de Precision - Recall
# -> Foca em precision e recall
# -> Mostra como o modelo se comporta nos casos positivos
pr_curve(pred_xgb, truth = churn, .pred_Yes) |> autoplot()

#3 - Distribuição das probabilidades previstas
# -> Mostra se o modelo separa bem as classes
pred_xgb |> ggplot(aes(.pred_Yes, fill = churn))+
  geom_density(alpha = 0.4)

#4 - Tabela de threshold (cutoff)
#Mostra como métricas mudam conforme o threshold varia. Útil para escolher o melhor ponto de corte.
threshold_perf(pred_log, truth = churn, .pred_Yes)

#5 - Ganho acumulado / Lift Chart
# -> Mostra quanto o modelo melhora a seleção de positivos
gain_curve(pred_ada, truth = churn, .pred_Yes) |>  autoplot()

# Modelo final -> XGBoost
# Treino do Modelo no conjunto de treino, fora da validação cruzada
final_fit <- fit(wf_xgb, data = train)

#Previsões no conjunto de teste
pred_test <- predict(final_fit, new_data = test, type = "prob") |> 
  bind_cols(predict(final_fit, new_data = test, type = "class")) |> 
  bind_cols(test |> select(churn))

#Métricas de classificação somente para o teste
test_metrics <- metric_set(accuracy, sens, spec) #conjunto de métricas só para o teste
test_metrics(pred_test, truth = churn, estimate = .pred_class)

#Área sob a curva ROC
pred_test |>  roc_auc(truth = churn, .pred_Yes, event_level = "first")

#Matriz de confusão
conf_mat(pred_test, truth = churn, estimate = .pred_class)

#📊 Leitura das métricas
#✅ 1. Accuracy = 0.843
#O modelo acerta 84.3% das previsões no conjunto de teste.
#Parece bom, mas accuracy engana quando há desbalanceamento (como churn).
#Por isso, as métricas mais importantes são sensibilidade e especificidade.

#🎯 2. Sensibilidade (sens) = 0.665
#Sensibilidade mede:
  
# valor: 66.5%
#Interpretação:
#O modelo identifica 2 de cada 3 churns reais.
#Isso é razoável, especialmente se a classe positiva for pequena.
#Em churn, sensibilidade costuma ser mais importante que accuracy.

# 3. Especificidade (spec) = 0.935
#Especificidade mede:
  
# valor: 93.5%
#Interpretação:
#O modelo quase não gera falsos positivos.
#Ele é muito bom em reconhecer clientes que não vão churnar.

#🧠 O que isso significa no contexto de churn
#Seu modelo está:
#• 	Muito bom em identificar quem NÃO vai churnar (spec alta)
#• 	Razoável em identificar quem VAI churnar (sens moderada)
#• 	Globalmente bom (accuracy alta)
#Isso é típico de modelos treinados em bases desbalanceadas, onde a classe “No” é muito mais frequente.
#

#Usando Tune_Grid() para melhorar o modelo
#Abaixo fazemos os ajustes dos hiperparâmetros a serem tunados
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  loss_reduction = tune()
) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

#Workflow
wf_xgb_tune <- workflow() |> 
  add_model(xgb_model) |> 
  add_recipe(churn_recipe)

#Grade de Hiperparâmetros
grid <- grid_space_filling(
  finalize(mtry(), train),
  trees(),                         
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction(),
  size = 20)

#Execução🔁
tuned_xgb <- tune_grid(
  wf_xgb_tune,
  resamples = folds,     # validação cruzada
  grid = grid,
  metrics = metric_set(roc_auc, accuracy, sens, spec),
  control = control_grid(save_pred = TRUE)
)

#🏆 Escolha dos melhores hiperparâmetros
best_params <- select_best(tuned_xgb, metric = "sens")
best_params

#🔍 Interpretação dos melhores hiperparâmetros

#🌳 1. 
#O modelo está usando 5 variáveis por split.
# • 	Isso reduz correlação entre árvores
# • 	Ajuda a evitar overfitting
# • 	É um valor comum quando se busca sensibilidade (recall)
# 
# 🌲 2. 
# Um número alto de árvores.
# Isso faz sentido porque:
# • 	estou usando um learning rate extremamente baixo
# • 	então o modelo precisa de muitas árvores para aprender
# • 	isso tende a melhorar recall, porque o modelo vai “lapidando” lentamente os padrões da classe minoritária
# 
# 🧠 3. 
# Árvores profundas.
# • 	Árvores profundas capturam interações complexas
# • 	Isso ajuda a identificar padrões raros (como churn)
# • 	Mas aumenta risco de overfitting — que é compensado pelo learning rate minúsculo
# 
# 🐢 4.  learn_rate -> (1e-10)
# Um learning rate tão baixo significa:
# • 	cada árvore contribui quase nada
# • 	o modelo precisa de muitas árvores
# • 	o aprendizado é extremamente lento
# • 	isso pode melhorar recall, mas também pode indicar que o espaço de busca encontrou um “canto” estranho
# Esse valor é suspeito — não errado, mas incomum.
# Pode indicar:
# • 	a grade de hiperparâmetros está muito ampla
# • 	o modelo está tentando compensar overfitting
# • 	a métrica sensibilidade está favorecendo combinações extremas
# 
# 🔧 5. loss_reduction -> 0.000113
# Esse é o gamma do XGBoost.
# • 	Valores baixos permitem splits mais agressivos
# • 	Isso aumenta sensibilidade
# • 	Ajuda a capturar padrões da classe minoritária
# 
# 🎯 Resumo da interpretação
# O modelo está:
# • 	usando muitas árvores
# • 	com aprendizado extremamente lento
# • 	árvores profundas
# • 	splits agressivos
# • 	e poucas variáveis por split
# Esse conjunto tende a:
# • 	aumentar sensibilidade (meu objetivo)
# • 	aumentar recall da classe positiva
# • 	mas pode reduzir precisão
# • 	e pode aumentar tempo de treino

# Possiveis soluções
# 1) Ver outras combinações de modelos
show_best(tuned_xgb, metric = "sens", n = 10)
# 🎯 1) O que o  está dizendo
# A melhor combinação encontrada tem:
# • 	sensibilidade = 1.00 (perfeita)
# • 	erro padrão = 0
# • 	hiperparâmetros extremamente extremos (learning rate absurdamente baixo)
# Isso é um sinal claro de:
# 👉 Overfitting dentro da validação cruzada
# O modelo encontrou uma combinação que memoriza padrões da classe positiva nos folds, mas isso não generaliza.
# Porém no conjunto de teste, a sensibilidade não chega nem perto de 1.00.
# 
# 🔍 2) Por que isso acontece?
# Os hiperparâmetros das melhores linhas:
# Linha 1 (sens = 1.00)
# Linha 2 (sens = 0.973)
# Linha 3 (sens = 0.732)
# Esses padrões mostram:
# ✔️ O modelo está explorando regiões extremas do espaço de hiperparâmetros
# • 	learning rate muito baixo
# • 	árvores muito profundas
# • 	ou até uma única árvore (linha 3)
# Isso é típico quando:
# •	a grade é muito ampla
# • 	a métrica favorece recall a qualquer custo
# • 	a classe positiva é pequena
# • 	o modelo tenta “memorizar” os churns nos folds
# 
# ⚠️ 3) O maior alerta: sens = 1.00 com std_err = 0
# Isso significa:
# • 	em todos os 5 folds, a sensibilidade foi 1.00
# • 	isso é extremamente improvável em churn real
# • 	indica que o modelo está decorando os padrões da classe positiva nos folds
# Esse tipo de solução que não generaliza.
# 
# 🧠 4) Outras linhas
# Sensibilidade cai rapidamente:
# • 	0.973
# • 	0.732
# • 	0.693
# • 	0.692
# • 	0.690
# • 	0.675
# • 	0.674
# • 	0.673
# Isso mostra que:
# a maior parte das combinações está em torno de 0.67–0.73
# esses valores são muito mais realistas
# a solução com sens = 1.00 é um outlier causado por hiperparâmetros extremos
#Vou rodar novamente o modelo agora com um intervalo de learn-rate e tree_depth
#learn_rate(range = c(-5, -1))  # 1e-5 a 1e-1
#tree_depth(range = c(2, 10))

#Reexecução do processo de tunagem do modelo agora com intervalos🔁
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  loss_reduction = tune()
) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

#Workflow
wf_xgb_tune2 <- workflow() |> 
  add_model(xgb_model) |> 
  add_recipe(churn_recipe)

#Nova Grade de Hiperparâmetros
grid2 <- grid_space_filling(
  finalize(mtry(), train),
  trees(),                         
  tree_depth(range = c(2, 10)),
  learn_rate(range = c(-5, -1)),
  loss_reduction(),
  size = 20)

#ReExecução
tuned_xgb2 <- tune_grid(
  wf_xgb_tune2,
  resamples = folds,     # validação cruzada
  grid = grid2,
  metrics = metric_set(roc_auc, accuracy, sens, spec),
  control = control_grid(save_pred = TRUE)
)

#Vendo combinações dos 10 melhores modelos depois da reexecução
show_best(tuned_xgb2, metric = "sens", n = 10)

# 📊 Interpretação dos 10 melhores modelos
# A sensibilidade dos 10 melhores está entre 0.670 e 0.676.
# Isso é excelente: significa que o modelo atingiu um patamar estável.
# Vamos olhar os padrões.
# 
# 🌳 1. mtry variando de 1 a 10
# Isso mostra que:
# • 	o modelo não depende fortemente de um número específico de variáveis por split
# • 	várias combinações funcionam bem
# • 	isso é típico quando as variáveis têm relevância distribuída
# 
# 🌲 2. trees variando de 211 a 2000
# Isso indica:
# • 	modelos com muitas árvores continuam sendo competitivos
# • 	mas modelos menores (ex.: 316, 421, 527) também funcionam bem
# • 	o learning rate controla o ritmo de aprendizado
# 
# 🧠 3. tree_depth entre 3 e 10
# Isso é ótimo:
# • 	profundidades moderadas → menos overfitting
# • 	profundidades altas (8–10) aparecem entre os melhores
# • 	profundidades baixas (3–4) também aparecem
# Ou seja: o modelo está flexível, mas não exagerado.
# 
# 🐢 4. learn_rate agora está dentro do intervalo realista
# Valores como:
# • 	0.000113
# • 	0.00001
# • 	0.00546
# • 	0.000785
# • 	0.0144
# • 	0.1
# Isso é perfeito:
#   o modelo está explorando desde learning rates lentos até rápidos, sem cair em extremos absurdos.
# 
# 🔧 5. loss_reduction variando muito
# Isso é esperado:
# • 	valores pequenos → splits mais agressivos
# • 	valores grandes → splits mais conservadores
# • 	ambos aparecem entre os melhores
# Isso mostra que o modelo está encontrando boas soluções em diferentes regimes de complexidade.
# 

#🏆Selecionando o melhor modelo e rodando o treino inteiro nele
# 1. 	tune_grid() → gera várias combinações
# 2. 	select_best() → retorna um tibble com os hiperparâmetros
# 3. 	finalize_workflow() → coloca esses hiperparâmetros dentro do workflow
# 4. 	fit() → treina o modelo final

best_model <- select_best(tuned_xgb2, metric = "sens")
best_model

final_wf2 <- finalize_workflow(wf_xgb_tune2, best_model)
final_fit2 <- fit(final_wf2, data = train)

pred_test_final <- predict(final_fit2, new_data = test, type = "prob") |> 
  bind_cols(predict(final_fit2, new_data = test, type = "class")) |> 
  bind_cols(test |> select(churn))

test_metrics(pred_test_final, truth = churn, estimate = .pred_class)
# Abaixo a analise das métricas:
# 🎯 1. Sensibilidade = 0.671
# Esse é o ponto central, já que você otimizou o modelo para sensibilidade.
# • 	O modelo está capturando 2 de cada 3 clientes que realmente churnam.
# • 	Isso é muito bom para um problema de churn, especialmente se a classe positiva for pequena.
# • 	E, o mais importante:esse valor é consistente com o que o tuning encontrou (0.67–0.68).
# Ou seja:
#   👉 o modelo generalizou bem
#   👉 não houve overfitting
#   👉 o tuning funcionou
# 
# 🛡️ 2. Especificidade = 0.935
# Isso significa:
# • 	O modelo quase não gera falsos positivos.
# • 	Ele identifica corretamente 93,5% dos clientes que não churnam.
# Esse equilíbrio é ótimo: aumentou recall sem destruir a capacidade de prever “não churn”.
# 
# 🧠 3. Accuracy = 0.845
# Esse valor é praticamente igual ao do modelo anterior (0.843), mas agora:
# • 	com tuning mais estável
# • 	sem hiperparâmetros extremos
# • 	com sensibilidade melhor calibrada
# ganhou qualidade, não apenas números.

#Área sob a curva ROC
pred_test_final |>  roc_auc(truth = churn, .pred_Yes, event_level = "first")

#Matriz de confusão
conf_mat(pred_test_final, truth = churn, estimate = .pred_class)

#Curvas de densidade de probabilidades
pred_test_final |> ggplot(aes(.pred_Yes, fill = churn))+
geom_density(alpha = 0.4)

#Curva de lift/Gain
gain_curve(pred_test_final, truth = churn, .pred_Yes) |>  autoplot()

#📈 AUC = 0.806
# Um AUC de 0.806 significa que:
# • 	o modelo tem boa capacidade discriminativa
# • 	separa bem clientes que churnam dos que não churnam
# • 	está acima do patamar típico de modelos baseline (0.60–0.70)
# Em churn, AUC acima de 0.80 já é considerado muito bom.
# Isso confirma que:
# • 	o tuning funcionou
# • 	o modelo generaliza bem
# • 	não houve overfitting
# 🎯 1. Verdadeiros Positivos (TP) = 919
# Clientes que churnaram e o modelo acertou.
# Isso corresponde à sensibilidade de 0.671, exatamente o que você já viu.
# 
# 🛑 2. Falsos Negativos (FN) = 450
# Clientes que churnaram, mas o modelo previu “No”.
# Esse é o grupo que você tenta reduzir quando otimiza sensibilidade.
# O tuning ajudou a diminuir esse número sem sacrificar muito a precisão.
# 
# 🟩 3. Verdadeiros Negativos (TN) = 2460
# Clientes que não churnaram e o modelo acertou.
# Isso corresponde à especificidade de 0.935 — excelente.
# 
# 🟨 4. Falsos Positivos (FP) = 172
# Clientes que não churnaram, mas o modelo previu “Yes”.

# Esse número é baixo, o que é ótimo para evitar ações desnecessárias (ex.: oferecer desconto para quem não ia sair).
