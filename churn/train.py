# %%
import pandas as pd
pd.options.display.max_rows = 999

df = pd.read_csv('../data/abt_churn.csv')
df.head()

# %%
# Separando a base out of time
oot = df[df['dtRef'] == df['dtRef'].max()].copy()

#%%
# Definir o dataframe de treino
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()

#%%

# Essas são as variaveis
features = df_train.columns[2:-1]

# Essa é a target
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                     random_state=42,
                                                                     test_size=0.2,
                                                                     stratify=y) 

# %%

print("Taxa de variável resposta Teste:",y.mean())
print("Taxa de variável resposta Teste:",y_train.mean())
print("Taxa de variável resposta Treino:",y_test.mean())

# %%

# EXPLORE MISSINGS

X_train.isna().sum().sort_values(ascending=False)

# %%
df_analise = X_train.copy()
df_analise[target] =  y_train
sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
display(sumario)

# %%

sumario['diff_abs'] = sumario[0] - sumario[1]         
sumario['diff_rel'] = sumario[0]/sumario[1] 
sumario.sort_values(by=['diff_rel'], ascending=False)        

# %%

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
arvore.fit(X_train, y_train)

# %%

# Definindo as principais features
feature_importance = (pd.Series(arvore.feature_importances_,
                                index=X_train.columns)
                                .sort_values(ascending=False)
                                .reset_index())
feature_importance['acum'] = feature_importance[0].cumsum()
best_features = (feature_importance[feature_importance['acum'] < 0.96]['index']
                 .tolist())
best_features

# %%

# MODIFY
from feature_engine import discretisation, encoding # type: ignore
from sklearn import pipeline, linear_model, metrics


## Discretizar
# Regression = False se as variáveis forem para modelar uma classificação
tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                             bin_output = 'bin_number',
                                                             regression=False,
                                                             cv=3
                                                             )

# OneHot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# Model
reg = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=10000)

# Model pipeline
model_pipeline = pipeline.Pipeline(
    steps = [
        ("Discretizar", tree_discretization),
        ("OneHot", onehot),
        ("Model", reg)
    ]
)

model_pipeline.fit(X_train, y_train)

#%%

y_train_predict = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

# Calcular as métricas na base de dados de treino
acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print(f"Acurácia Treino: {acc_train}")
print(f"AUC Treino: {auc_train}")

# %%
# Aplicar as métricas em cima da base de teste
y_test_predict = model_pipeline.predict(X_test)
y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

# Calcular as métricas na base de dados de test
acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print(f"Acurácia teste: {acc_test}")
print(f"AUC teste: {auc_test}")

# %%

# Fazer a mesma coisa na oot (Out of time)
oot_predict = model_pipeline.predict(oot[features])
oot_proba = model_pipeline.predict_proba(oot[features])[:,1]

# Calcular as métricas na base de dados de test
acc_oot = metrics.accuracy_score(oot[target], oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], oot_proba)
print(f"Acurácia oot: {acc_oot}")
print(f"AUC oot: {auc_oot}")
# %%
