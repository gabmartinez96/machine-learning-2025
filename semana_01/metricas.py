# %%

import pandas as pd

df = pd.read_csv('data/dados_comunidade.csv')
df.head()

#%%
# Transformar os dados Sim em Não para numérico (1 e 0)
df = df.replace({'Sim':1, 'Não':0})
df.head()

#%%
df.columns.tolist()
# %%

# Criar dummies para as variaveis categoricas com mais de uma categoria:
dummy_vars = [
 'Como conheceu o Téo Me Why?',
 'Quantos cursos acompanhou do Téo Me Why?',
 'Estado que mora atualmente',
 'Área de Formação',
 'Tempo que atua na área de dados',
 'Posição da cadeira (senioridade)']

# Separar as variaveis numericas
num_vars = [
 'Curte games?',
 'Curte futebol?',
 'Curte livros?',
 'Curte jogos de tabuleiro?',
 'Curte jogos de fórmula 1?',
 'Curte jogos de MMA?',
 'Idade'
]

# Colocar em um novo dataframe
df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa_feliz'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise.head()
#%%
df_analise.columns.tolist()
# %%

# Construção dos modelos

from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

# Definindo as variaveis independentes e resposta
X = df_analise.drop(columns=['pessoa_feliz']).astype(int)
y = df_analise['pessoa_feliz'].astype(int)

# Arvore de decisão
arvore = tree.DecisionTreeClassifier(random_state=42,
                                      min_samples_leaf=5
                                      )
arvore.fit(X,y)

# Naive Bayes
naive = naive_bayes.GaussianNB()
naive.fit(X,y)

# Regressão Logistica
reg = linear_model.LogisticRegression()
reg.fit(X, y)

# %%
# Testando o predict do modelo Arvore
arvore_predict = arvore.predict(X)
df_predict = df_analise[['pessoa_feliz']].astype(int)
df_predict['predict_arvore'] = arvore_predict
df_predict['proba_arvore'] = arvore.predict_proba(X)[:,1]

# Predict do modelo naive
df_predict['predict_naive'] = naive.predict(X)
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]

# Predict do modelo de regressão logistica
df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]


# %% 
df_predict.dtypes

# %%
from sklearn import metrics

# Metricas Arvores
acc_arvore = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'])
roc_arvore = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_arvore'])
auc_arvore = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_arvore'])

# Metrica Naive
acc_naive = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_naive'])
precisao_naive = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_naive'])
recall_naive = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_naive'])
roc_naive = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_naive'])
auc_naive = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_naive'])

# Metricas Regressao
acc_reg = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_reg'])
precisao_reg = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_reg'])
recall_reg = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_reg'])
roc_reg = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_reg'])
auc_reg = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_reg'])
auc_reg, auc_naive, auc_arvore

# %%

import matplotlib.pyplot as plt

plt.plot(roc_arvore[0], roc_arvore[1], 'o-')
plt.plot(roc_naive[0], roc_naive[1], 'o-')
plt.plot(roc_reg[0], roc_reg[1], 'o-')

plt.grid()
plt.xlabel('1 - Especificidade')
plt.ylabel('Recall')
plt.title('ROC Curve')
plt.legend([f'Arvore {auc_arvore:.2f}', f'Naive {auc_naive:.2f}', f'Regressao {auc_reg:.2f} '])

# %%

pd.Series({"model":reg, "features": X.columns}).to_pickle("model_feliz.pkl")
# %%

# %%
