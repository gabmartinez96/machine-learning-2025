# %%

import pandas as pd

df = pd.read_csv('data/dados_comunidade.csv')
df.head()

#%%
# Transformar os dados Sim em Não para numérico (1 e 0)
df = df.replace({'Sim':1, 'Não':0})
df.head()

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

# %%

# Construção dos modelos

from sklearn import tree

# Definindo as variaveis independentes e resposta
X = df_analise.drop(columns=['pessoa_feliz']).astype(int)
y = df_analise['pessoa_feliz'].astype(int)

# Arvore de decisão
arvore = tree.DecisionTreeClassifier(random_state=42,
                                      min_samples_leaf=5
                                      )
arvore.fit(X,y)
# %%
# Testando o predict do modelo
arvore_predict = arvore.predict(X)
df_predict = df_analise[['pessoa_feliz']]
df_predict['predict_arvore'] = arvore_predict
df_predict

#%%

## Acurácia

(df_predict['pessoa_feliz'] == df_predict['predict_arvore']).mean()

# %%

# Matriz de confusao

pd.crosstab(df_predict['pessoa_feliz'], df_predict['predict_arvore'])

# %%

(df_predict['pessoa_feliz'] == 1).sum()

# %