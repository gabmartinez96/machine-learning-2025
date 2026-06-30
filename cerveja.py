# %%

import pandas as pd

# Importou os dados
df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

# %%

from sklearn import linear_model

# Separou as variáveis independentes e a resposta
X = df[['cerveja']] # O X vai ser uma matriz (Dataframe)
y = df['nota'] # A variavel resposta vai ser um vetor (Serie)

# Cria um modelo e fita aos dados
reg = linear_model.LinearRegression()
reg.fit(X, y)

#%%

# Visualizar os coeficientes da regressão
a, b = reg.intercept_, reg.coef_[0]
print(a,b)

#%%

# Testar o modelo para predizer dados
predict =  reg.predict(X.drop_duplicates())

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid()
plt.title("Relação Cerveja vs Nota")
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X['cerveja'].drop_duplicates(), predict)

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f}x'])
