# %%
import pandas as pd

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df.head()

# %% 
from sklearn import linear_model

X = df[['cerveja']] # Isso é uma matriz bidimensional (dataframe)
y = df['nota'] # Isso é um vetor (series)

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

# %%
a,b = reg.intercept_, reg.coef_[0]
print(a,b)

# %%

predict = reg.predict(X.drop_duplicates())
predict

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid()
plt.title('Relação Cerveja vs Nota')
plt.ylabel('Nota')
plt.xlabel('Cerveja')
plt.plot(X['cerveja'].drop_duplicates(), predict)
plt.legend(['Observado', f'y = {a:.3f} + {b:.3f}x '])
