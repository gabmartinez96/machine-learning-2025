# %%
import pandas as pd

df = pd.read_excel('data/dados_frutas.xlsx')
df.head()

#%%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

#%%

# Definir as features / targets
features = ['Arredondada', 'Suculenta', 'Vermelha','Doce']
target = 'Fruta'
x = df[features]
y = df[target]

# Passar para o modelo
arvore.fit(x, y)

#%%
arvore.predict([[0,0,0,0]])

#%%

import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(arvore,
                feature_names=features,
                  class_names=arvore.classes_,
                    filled=True)



# %%
proba = arvore.predict_proba([[1,1,0,0]])[0]
pd.Series(proba, index=arvore.classes_)