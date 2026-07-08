#%%
import pandas as pd

df = pd.read_parquet('data/dados_clones.parquet')
df.head()

# %%

features = ['Estatura(cm)', 'Massa(em kilos)']
target = df.columns[-1]

X = df[features]
y = df[[target]]


# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y = y)
# %%
tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3)