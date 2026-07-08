# %%
import pandas as pd

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

X_train, X_test, y_train, y_teste = model_selection.train_test_split(X, y,
                                                                     random_state=42,
                                                                     test_size=0.2,
                                                                     stratify=y) 

# %%

print("Taxa de variável resposta Teste:",y_train.mean())
print("Taxa de variável resposta Treino:",y_teste.mean())

# %%

from sklearn import tree
from sklearn import linear_model
from sklearn import naive_bayes

arvore = tree.DecisionTreeClassifier()
nb = naive_bayes.GaussianNB()
reg = linear_model.LogisticRegression()