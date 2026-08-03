# %%
import pandas as pd

model_df = pd.read_pickle('model.pkl')
model_df

# %%

# Receber novos dados
df = pd.read_csv('../data/abr_churn.csv')
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
# %%
predicao = model.predict_proba(amostra[features])[:,1]

# %%
