# %%
import pandas as pd
import mlflow


# Import do modelo via mlflow
mlflow.set_tracking_uri('http://localhost:5000')
models =mlflow.search_registered_models(filter_string="name = 'model_churn' ")
latest_version = max([i.version for i in models[0].latest_versions])
model = mlflow.sklearn.load_model(f"models:/model_churn/{latest_version}")
features = model.feature_names_in_


# # Importar o modelo
# model_df = pd.read_pickle('model.pkl')
# model = model_df['model']
# features = model_df['features']

# %%

# Importar novos dados
df = pd.read_csv('../data/abt_churn.csv')
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
# %%

# Predição em cima de novos dados
predicao = model.predict_proba(amostra[features])[:,1]
amostra['predicao'] = predicao
amostra



# %%