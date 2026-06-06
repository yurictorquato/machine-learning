# %%

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree

df = pd.read_parquet(path="../data/dados_clones.parquet")

df.head()

# %%


def get_number_metrics(feature: list[str]) -> dict[str, int]:
    """Retorna um dicionário com valores únicos mapeados para índices."""

    feature_dict = {}
    for i, metric in enumerate(feature):
        feature_dict[metric] = i

    return feature_dict


# %%


def transform_feature_metrics(fields: pd.Series) -> list[str]:
    """Transforma múltiplas colunas em listas de valores únicos."""

    result = []
    for field in fields:
        result.append(df[field].sort_values().unique().tolist())

    return result


# %%

df = df.drop(columns=["p2o_master_id", "General Jedi encarregado"])

# Pegar colunas do tipo object, exceto a coluna que representa o target
type_df_objects = [column for column in df.columns[:-1] if df[column].dtype == "object"]

# Criar dicionários de mapeamento
metrics_dict = {}
for column in type_df_objects:
    unique_values = df[column].sort_values().unique().tolist()
    metrics_dict[column] = {value: i for i, value in enumerate(unique_values)}

# Aplicar transformação em todas as colunas object
for column in type_df_objects:
    df[column] = df[column].map(metrics_dict[column])

# %%

df_columns = df.columns.tolist()  # Características
features = df_columns[:-1]
target = df_columns[-1:]  # Variável resposta

X = df[features]
y = df[target]

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# %%

# Plota o gráfico
plt.figure(dpi=400)

tree.plot_tree(
    decision_tree=model,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    max_depth=3,
)
