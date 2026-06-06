# %%

import pandas as pd

df = pd.read_excel("../data/dados_frutas.xlsx")
df

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(
    random_state=42
)  # Define o algoritmo que será usado

# %%

y = df["Fruta"]

caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

# %%

# Ensina a máquina; a tradução de fit é ajuste; isso aqui é machine learning

arvore.fit(X, y)

# %%

# Faz uma predição

arvore.predict([[0, 0, 1, 0]])

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(
    arvore, feature_names=caracteristicas, class_names=arvore.classes_, filled=True
)

# %%

probabilidade = arvore.predict_proba(
    [[1, 1, 1, 1]]
)[0]  # Da a probabilidade de cada uma das classes fornecidas
pd.Series(probabilidade, index=arvore.classes_)
