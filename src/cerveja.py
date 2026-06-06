# %%

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

df = pd.read_excel(io="../data/dados_cerveja.xlsx")

df

# %%

features = ["temperatura", "copo", "espuma", "cor"]  # Características
target = "classe"  # Variável resposta

X = df[features]
y = df[target]

X = X.replace({"mud": 1, "pint": 2, "não": 0, "sim": 1, "clara": 0, "escura": 1})

# %%

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

model.classes_

# %%

plt.figure(dpi=400)

tree.plot_tree(
    decision_tree=model, feature_names=features, class_names=model.classes_, filled=True
)
