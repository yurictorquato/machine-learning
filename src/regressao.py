# %%

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model

df = pd.read_excel(io="../data/dados_cerveja_nota.xlsx", engine="openpyxl")

df.head()

# %%

X = df[["cerveja"]]  # X sempre é uma matriz, por isso ele é maiúsculo
y = df["nota"]  # y representa um vetor

reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%

a, b = reg.intercept_, reg.coef_[0]
print(a, b)

# %%

predict = reg.predict(X.drop_duplicates(ignore_index=True))
predict

# %%

plt.plot(X["cerveja"], y, "o")
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates(ignore_index=True)["cerveja"], predict)

plt.legend(["Observado", f"y = {a:.2f} + {b:.2f}x"])
