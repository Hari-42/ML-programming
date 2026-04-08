import pandas as pd
from sklearn.model_selection import train_test_split





data = pd.read_csv("m245_lu02_a02_input.csv")
X = data.drop("buy", axis=1)
Y = data["buy"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

