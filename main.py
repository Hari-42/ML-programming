import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# Teil A – ML-Pipeline bauen
# ============================================================

# 1. CSV einlesen
data = pd.read_csv("shop_data.csv", sep=";")

# 2. Features (X) und Zielvariable (y) trennen
X = data.drop("buy", axis=1)
y = data["buy"]

# 3. Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Pipeline: StandardScaler + Logistische Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# 5. Modell trainieren
pipeline.fit(X_train, y_train)

# 6. Vorhersagen berechnen
y_pred = pipeline.predict(X_test)

# ============================================================
# Teil B – Bewertung
# ============================================================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))