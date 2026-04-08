import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


# 1) Load the dataset from CSV using ';' as separator.
data = pd.read_csv("m245_lu02_a02_input.csv", sep=";")

# 2) Select feature columns (X) and target column (y).
X = data[["age", "past_purchases", "minutes_on_page"]]
y = data["buy"]

# 3) Split into training and test sets (80/20).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Create and train the classifier on training data.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5) Make predictions on the test data.
y_pred = model.predict(X_test)

# 6) Evaluate and print required metrics.
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
