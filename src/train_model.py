import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv("data/housing.csv")

X = df[["sqft", "bedrooms"]]
y = df["luxury"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

with open("results/metrics.txt", "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}")

print("Training complete.")

