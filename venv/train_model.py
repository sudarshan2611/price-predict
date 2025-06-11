# train_model.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Use a few features
X = df[['MedInc', 'AveRooms', 'AveOccup']]
y = df['MedHouseVal']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
