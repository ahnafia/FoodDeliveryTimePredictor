import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("Food_Delivery_Times.csv")
df = df.dropna()

X = df.drop(columns=["Order_ID", "Delivery_Time_min"])
y = df["Delivery_Time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_features = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_tensor = tf.convert_to_tensor(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

model = models.Sequential([
    layers.Input(shape=(X_train_tensor.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train_tensor, y_train_tensor,
    validation_data=(X_test_tensor, y_test_tensor),
    epochs=50,
    batch_size=8,
    verbose=1
)

loss, mae = model.evaluate(X_test_tensor, y_test_tensor)
print(f"Test Loss: {loss}, Test MAE: {mae}")

sample_input = pd.DataFrame({
    "Distance_km": [1.2],
    "Preparation_Time_min": [15],
    "Courier_Experience_yrs": [3],
    "Weather": ["Rainy"],
    "Traffic_Level": ["Low"],
    "Time_of_Day": ["Morning"],
    "Vehicle_Type": ["Scooter"]
})

sample_input_processed = preprocessor.transform(sample_input)
sample_input_tensor = tf.convert_to_tensor(sample_input_processed.toarray() if hasattr(sample_input_processed, "toarray") else sample_input_processed, dtype=tf.float32)

prediction = model.predict(sample_input_tensor)
print(prediction)
