import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. Завантаження даних
df = pd.read_csv('data/housing.csv')

# Препроцесинг категоріальних колонок (yes/no -> 1/0) 
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

# Кодування текстового статусу меблювання (furnished, semi-furnished, unfurnished)
if 'furnishingstatus' in df.columns:
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

# Feature Engineering (корисні характеристики під новий датасет)
if 'area' in df.columns and 'bedrooms' in df.columns:
    df['area_per_room'] = df['area'] / (df['bedrooms'] + 1e-5)

if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
    df['bath_per_bed'] = df['bathrooms'] / (df['bedrooms'] + 1e-5)

# Визначення цільової колонки (y) та ознак (X)
target_col = 'price' if 'price' in df.columns else df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Розділення даних на train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Масштабування ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Навчання XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# 5. Збереження моделей та скалера
os.makedirs('models', exist_ok=True)
joblib.dump(xgb_model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Успішно! Модель XGBoost та скалер збережено в папку models/")