import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Оцінка вартості житла", page_icon="🏠", layout="centered")

st.title("🏠 Прогнозування вартості будинку")
st.write("Введіть характеристики житла для розрахунку його оціночної вартості.")

@st.cache_resource
def load_assets():
    model = joblib.load('models/house_price_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    expected_features = list(scaler.feature_names_in_)
except Exception as e:
    st.error(f"Помилка завантаження моделі з папки models/: {e}")
    st.stop()

st.header("Параметри об'єкта:")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Загальна площа (кв. фути)", value=7420, step=100)
    bedrooms = st.slider("Кількість спалень", 1, 6, 4)
    bathrooms = st.slider("Кількість ванних кімнат", 1, 5, 2)
    stories = st.slider("Кількість поверхів", 1, 4, 3)
    parking = st.slider("Місць для паркування", 0, 4, 2)

with col2:
    mainroad = st.selectbox("Вихід на головну дорогу?", ["Так", "Ні"])
    guestroom = st.selectbox("Наявність гостьової кімнати?", ["Ні", "Так"])
    basement = st.selectbox("Наявність підвалу/цоколю?", ["Ні", "Так"])
    hotwaterheating = st.selectbox("Водяне опалення?", ["Ні", "Так"])
    airconditioning = st.selectbox("Наявність кондиціонера?", ["Так", "Ні"])
    prefarea = st.selectbox("Престижний район?", ["Так", "Ні"])
    furnishing = st.selectbox("Меблювання", ["furnished", "semi-furnished", "unfurnished"])

if st.button("🔮 Розрахувати ціну", type="primary"):
    # 1. Формуємо базові вхідні дані
    raw_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': 1 if mainroad == "Так" else 0,
        'guestroom': 1 if guestroom == "Так" else 0,
        'basement': 1 if basement == "Так" else 0,
        'hotwaterheating': 1 if hotwaterheating == "Так" else 0,
        'airconditioning': 1 if airconditioning == "Так" else 0,
        'parking': parking,
        'prefarea': 1 if prefarea == "Так" else 0,
        'furnishingstatus_semi-furnished': 1 if furnishing == "semi-furnished" else 0,
        'furnishingstatus_unfurnished': 1 if furnishing == "unfurnished" else 0,
    }

    # 2. Відтворюємо Feature Engineering з train.py
    raw_data['area_per_room'] = area / (bedrooms + 1e-5)
    raw_data['bath_per_bed'] = bathrooms / (bedrooms + 1e-5)

    df_input = pd.DataFrame([raw_data])

    # 3. Вирівнюємо колонки відповідно до того, як їх навчив scaler
    final_input = pd.DataFrame()
    for col in expected_features:
        final_input[col] = df_input[col] if col in df_input.columns else 0.0

    # 4. Прогнозування
    scaled_data = scaler.transform(final_input)
    prediction = model.predict(scaled_data)[0]

    st.markdown("---")
    st.success(f"### 💵 Оціночна вартість: ₹{prediction:,.0f} (приблизно ${prediction/83.5:,.0f} USD)")    
    col_min, col_max = st.columns(2)
    col_min.metric("Нижня межа (-5%)", f"{prediction * 0.95:,.0f}")
    col_max.metric("Верхня межа (+5%)", f"{prediction * 1.05:,.0f}")