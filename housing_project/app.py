from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Завантаження моделей
price_model = joblib.load('models/house_price_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict/price', methods=['POST'])
def predict_house_price():
    try:
        data = request.json
        
        # Перетворення в DataFrame
        df = pd.DataFrame([data])
        
        # Feature engineering (згідно з описом у проєкті)
        if 'total_sqft' in df.columns and 'area' in df.columns:
            df['price_per_sqft'] = df['total_sqft'] / df['area']
        if 'total_rooms' in df.columns and 'area' in df.columns:
            df['room_density'] = df['total_rooms'] / df['area']

        # Масштабування
        features = scaler.transform(df)

        # Прогноз
        prediction = price_model.predict(features)
        
        confidence = price_model.predict_proba(features) if hasattr(price_model, 'predict_proba') else None

        return jsonify({
            'predicted_price': float(prediction[0]),
            'price_range': {
                'min': float(prediction[0] * 0.95),
                'max': float(prediction[0] * 1.05)
            },
            'confidence': float(confidence[0][1]) if confidence is not None else 0.85
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)