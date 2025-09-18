from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import HeartAttackModel

app = FastAPI(title="Heart Attack Risk Prediction API", 
              description="API для предсказания риска сердечного приступа", 
              version="1.0.0")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели при запуске приложения
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'heart_attack_model.pkl')
if not os.path.exists(model_path):
    raise Exception(f"Модель не найдена по пути: {model_path}. Сначала обучите модель.")
    
model = HeartAttackModel().load(model_path)

class PredictionRequest(BaseModel):
    data: dict  # Принимаем словарь вместо списка

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list

@app.get("/")
async def root():
    return {"message": "Heart Attack Risk Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    try:
        # Преобразуем данные в DataFrame
        data_df = pd.DataFrame([request.data])
        
        # Делаем предсказание
        prediction = model.predict(data_df)[0]
        probabilities = model.predict_proba(data_df)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file_path: str):
    try:
        # Загрузка данных из CSV
        data_df = pd.read_csv(file_path)
        
        # Проверяем наличие колонки id
        if 'id' not in data_df.columns:
            raise HTTPException(status_code=400, detail="CSV файл должен содержать колонку 'id'")
        
        # Делаем предсказания
        predictions = model.predict(data_df.drop('id', axis=1))
        
        # Создаем результат
        result = []
        for i, (idx, pred) in enumerate(zip(data_df['id'], predictions)):
            result.append({
                "id": int(idx),
                "prediction": int(pred)
            })
        
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model.is_trained}