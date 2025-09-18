import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from .preprocessor import HeartAttackPreprocessor

class HeartAttackModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.preprocessor = HeartAttackPreprocessor()
        self.is_trained = False
        
    def train(self, X, y):
        # Предобработка данных
        X_processed = self.preprocessor.fit_transform(X)
        
        # Обучение модели
        self.model.fit(X_processed, y)
        self.is_trained = True
        
        return self
        
    def predict(self, X):
        if not self.is_trained:
            raise Exception("Модель не обучена. Сначала вызовите метод train().")
            
        # Предобработка данных
        X_processed = self.preprocessor.transform(X)
        
        # Предсказание
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise Exception("Модель не обучена. Сначала вызовите метод train().")
            
        # Предобработка данных
        X_processed = self.preprocessor.transform(X)
        
        # Предсказание вероятностей
        return self.model.predict_proba(X_processed)
    
    def save(self, path):
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'is_trained': self.is_trained
        }, path)
        
    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.preprocessor = data['preprocessor']
        self.is_trained = data['is_trained']
        return self