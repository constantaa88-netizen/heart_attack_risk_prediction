import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class HeartAttackPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_cols = None
        self.categorical_cols = None
        
    def fit(self, X):
        # Определяем числовые и категориальные колонки
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Обучаем импьютер и скейлер на числовых данных
        self.imputer.fit(X[self.numeric_cols])
        self.scaler.fit(self.imputer.transform(X[self.numeric_cols]))
        
        return self
        
    def transform(self, X):
        # Копируем данные
        X_transformed = X.copy()
        
        # Обрабатываем числовые колонки
        if self.numeric_cols:
            imputed_data = self.imputer.transform(X[self.numeric_cols])
            scaled_data = self.scaler.transform(imputed_data)
            X_transformed[self.numeric_cols] = scaled_data
        
        # Обрабатываем категориальные колонки (one-hot encoding)
        if self.categorical_cols:
            X_transformed = pd.get_dummies(X_transformed, columns=self.categorical_cols)
        
        return X_transformed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)