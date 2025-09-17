import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Создаем копию данных
    df_processed = df.copy()
    
    # Удаляем ненужные столбцы
    columns_to_drop = ['Unnamed: 0', 'id', 'Blood sugar', 'CK-MB', 'Troponin']
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Кодируем категориальные переменные
    le = LabelEncoder()
    df_processed['Gender'] = le.fit_transform(df_processed['Gender'])
    
    # Разделяем данные на числовые и категориальные признаки
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('Heart Attack Risk (Binary)')  # Исключаем целевую переменную
    
    # Заполняем пропущенные значения
    imputer = SimpleImputer(strategy='median')
    df_processed[numeric_features] = imputer.fit_transform(df_processed[numeric_features])
    
    # Нормализуем числовые признаки
    scaler = StandardScaler()
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    return df_processed

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('../data/heart_train.csv')
    
    # Предобработка
    df_processed = preprocess_data(df)
    
    # Сохранение обработанных данных
    df_processed.to_csv('../data/heart_train_processed.csv', index=False)
    
    print(f"Обработанные данные сохранены. Размер: {df_processed.shape}")
    print("Первые 5 строк обработанных данных:")
    print(df_processed.head())