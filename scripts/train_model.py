import pandas as pd
import sys
import os

# Добавляем путь к src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import HeartAttackModel

def main():
    # Загрузка данных
    train_df = pd.read_csv('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/data/heart_train.csv')
    
    print("Столбцы в данных:", train_df.columns.tolist())
    print("Размер данных:", train_df.shape)
    
    # Определяем целевой столбец
    target_column = 'Heart Attack Risk (Binary)'
    
    # Убедимся, что столбец существует
    if target_column not in train_df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' не найден в данных")
    
    # Удаляем столбец 'id' так как он не должен использоваться для обучения
    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)
    
    # Разделение на признаки и целевую переменную
    X = train_df.drop(target_column, axis=1)
    y = train_df[target_column]
    
    print(f"Признаки: {X.shape}, Целевая переменная: {y.shape}")
    
    # Обучение модели
    model = HeartAttackModel()
    model.train(X, y)
    
    # Сохранение модели
    model.save('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/models/heart_attack_model.pkl')
    print("Модель успешно обучена и сохранена в models/heart_attack_model.pkl")

if __name__ == "__main__":
    main()