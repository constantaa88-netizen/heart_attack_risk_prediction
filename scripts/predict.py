import pandas as pd
import sys
import os

# Добавляем путь к src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import HeartAttackModel

def main():
    # Загрузка тестовых данных
    test_df = pd.read_csv('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/data/test.csv')
    
    print("Столбцы в тестовых данных:", test_df.columns.tolist())
    print("Размер тестовых данных:", test_df.shape)
    
    # Сохраняем id для результата
    if 'id' not in test_df.columns:
        raise ValueError("Тестовые данные должны содержать столбец 'id'")
    
    test_ids = test_df['id']
    
    # Удаляем столбец 'id' для предсказания
    X_test = test_df.drop('id', axis=1)
    
    # Загрузка модели
    model = HeartAttackModel().load('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/models/heart_attack_model.pkl')
    
    # Предсказание
    predictions = model.predict(X_test)
    
    # Создание DataFrame с предсказаниями
    result_df = pd.DataFrame({
        'id': test_ids,
        'prediction': predictions
    })
    
    # Сохранение результатов
    os.makedirs('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/predictions', exist_ok=True)
    result_df.to_csv('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/predictions/predictions.csv', index=False)
    print("Предсказания сохранены в predictions/predictions.csv")
    
    # Вывод первых 5 предсказаний
    print("Первые 5 предсказаний:")
    print(result_df.head())

if __name__ == "__main__":
    main()