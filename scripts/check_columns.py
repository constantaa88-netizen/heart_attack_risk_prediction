import pandas as pd

# Загрузка данных
train_df = pd.read_csv('C:/Users/Pc4/Documents/GitHub/Hearts/heart_attack_risk_prediction/data/heart_train.csv')

print("Столбцы в тренировочных данных:")
print(train_df.columns.tolist())

print("\nПервые 5 строк:")
print(train_df.head())

print("\nИнформация о данных:")
print(train_df.info())