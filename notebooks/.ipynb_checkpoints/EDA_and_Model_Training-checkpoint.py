# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Настройка отображения
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Загрузка данных
df = pd.read_csv('../data/heart_train.csv')
print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк:")
display(df.head())

# Базовая информация о данных
print("\nИнформация о данных:")
df.info()

print("\nОписательная статистика:")
display(df.describe())

# Проверка пропущенных значений
print("\nПропущенные значения:")
missing = df.isnull().sum()
print(missing[missing > 0])