import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('dados.csv')
corr = df.corr()
print("\nCorrelação com PE:")
print(corr['PE'].sort_values(ascending=False))


print("\nCorrelação entre variáveis independentes:")
print(corr[['AT','V','AP','RH']].loc[['AT','V','AP','RH']])

X = df[['AT','V','AP','RH']]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = num.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)