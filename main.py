import pandas as pd

# Load the dataset
df = pd.read_csv('insurance.csv')
print(df.head())
print(df.info())

print(df.isnull().sum())  # Check for null values
print(df.describe())      # Summary statistics

df_encoded = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
print("RMSE:", rmse)


import matplotlib.pyplot as plt

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()
