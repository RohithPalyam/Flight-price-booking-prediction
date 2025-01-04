# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the Data
data = pd.read_csv("Flight_Booking.csv")  # Replace with actual dataset path
data = data.drop(['Unnecessary_Column'], axis=1)  # Adjust column name

# 2. Data Inspection
print(data.shape)
print(data.info())
print(data.describe())

# 3. Handling Missing Values
data = data.dropna()  # Drop rows with missing values

# 4. Data Visualization
sns.countplot(x='Airline', data=data)
plt.title("Count of Flights by Airline")
plt.show()

sns.boxplot(x='Class', y='Price', data=data)
plt.title("Price Range by Class")
plt.show()

sns.scatterplot(x='Days_left', y='Price', hue='Source_City', data=data)
plt.title("Price vs Days Left by Source City")
plt.show()

# 5. Encoding Categorical Features
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = encoder.fit_transform(data[['Airline', 'Source_City', 'Destination_City', 'Class']])
data = data.join(pd.DataFrame(encoded_features, index=data.index))
data = data.drop(['Airline', 'Source_City', 'Destination_City', 'Class'], axis=1)

# 6. Feature Selection
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Dropping high-VIF features (like 'Stops')
data = data.drop(['Stops'], axis=1)

# 7. Linear Regression Model
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Metrics
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# 8. Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Metrics
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))

# 9. Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Metrics
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
