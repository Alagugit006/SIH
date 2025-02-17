import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('C:/Users/BOOBATHIRAJA.K.M/Desktop/SIH upload/fuel_tank_explosions.csv')

# Split the data into features and target
features = data[['volume', 'fuel_type', 'temperature', 'pressure']]
target = data['radius_of_fire']

# Convert 'fuel_type' categorical data to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.50, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = np.mean((y_pred - y_test) ** 2)

# Print the mean squared error
print('Mean squared error:', mse)

# Now, you can proceed with making a prediction for a new fuel tank explosion:

# Prompt the user to input data for a new fuel tank explosion
V = float(input('Enter the Volume: '))
print('Enter 1 for Yes, 0 for No')
D = int(input('Is the fuel Diesel? : '))
G = int(input('Is the fuel Gasoline? : '))
K = int(input('Is the fuel Kerosene? : '))
T = float(input('Enter the Temperature: '))
P = float(input('Enter the Pressure: '))

# Create a DataFrame for the new fuel tank explosion
new_fuel_tank_explosion = pd.DataFrame({
    'volume': [V],
    'fuel_type_Diesel': [D],
    'fuel_type_Gasoline': [G],
    'fuel_type_Kerosene': [K],
    'temperature': [T],
    'pressure': [P]
})

# Ensure the columns order matches the order in the training data
new_fuel_tank_explosion = new_fuel_tank_explosion.reindex(columns=X_train.columns, fill_value=0)

# Predict the radius of fire for the new fuel tank explosion
radius_of_fire = model.predict(new_fuel_tank_explosion)

# Print the predicted radius of fire
print('Predicted radius of fire:', radius_of_fire)