AI_Phase5
https://drive.google.com/file/d/1T2pmXIk8jA-edonn8nn6M_oyFNAs4vYI/view?usp=drivesdk
# Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Load your dataset

data = pd.read_csv('your_dataset.csv')

# Assume 'target' is your target variable

X = data.drop('target', axis=1)

y = data['target']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, depending on the algorithm used)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Output the preprocessed data (optional)

print("Preprocessed Training Data:")

print(X_train.head())
# Import the regression model you want to use

from sklearn.linear_model import LinearRegression

# Create an instance of the model

model = LinearRegression()

# Train the model on the training set

model.fit(X_train, y_train)

# Output a message indicating the completion of training

print("Model trained successfully.")
# Import necessary metrics for evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the test set

y_pred = model.predict(X_test)

# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred, squared=False)

r2 = r2_score(y_test, y_pred)

# Print or log the evaluation metrics

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

print(f'Root Mean Squared Error: {rmse}')

print(f'R-squared: {r2}')
