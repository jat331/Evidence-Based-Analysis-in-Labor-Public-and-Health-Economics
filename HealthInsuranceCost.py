#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:47:54 2024

@author: jamesturner
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:28:29 2024

@author: jamesturner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
insurance_dataset = pd.read_csv('/Users/jamesturner/Downloads/medical_insurance.csv')

# Display the first few rows and the shape of the dataset
print(insurance_dataset.head())
print("Shape of the dataset:", insurance_dataset.shape)

# Display the info and check for missing values
print("\nInfo of the dataset:")
insurance_dataset.info()
print("\nMissing values in each column:\n", insurance_dataset.isnull().sum())

# Display statistical description of the dataset
print("\nStatistical description of the dataset:\n", insurance_dataset.describe())

# Plotting distributions and counts
def plot_distribution(column, plot_type='hist', title=None, kde=True):
    plt.figure(figsize=(6, 6))
    if plot_type == 'hist':
        sns.histplot(insurance_dataset[column], kde=kde)
    elif plot_type == 'count':
        sns.countplot(x=column, data=insurance_dataset)
    plt.title(title or f'{column} Distribution')
    plt.show()

plot_distribution('age', title='Age Distribution')
plot_distribution('sex', plot_type='count', title='Sex Distribution')
plot_distribution('bmi', title='BMI Distribution')
plot_distribution('children', plot_type='count', title='Children Distribution')
plot_distribution('smoker', plot_type='count', title='Smoker Distribution')
plot_distribution('region', plot_type='count', title='Region Distribution')
plot_distribution('charges', title='Charges Distribution', kde=False)

# Encoding categorical columns
encoded_columns = {
    'sex': {'male': 0, 'female': 1},
    'smoker': {'yes': 0, 'no': 1},
    'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
}
insurance_dataset.replace(encoded_columns, inplace=True)

# Splitting the data into features and target
X = insurance_dataset.drop(columns='charges')
Y = insurance_dataset['charges']

print(X.head())
print(Y.head())

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(f"Shapes - X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Training the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediction on training data and calculating R squared value
r2_train = regressor.score(X_train, Y_train)
print('R squared value on Training data:', r2_train)

# Prediction on test data and calculating R squared value
r2_test = regressor.score(X_test, Y_test)
print('R squared value on Test Data:', r2_test)

# Predicting insurance cost for a sample input
input_data = (31, 1, 25.74, 0, 1, 0)
input_data_reshaped = np.array(input_data).reshape(1, -1)
prediction = regressor.predict(input_data_reshaped)
print('The insurance cost is USD', prediction[0])