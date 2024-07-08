#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:29:17 2024

@author: jamesturner
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV file into a DataFrame
credit_card_data = pd.read_csv('/Users/jamesturner/Downloads/creditcard.csv')

# Display the first few rows of the DataFrame
print(credit_card_data.head())

# Checking for missing values in each column and the distribution of the data
missing_values = credit_card_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

data_description = credit_card_data.describe()
print("\nSummary of the distribution of data:\n", data_description)

# Checking the class distribution
class_counts = credit_card_data['Class'].value_counts()
print("\nValue counts of 'Class' column:\n", class_counts)

# Separating the data for analysis
legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]

# Statistical measures of the 'Amount' column for legitimate transactions
legit_amount_desc = legit['Amount'].describe()
print("\nStatistical measures of 'Amount' for legitimate transactions:\n", legit_amount_desc)

# Statistical measures of the 'Amount' column for fraudulent transactions
fraud_amount_desc = fraud['Amount'].describe()
print("\nStatistical measures of 'Amount' for fraudulent transactions:\n", fraud_amount_desc)

# Combine the two summaries into a single DataFrame for comparison
summary_df = pd.DataFrame({
    "Legitimate": legit_amount_desc,
    "Fraudulent": fraud_amount_desc
})
print("\nCombined statistical measures of 'Amount':\n", summary_df)

# Resampling the dataset to address class imbalance
legit_sample = legit.sample(n=fraud.shape[0], random_state=1)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset['Class'].value_counts())

# Feature and target split
X = new_dataset.drop(columns='Class')
Y = new_dataset['Class']

# Train-test split with stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

print(f"Shapes - X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Training the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy on training data
training_data_accuracy = model.score(X_train, Y_train)
print('Accuracy on Training data:', training_data_accuracy)

# Accuracy on test data
test_data_accuracy = model.score(X_test, Y_test)
print('Accuracy score on Test Data:', test_data_accuracy)