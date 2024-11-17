import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('dataset1_test.csv')

# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())

# Convert categorical data ('Dismissal' and 'Pos') into numerical data
df['Dismissal'] = df['Dismissal'].astype('category').cat.codes
df['Pos'] = df['Pos'].astype('category').cat.codes
plt.show()
# Handling any other missing data (if any) by filling with the mean or dropping rows
df.fillna(df.mean(), inplace=True)

# Feature selection
X = df[['Mins', 'BF', '4s', '6s', 'SR', 'Pos', 'Dismissal', 'Inns']]  # Independent variables
y = df['Runs']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')



# Box plot of Runs
plt.figure(figsize=(8,6))
sns.boxplot(x=df['Runs'])
plt.title('Boxplot of Runs')
plt.show()

# Histogram of Runs
plt.figure(figsize=(8,6))
sns.histplot(df['Runs'], kde=True)
plt.title('Histogram of Runs')
plt.show()

# Scatter plot of Runs vs Balls Faced
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['BF'], y=df['Runs'])
plt.title('Scatter Plot of Runs vs Balls Faced')
plt.xlabel('Balls Faced')
plt.ylabel('Runs')
plt.show()

# Compute the Pearson correlation matrix
corr_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()

# Compare actual vs predicted Runs
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# You can also plot the predictions vs actuals
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs Predicted Runs')
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.show()







