# CODSOFT-Movie-Rating-Prediction
# Step 1: Data Preprocessing.
import pandas as pd

# Loading the dataset
file_path = "IMDb Movies India.csv"  
data = pd.read_csv(file_path)

# Dropping rows with missing values
data.dropna(inplace=True)

# Step 2: Feature Engineering
# Since we're predicting movie ratings, we'll use the Genre, Director, and Actor columns as features.
# Converting categorical features to dummy variables.
data = pd.get_dummies(data, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])

# Separating features (X) & target variable (y)
X = data.drop(columns=['Name', 'Year', 'Rating'])  # Features
y = data['Rating']  # Target variable

# Step 3: Model Selection:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 4: Model Training:
# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation:
# Making predictions on the testing set.
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
