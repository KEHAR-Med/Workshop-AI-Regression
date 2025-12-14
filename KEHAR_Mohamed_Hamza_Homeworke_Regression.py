#kEHAR Mohamed Hamza_Homework_Regression
print('*'*20 + ' KEHAR Mohamed Hamza_Homework_Regression ' + '*'*20,'\n')

# Import libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

#I changed the code here to use the local CSV file instead of scikit-learn dataset

#Load Data set
data = pd.read_csv(r'C:\Users\DELL\Desktop\homework.csv') #File path
columns = ['rownames', 'price', 'lotsize', 'bedrooms',
           'bathrms', 'stories','driveway', 'recroom', 'fullbase', 'gashw', 'airco','garagepl','prefarea']
print('Colums : ',columns,'\n')
#*******************************************************************************************************************
print('*'*150)
print('*'*150)

# I)Simple linear regression:
print('Simple linear regression :', '\n')
print('Selected feature X = lotsize')
print('Selected target Y = price', '\n')
#I had to reshape X to 2D to avoid error in  thecase of single features
X = data['lotsize'].values.reshape(-1, 1)
y = data['price'].values



# Split into train & test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Multivariate Linear Regression (Real Data)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

#Polynomial Regression
#We apply polynomial transformation on 1 feature for visualization.
# Use only feature 0 (MedInc) for visualization
X_poly = X[:, [0]]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Degree = 3 polynomial
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_p)
X_test_poly = poly.transform(X_test_p)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_p)

y_pred_poly = model_poly.predict(X_test_poly)

print("Polynomial Regression (degree=3)")
print("MSE:", mean_squared_error(y_test_p, y_pred_poly))
print("R²:", r2_score(y_test_p, y_pred_poly))


#3. Regression with Gradient Descent (From Scratch)
#We use only one feature to simplify the demonstration.
# Use feature 0 only
X_gd = X[:, [0]]
y_gd = y

# Normalize
X_norm = (X_gd - np.mean(X_gd)) / np.std(X_gd)

# Add bias term
X_b = np.c_[np.ones((len(X_norm), 1)), X_norm]

# Initialize parameters
theta = np.zeros((2, 1))

# Hyperparameters
alpha = 0.01
epochs = 1000

# Gradient Descent
for i in range(epochs):
    gradients = (2/len(X_b)) * X_b.T.dot(X_b.dot(theta) - y_gd.reshape(-1, 1))
    theta -= alpha * gradients

print("Gradient Descent (from scratch)")
print("Estimated parameters (theta):", theta.ravel())

#4. Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

print("Lasso Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))

#5. Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

print("Ridge Regression")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))

#6. Elastic Net Regression (L1 + L2 Regularization)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

y_pred_elastic = elastic.predict(X_test)

print("Elastic Net Regression")
print("MSE:", mean_squared_error(y_test, y_pred_elastic))
print("R²:", r2_score(y_test, y_pred_elastic))

print()
#*******************************************************************************************************************
print('*'*150)
print('*'*150)

# II)Multiple linear regression:

print('Multiple linear regression :', '\n')
print('Selected features X = lotsize, bedrooms, bathrms, stories, garagepl')
print('Selected target Y = price', '\n')

#I only selected features with numerical values to avoid conversion errors

selected_features = ['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']
X = data[selected_features].values
y = data['price'].values



# Split into train & test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Multivariate Linear Regression (Real Data)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

#Polynomial Regression
#We apply polynomial transformation on 1 feature for visualization.
# Use only feature 0 (MedInc) for visualization
X_poly = X[:, [0]]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Degree = 3 polynomial
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_p)
X_test_poly = poly.transform(X_test_p)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_p)

y_pred_poly = model_poly.predict(X_test_poly)

print("Polynomial Regression (degree=3)")
print("MSE:", mean_squared_error(y_test_p, y_pred_poly))
print("R²:", r2_score(y_test_p, y_pred_poly))


#3. Regression with Gradient Descent (From Scratch)
#We use only one feature to simplify the demonstration.
# Use feature 0 only
X_gd = X[:, [0]]
y_gd = y

# Normalize
X_norm = (X_gd - np.mean(X_gd)) / np.std(X_gd)

# Add bias term
X_b = np.c_[np.ones((len(X_norm), 1)), X_norm]

# Initialize parameters
theta = np.zeros((2, 1))

# Hyperparameters
alpha = 0.01
epochs = 1000

# Gradient Descent
for i in range(epochs):
    gradients = (2/len(X_b)) * X_b.T.dot(X_b.dot(theta) - y_gd.reshape(-1, 1))
    theta -= alpha * gradients

print("Gradient Descent (from scratch)")
print("Estimated parameters (theta):", theta.ravel())

#4. Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

print("Lasso Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))

#5. Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

print("Ridge Regression")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))

#6. Elastic Net Regression (L1 + L2 Regularization)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

y_pred_elastic = elastic.predict(X_test)

print("Elastic Net Regression")
print("MSE:", mean_squared_error(y_test, y_pred_elastic))
print("R²:", r2_score(y_test, y_pred_elastic))