# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
sc_X = sc_X.fit_transform(X)
sc_y = sc_y.fit_transform(y)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
#sc_X.transform(np.array([[6.5]])) is used because dataset X is feature scaled
#sc_X is only transformed because it dataset X is already fitted
#[]for vector & [[]] for singleton matrix
#np.array to convert numerical value to matrix
from scipy import sparse

A = np.array([[6.5]])
sparse_matrix1 = sparse.csr_matrix(A)
#transform returns sparse matrix
y_pred = sc_y.inverse_transform( regressor.predict( sc_X.transform( np.array([[6.5]]) ) ) )

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
