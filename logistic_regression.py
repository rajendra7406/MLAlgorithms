#Logistic regression

#Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 0)

#applying feature scaling to the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting the logistic regression model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test result
y_pred = classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#visualizing the training results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid( np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                      np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01) )
plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape) , 
            alpha=0.75, cmap = ListedColormap(('red','green')) )
plt.xlim( X1.min(),X2.max() )
plt.ylim( X1.min(),X2.max() )
for i, j in enumerate(np.unique(y_set)) :
    plt.scatter( X_set[y_set == j, 0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green'))(i), label=j)
plt.title('Classification of training results (Logistic Regression)')    
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()

#visualizing the test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid( np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                      np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01) )
plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape) , 
            alpha=0.75, cmap = ListedColormap(('red','green')) )
plt.xlim( X1.min(),X2.max() )
plt.ylim( X1.min(),X2.max() )
for i, j in enumerate(np.unique(y_set)) :
    plt.scatter( X_set[y_set == j, 0],X_set[y_set == j,1],
                 c = ListedColormap(('red','green'))(i), label=j)
plt.title('Classification of test results (Logistic Regression)')    
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()    