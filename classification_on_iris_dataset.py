# -*- coding: utf-8 -*-
"""classification_on_iris_dataset


"""

import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



location = "/home/kamrul/Documents/"
data = pd.read_csv(location + "IRIS.csv")

#  Checked out the shape/size of our dataset

data.shape

#  Checking over all state of our dataset.

data.describe()

data.head()

# Checks if there is null value

data.isnull().sum()

#  Checks numbers of each species[target variable]

data['species'].value_counts()

#  Plots a heatmap showing the co relation matrix. 

plt.figure(figsize=(16,7))
sns.heatmap(data.corr(),annot=True)

#  Plots a pairplot showing important features

sns.pairplot(data,hue='species',height=3)

#  Currently target values are string which is not useful.Lets convert our target variable from string to int.

le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

data['species']

#  lets define our features and target variable.

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = data[['species']].values
y = np.ravel(y) #flatten the target y for working with RandomForestRegressor

#  Spliting entire dataset for training and testing purpose(80%/20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  Defining pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

#  Lets set some classifiers and parameters.

param_grid = [
    {
        'classifier':[RandomForestClassifier()],
        'scaler':[StandardScaler(), MinMaxScaler()],
        'classifier__n_estimators':[30, 40 , 50]
    },
    {
        'classifier':[KNeighborsClassifier()],
        'scaler':[StandardScaler(), MinMaxScaler()],
        'classifier__n_neighbors':[5, 7, 9]
    },
    {
        'classifier':[LogisticRegression()],
        'scaler':[StandardScaler(), MinMaxScaler()]
    }
]

#  Performing gridsearch and showing the best estimator

grid = GridSearchCV(pipe, param_grid, n_jobs=-1)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)

grid.best_estimator_

#  Lets see how our model performs various on performance measurement scale

model = grid.best_estimator_
y_predicted = model.predict(X_test)
cm=confusion_matrix(y_test,y_predicted)
print(classification_report(y_test, y_predicted))

#  lets plot a confusion matrix for better understanding of our models performance</h5>"""

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

