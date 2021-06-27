# -*- coding: utf-8 -*-
"""Untitled.ipynb


We will perform regression operation kc_house_dataset.
"""

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#  Lets load the data

location = "/home/kamrul/Documents/"
data = pd.read_csv(location + "kc_house_data.csv")

#  Overview of the data

data.info()

data.head()

#  Lets drop the column 'id' as it is not useful

data = data.drop('id', axis=1)

data.head()

#  Lets find out if there is null value

data.isnull().sum()

#  So we have two null values on column sqft_above.We can actually drop these two row and it shouldn't hamper the performance of our model

data = data.dropna()

#  Column date is pretty useless, lets transform it into Year format and find out the age when the house was sold

data['date'] = pd.to_datetime(data['date'])
data['Year'] = data['date'].apply(lambda date: date.year)

data.head()

data['Age'] = data['Year'] - data['yr_built']

data.head()

#  Now lets drop those redundant columns

data.drop('date',inplace=True,axis=1)
data.drop('yr_built',inplace=True,axis=1)
data.drop('Year',inplace=True,axis=1)

data.head()

#  If we visualize the density of price we can observe that there remains outliers over 2million

plt.figure(figsize=(10,6))
sns.histplot(data['price'])

#  Now, let's see how the price changed along the years.it is quite clear that age don't have much of a effect on price


plt.figure(figsize=(10,6))
sns.scatterplot(x='Age',y='price',data= data)

#  The heat map below gives us the clear idea of how each feature affecting the traget variable Price

plt.figure(figsize=(16,7))
sns.heatmap(data.corr(),annot=True)

data.head()

#  Lets choose our features and target. and then we will split our entire dataset into train and test set(70%/30%)

X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition',
       'grade','sqft_above','sqft_basement','sqft_living15','sqft_lot15']].values
y = data['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#  we create pipeline and provide paramaters so that gridsearch can find the best estimator for us

pipe = Pipeline([('scaler', StandardScaler()),
                ('regressor', LinearRegression())])
param_grid = [
    {'regressor':[LinearRegression()],
     'scaler':[StandardScaler(), MinMaxScaler()]
    },
    {
        'regressor':[Ridge()],
        'scaler':[StandardScaler(), MinMaxScaler()],
        'regressor__alpha':[0.1, 1, 10]
    },
    {
        'regressor':[RandomForestRegressor()],
        'scaler':[StandardScaler()],
        'regressor__n_estimators':[100]
    }
]

#  Now lets search over the parameters and find the best estimator

grid = GridSearchCV(pipe, param_grid, n_jobs=-1)
grid.fit(X_train, y_train)
r2 = grid.score(X_test, y_test)
print(r2)

#  And here is our best estimator

print(grid.best_estimator_)

#  Lets find mae,mse value for our model.

model = grid.best_estimator_
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print('r2: ', r2)
print('mae: ', mae)
print('mse: ', mse)

