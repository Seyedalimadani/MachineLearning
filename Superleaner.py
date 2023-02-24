import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

dataset_AT = pd.read_csv('data.csv') #reading the dataset
dataframe_AT = pd.DataFrame(dataset_AT) #turning it to pandas dataframe
#set x and y for a multinput and single target dataset
x = dataframe_AT.iloc[:, : -1].values
y = dataframe_AT.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)

class SuperLearner(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
            
        X_meta = np.column_stack([model.predict(X) for model in self.base_models])
        self.meta_model.fit(X_meta, y)
        
    def predict(self, X):
        X_meta = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_model.predict(X_meta)
        
# Example usage
base_models = [LinearRegression(), RandomForestRegressor(), KNeighborsRegressor()]
meta_model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam')
super_learner = SuperLearner(base_models, meta_model)
super_learner.fit(x_train, y_train)
y_pred = super_learner.predict(x_test)
print(mean_squared_error(y_test, y_pred))