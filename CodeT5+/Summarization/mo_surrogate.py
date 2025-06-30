import random
import logging

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge, PoissonRegressor, Perceptron, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class SurrogateModel(BaseEstimator, RegressorMixin):
    def __init__(self, accuracy_model_class=GradientBoostingRegressor, energy_model=None):
        self.accuracy_model_class = accuracy_model_class
        self.gpu_energy_model = None
        self.cpu_energy_model = None
        self.accuracy_model = accuracy_model_class()
        self.scaler = StandardScaler()

    def fit(self, dataset):
        X_train, y_accuracy, y_energy_gpu, y_energy_cpu = dataset[0], dataset[1], dataset[2], dataset[3]

        # Scale features
        X_train = self.scaler.fit_transform(X_train)

        # Fit surrogate model for accuracy
        y_accuracy = np.clip(y_accuracy, 1e-12, 1 - 1e-12)  # Ensure no values exactly at 0 or 1
        self.accuracy_model.fit(X_train, logit(y_accuracy))
        accuracy_preds = expit(self.accuracy_model.predict(X_train))
        logging.info("Accuracy Model MAE: {}".format(mean_absolute_error(y_accuracy, accuracy_preds)))

        # Fit surrogate model for GPU energy
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25],
            'max_depth': [2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        }

        # Implement grid search with cross-validation
        gbr = GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3,
                                   scoring='neg_mean_absolute_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_energy_gpu)

        # Get the best parameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        self.gpu_energy_model = best_model
        self.gpu_energy_model.fit(X_train, y_energy_gpu)
        energy_preds = self.gpu_energy_model.predict(X_train)
        logging.info("GPU Energy Model MAE: {}".format(mean_absolute_error(y_energy_gpu, energy_preds)))


        # Fit surrogate model for CPU energy
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25],
            'max_depth': [2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        }

        # Implement grid search with cross-validation
        gbr = GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3,
                                   scoring='neg_mean_absolute_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_energy_cpu)

        # Get the best parameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        self.cpu_energy_model = best_model
        self.cpu_energy_model.fit(X_train, y_energy_cpu)
        energy_preds = self.cpu_energy_model.predict(X_train)
        logging.info("CPU Energy Model MAE: {}".format(mean_absolute_error(y_energy_cpu, energy_preds)))

        return self

    def predict_accuracy(self, X):
        X = self.scaler.transform(X)
        return expit(self.accuracy_model.predict(X))

    def predict_gpu_energy(self, X):
        X = self.scaler.transform(X)
        # Predict and ensure non-negative integer values
        return np.maximum(0, np.round(self.gpu_energy_model.predict(X))).astype(int)
    
    def predict_cpu_energy(self, X):
        X = self.scaler.transform(X)
        # Predict and ensure non-negative integer values
        return np.maximum(0, np.round(self.cpu_energy_model.predict(X))).astype(int)


