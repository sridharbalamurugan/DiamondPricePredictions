import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging
from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('splitting Dependent and Independent variables from train and test')
           
            X_train, y_train, X_test, y_test = (
                         train_array[:, :-1],  # Features
                         train_array[:, -1],   # Labels (target)
                         test_array[:, :-1],   # Features
                         test_array[:, -1]     # Labels (target)
)

            
            
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor(),
        }
            
            model_report: dict = evaluate_model(X_train, X_test, y_train, y_test, models)
            print(model_report)
            print('\n==================================================================')
            logging.info(f'Model Report :{model_report}')
            
            
            # To get Best model score from dictionary
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)        
            ]
            
            best_model = models[best_model_name]
            
            print(f'Best Model Found, Model Name: {best_model_name} , R2 score :{best_model_score}')
            print('\n=======================================================')
            logging.info(f'Best Model Found, Model Name: {best_model_name},R2 score :{best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info('Exception occured at model training')
            raise CustomException(e,sys)