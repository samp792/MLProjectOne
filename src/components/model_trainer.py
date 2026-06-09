import os
import sys
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Booster":GradientBoostingRegressor(),
                "XGB Regressor":XGBRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "Catboost Regressor":CatBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, 
                                             y_test=y_test, models=models)

            #Best model score returned from model_report
            best_model_score = max(sorted(model_report.values()))

            #Best Model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and test data set")
            

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as ex:
            raise CustomException(ex, sys)