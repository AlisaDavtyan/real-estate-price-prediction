import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import joblib


class HousePricePredictor:
    def __init__(self):
        self.preprocessor = None
        self.feature_selector = None
        self.selected_feat = None  
        self.best_model = None
        self.best_params_dict = {}  
        self.predictions_dict = {} 
        self.y_test = None
        self.predictions_df = None

    def preprocessing(self, df):
        df = df.rename(
            columns={"Առաստաղի բարձրություն": "Height", 
                    "Գին (Ք.Մ.)": "Price_sq", 
                    "Հարկ/Հարկանի":'Floor',
                    "Մակերես":"Square", 
                    "Շինության տիպը" : "Building_type",
                    "Սանհանգույց" :"Num_toil",
                    "Սենյակ":"Num_room",
                    "Վիճակը":"Condition",
                    "Գին" : "Price"})
    
        df = df.drop(df[df.Price_sq == 'Պայմ.'].index)
        df.drop(['ID','street', 'town','Unnamed: 0'],axis = 1,inplace= True)

        df['Height'] = df['Height'].str.replace(' Մ','').str.replace('+','').astype(float)
        df['Price_sq'] = df['Price_sq'].str.replace(',','').astype(float)
        df['Square'] = df['Square'].str.replace(' Ք.Մ.','').astype(float)
        df['Num_toil'] = df['Num_toil'].str.replace('+','').astype(float)
        df['Num_room'] = df['Num_room'].str.replace('+','').astype(float)
        df['Price'] = df['Price'].str.replace(',','').astype(float)
        df['Floor'] = df['Floor'].apply(lambda x: eval(x))
        return df

    def setup_pipeline(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        numeric_transform = Pipeline(
            steps=[
            ('imputer' , SimpleImputer(strategy='mean')),
            ('scaling', StandardScaler())
            ])
        
        text_transform = Pipeline(
            steps = [
                ('imputer' , SimpleImputer(strategy = 'most_frequent')), #changed to most_frequent, median is not always good for categorical
                ('Onehot', OneHotEncoder(drop='first' ,sparse_output=False, handle_unknown='ignore'))
            ])
        
        self.preprocessor = ColumnTransformer (
            transformers= [
                ('num', numeric_transform, numeric_cols) , 
                ('cath' ,text_transform, categorical_cols)
            ])
        
        self.feature_selector = Pipeline(
            steps = [('process',self.preprocessor),
                    ('feature_sel', Lasso())
            ])

    def select_features(self, X_train, y_train):
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        feature_names = self.preprocessor.get_feature_names_out()
        X_train_transf = pd.DataFrame(X_train_transformed, columns=feature_names)
        
        lasso = Lasso()
        lasso_params = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, 50, 100]}
        lasso_grid = GridSearchCV(lasso, lasso_params, scoring='neg_mean_squared_error', cv=10)
        lasso_grid.fit(X_train_transf, y_train)
        
        best_lasso = Lasso(alpha=lasso_grid.best_params_['alpha'], random_state=99)
        feature_sel_model = SelectFromModel(best_lasso)
        feature_sel_model.fit(X_train_transf, y_train)
        
        self.selected_feat = feature_names[feature_sel_model.get_support()] 
        return self.selected_feat
    
    def train_models(self,X_train, y_train, X_test, y_test):
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        feature_names = self.preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        
        X_train_selected = X_train_df[self.selected_feat] 
        X_test_selected = X_test_df[self.selected_feat]
        
        ridge_params = {'alpha' : np.linspace(0,1)}
        elastic_params = {'alpha': np.linspace(0,1) , 'l1_ratio':  np.linspace(0,1)} 
        knn_params = {'n_neighbors' : np.arange(1,21)}


        ridge = GridSearchCV(
                Ridge(), 
                ridge_params, 
                scoring='neg_mean_squared_error', 
                cv=10)
        
        elastic_net = GridSearchCV(
            ElasticNet(), 
            elastic_params,
            scoring='neg_mean_squared_error', 
            cv=10
        )

        knn = GridSearchCV(
            KNeighborsRegressor(),
            knn_params,
            scoring='neg_mean_squared_error', 
            cv=10
        )
        

        models = {
            "LinearRegression" :LinearRegression() ,
            "ElasticNet" : elastic_net,
            'Ridge' : ridge,
            'KNN' : knn
        }

        best_model = None
        best_score = float('inf')
        predictions_dict = {}
        best_params = float('inf')
        predictions_df = pd.DataFrame(index=X_test.index)

        for name, model in models.items():
            model.fit(X_train_selected, y_train)
            y_pred_train = model.predict(X_train_selected)  
            y_pred_test = model.predict(X_test_selected) 
            self.predictions_dict[name] = y_pred_test

            
            try:
                self.best_params_dict[name] = model.best_params_
            except AttributeError:
                self.best_params_dict[name] = "No hyperparameters (simple model)"

            print(f"{name} Best Params: {self.best_params_dict[name]}")
            print('Mean absolute error: %.2f'% mean_absolute_error(y_test, y_pred_test))
            print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred_test))
            print('Root mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test, y_pred_test)))
            print('R-squared: %.2f' % r2_score(y_test, y_pred_test))
            print('-------------------')

            score = mean_squared_error(y_test, y_pred_test)
            if score < best_score:
                best_score = score
                best_model = model

            y_pred_train = model.predict(X_train_selected)
            predictions_df[name] = y_pred_test

        self.best_model = best_model
        self.y_test = y_test
        self.predictions_df = predictions_df

        print(f'Best Model: {best_model} with MSE: {best_score}')
        
        predictions_df["Actual"] = y_test
        return predictions_df
            
        


    def predict(self, X):
        return self.best_model.predict(X) if self.best_model else None
    

    def plot_predictions(self):
        """Plots actual vs. predicted values for all models."""
        if self.predictions_df is None:
            print("No predictions found. Run `train_models` first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'orange', 'green', 'red']
        for i, (model_name, y_pred) in enumerate(self.predictions_dict.items()):
            plt.scatter(self.y_test, y_pred, alpha=0.5, label=model_name, color=colors[i % len(colors)])
        
        
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', label='Perfect Prediction')
        
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices (All Models)")
        plt.legend()
        plt.show()

    
    def save_model(self, filename="house_price_model.pkl"):
        """Saves the trained model, selected features, and preprocessor."""
        model_data = {
            "model": self.best_model,
            "preprocessor": self.preprocessor,
            "selected_features": self.selected_feat
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="house_price_model.pkl"):
        """Loads a saved model, preprocessor, and selected features."""
        model_data = joblib.load(filename)
        self.best_model = model_data["model"]
        self.preprocessor = model_data["preprocessor"]
        self.selected_feat = model_data["selected_features"]
        print(f"Model loaded from {filename}")


