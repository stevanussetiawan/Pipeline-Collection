import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error

numerical_features = []
categorical_features = []

class modelMl:
    def __init__(self, numeric_transformer, categorical_features, X, y):
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])

        categorical_transformer  = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numerical_features),
                ('categoric', categorical_transformer, categorical_features)
            ]
        )
        
        self.mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def linear_regression_model_training_and_save(self):

        linear_reg = Pipeline(steps=[
                        ('preprocessor', self.preprocessor),
                        ('linreg', LinearRegression())
                    ])
        
        param_dist_lr = {
            'linreg__fit_intercept': [True, False],
            'linreg__normalize': [True, False],
            'linreg__copy_X': [True, False]
        }
        
        linear_reg_random = RandomizedSearchCV(estimator=linear_reg,
                                               param_distributions=param_dist_lr, 
                                               scoring=self.mse_scorer, 
                                               cv=3,
                                               random_state=101)
        
        linear_reg_random.fit(self.X_train, self.y_train)
        joblib.dump(linear_reg_random.best_estimator_, 'models/linear_regression_model.joblib')
        return linear_reg_random
    
    def elasticnet_regression_model_training_and_save(self):
        
        elasticnet_reg = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('elasticnet', ElasticNet())
        ])
        
        param_dist_elasticnet = {
            'elasticnet__alpha': np.logspace(-4, 4, 100),  # Regularization parameter
            'elasticnet__l1_ratio': np.linspace(0, 1, 50),   # Mixing parameter (l1_ratio = 0 corresponds to L2 penalty, l1_ratio = 1 to L1 penalty)
        }
        
        elasticnet_reg_random = RandomizedSearchCV(estimator=elasticnet_reg,
                                               param_distributions=param_dist_elasticnet,
                                               scoring=self.mse_scorer,
                                               cv=3,
                                               random_state=101)
                
        elasticnet_reg_random.fit(self.X_train, self.y_train)
        joblib.dump(elasticnet_reg_random, 'models/elasticnet_reg_random_model.joblib')
        
    def random_forest_regression_training_and_save(self):
        random_forest_reg = Pipeline([
            ('preprocessor', self.preprocessor),
            ('rf_reg', RandomForestRegressor())
        ])
            
        param_dist_rf = {
            'rf_reg__n_estimators': [50, 100, 200, 300],
            'rf_reg__max_depth': [None, 5, 10, 15, 20],
            'rf_reg__min_samples_split': [2, 5, 10],
            'rf_reg__min_samples_leaf': [1, 2, 4],
            'rf_reg__max_features': ['auto', 'sqrt', 'log2']
        }
        
        random_forest_reg_random = RandomizedSearchCV(estimator=random_forest_reg,
                                                      cv=3,
                                                      param_distributions=param_dist_rf,
                                                      random_state=101,
                                                      scoring=self.mse_scorer)
        
        random_forest_reg_random.fit(self.X_train, self.y_train)
        joblib.dump(random_forest_reg_random, 'models/random_forest_model.joblib')

    def xgb_regression_training_and_save(self):
        pipeline_xgb_reg = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("xgb_regressor", XGBRegressor())
        ])
        
        param_dist_xgb_regressor = {
            'xgb_regressor__learning_rate': np.linspace(0.01, 0.3, 30),  # Step size shrinkage to prevent overfitting
            'xgb_regressor__n_estimators': [100, 200, 300, 400, 500],       # Number of boosting rounds
            'xgb_regressor__max_depth': [3, 4, 5, 6, 7],                    # Maximum depth of a tree
            'xgb_regressor__subsample': np.linspace(0.5, 1.0, 6),           # Fraction of samples used for fitting the individual base learners
            'xgb_regressor__colsample_bytree': np.linspace(0.5, 1.0, 6),     # Fraction of features used for fitting the individual base learners
            'xgb_regressor__reg_alpha': np.logspace(-4, 4, 9),               # L1 regularization term on weights
            'xgb_regressor__reg_lambda': np.logspace(-4, 4, 9)               # L2 regularization term on weights
        }
        
        pipeline_xgb_reg_random = RandomizedSearchCV(estimator=pipeline_xgb_reg,
                                                     cv=3,
                                                     random_state=101,
                                                     scoring=self.mse_scorer,
                                                     param_distributions=param_dist_xgb_regressor)
        
        pipeline_xgb_reg_random.fit(self.X_train, self.y_train)
        joblib.dump(pipeline_xgb_reg_random, 'models/xgb_reg_model.joblib')