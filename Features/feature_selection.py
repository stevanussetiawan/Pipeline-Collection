from sklearn.base import BaseEstimator, TransformerMixin

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    def fit(self, df):
        return self
    def transform(self, df):
        df.drop(self.features_to_drop,axis=1,inplace=True)
        return df
        
        
