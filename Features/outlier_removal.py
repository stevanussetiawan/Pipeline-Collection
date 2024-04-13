from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_outliers):
        self.feat_with_outliers = feat_with_outliers
    def fit(self,df):
        return self
    def transform(self,df):
        # 25% quantile
        Q1 = df[self.feat_with_outliers].quantile(.25)
        # 75% quantile
        Q3 = df[self.feat_with_outliers].quantile(.75)
        IQR = Q3 - Q1
        # keep the data within 3 IQR
        df = df[~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) |(df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
        return df
