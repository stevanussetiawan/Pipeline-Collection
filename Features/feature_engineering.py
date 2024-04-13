import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X[self.feat_with_days] = np.abs(X[self.feat_with_days])
        return X
        
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness):
        self.feat_with_skewness = feat_with_skewness
    def fit(self, df):
        return self
    def transform(self, df):
        df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
        return df
        
class BinningNum(BaseEstimator, TransformerMixin):
    def __init__(self, features_bin):
        self.features_bin = features_bin
    def fit(self, df):
        return self
    def transform(self, df):
        for ft in self.features_bin:
            df[ft] = df[ft].map({1:'Y',0:'N'})
        return df
            
class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self, df):
        return self
    def transform(self, df):
        ordinal_enc = OrdinalEncoder()
        df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
        return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self, df):
        return self
    def transform(self, df):
        min_max_enc = MinMaxScaler()
        df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
        return df
    
class NumTarget(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target
    def fit(self, df):
        return self
    def transform(self, df):
        df[self.target] = pd.to_numeric(df[self.target])
        return df
    
class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self, target):
        self.target = target
    def fit(self, df):
        return self
    def transform(self, df):
        oversample = SMOTE(sampling_strategy='minority')
        X_balanced, y_balanced = oversample.fit_resample(df.loc[:, df.columns != self.target],df[self.target])
        df_balanced = pd.concat([pd.DataFrame(X_balanced),pd.DataFrame(y_balanced)],axis=1)
        return df_balanced
        
class OneHotEnc(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft):
        self.one_hot_enc_ft = one_hot_enc_ft
        
    def fit(self, df):
        return self

    def one_hot_enc(self, df, one_hot_enc_ft):
        one_hot_enc = OneHotEncoder()
        one_hot_enc.fit(df[one_hot_enc_ft])
        feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
        df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),columns=feat_names_one_hot_enc,index=df.index)
        return df  
    
    def concat_onehot(self, df, one_hot_enc_df, one_hot_enc_ft):
        rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
        df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]],axis=1)
        return df_concat    
    
    def transform(self, df):
        one_hot_enc_df = self.one_hot_enc(df,self.one_hot_enc_ft)
        concat_one_hot = self.concat_onehot(df,one_hot_enc_df,self.one_hot_enc_ft)
        return concat_one_hot
    

    

    
    
    
