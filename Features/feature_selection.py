from sklearn.base import BaseEstimator, TransformerMixin

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        # Constructor for the DropFeatures class.
        # Parameters:
        #   features_to_drop (list or array-like): List of column names to be dropped from the DataFrame.
        self.features_to_drop = features_to_drop

    def fit(self, df):
        # Fit method for the transformer.
        # Since dropping features does not require fitting any parameters,
        # this method just returns self.
        # Parameters:
        #   df (DataFrame): The DataFrame on which to apply the transformer.
        # Returns:
        #   self: Returns an instance of itself.
        return self

    def transform(self, df):
        # Transform method for the transformer.
        # This method applies the dropping of columns to the DataFrame.
        # Parameters:
        #   df (DataFrame): The DataFrame from which the specified features are to be dropped.
        # Modifies:
        #   df: The DataFrame with the specified features removed.
        # Returns:
        #   df: The modified DataFrame with specified columns dropped.
        df.drop(self.features_to_drop, axis=1, inplace=True)
        return df
