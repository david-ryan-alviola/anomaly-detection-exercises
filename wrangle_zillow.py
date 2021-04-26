import env
import utilities as utils

from acquire.acquire_zillow import acquire_zillow_df
from prepare.prepare_zillow import drop_unnecessary_columns, handle_missing_zillow_values, rename_zillow_columns, encode_zillow_categoricals, add_zillow_features, remove_zillow_outliers

def wrangle_zillow_data():
    """
    Returns the train, validate, test samples of the prepared zillow data frame. Samples are stratified by 'error'.
    """
    zillow_df = acquire_zillow_df()
    
    zillow_df = handle_missing_zillow_values(zillow_df)
    zillow_df = rename_zillow_columns(zillow_df)
    zillow_df = add_zillow_features(zillow_df)
    zillow_df = encode_zillow_categoricals(zillow_df)
    zillow_df = drop_unnecessary_columns(zillow_df)
    zillow_df = remove_zillow_outliers(zillow_df)
    
    return utils.split_dataframe(zillow_df)