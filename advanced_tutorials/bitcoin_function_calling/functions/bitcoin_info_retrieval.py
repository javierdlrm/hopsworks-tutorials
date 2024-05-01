import pandas as pd
from typing import Any, Dict, List
import datetime
import pandas as pd
import time
import inspect

def predict_price_for_specific_date(date: str, feature_view, model_deployment) -> pd.DataFrame:
    # predict price based on today's information
    # parsed_datetime = datetime.datetime.strptime(date, "%Y-%m-%d")
    parsed_datetime = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) 
    unix_datetime = int(time.mktime(parsed_datetime.timetuple()) * 1000) # converting todays datetime to unix

    response = model_deployment.predict(inputs=[unix_datetime])
    
    pred_encoded = pd.DataFrame(
        response["predictions"],
        columns=["close"],
    ) # since we applied transformation function to the 'close' columns,
      # now we need to provide a df with the same column name to decode.

    # decode features
    pred_decoded = decode_features(pred_encoded, feature_view=feature_view)
    pred_decoded = pred_decoded.rename(columns={"close": "predicted_price"})
    pred_decoded["date"] = [parsed_datetime.strftime("%Y-%m-%d")]
    
    return pred_decoded

def get_price_history_in_date_range(date_start: str, date_end: str, feature_view, model_deployment) -> pd.DataFrame:

    # Convert date strings to datetime objects
    date_start_dt = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
    date_end_dt = datetime.datetime.strptime(date_end, "%Y-%m-%d").date()
    if date_start_dt == date_end_dt:
        # if one-day range, increase end time one day so batch contains one row.
        date_end_dt = date_end_dt + datetime.timedelta(days=1)
    
    # Retrieve batch data for the specified date range
    batch_data = feature_view.get_batch_data(
        start_time=date_start_dt,
        end_time=date_end_dt# + datetime.timedelta(days=1),
    )

    # Parse unix datetime
    batch_data["date"] = batch_data["unix"].apply(convert_unix_to_date)
    batch_data.drop(columns=["unix"], inplace=True)
    
    # Filter dataframe
    batch_data = batch_data[['open', 'high', 'low', 'close', 'volume', 'date']]
    
    # Decode features
    batch_data = decode_features(batch_data, feature_view=feature_view)
    
    return batch_data.sort_values('date').reset_index(drop=True)

def transform_data(data, encoder):
    """
    Transform the input data by encoding the 'city_name' column and dropping unnecessary columns.
    
    Args:
    - data (DataFrame): Input data to be transformed.
    - encoder (LabelEncoder): Label encoder object to encode 'city_name'.
    
    Returns:
    - data_transformed (DataFrame): Transformed data with 'city_name_encoded' and dropped columns.
    """
    
    # Create a copy of the input data to avoid modifying the original data
    data_transformed = data.copy()
    
    # Transform the 'city_name' column in the batch data using the retrieved label encoder
    data_transformed['city_name_encoded'] = encoder.transform(data_transformed['city_name'])
    
    # Drop unnecessary columns from the batch data
    data_transformed = data_transformed.drop(columns=['unix_time', 'pm2_5', 'city_name', 'date'])

    return data_transformed


def get_future_data(date: str, city_name: str, feature_view, model, encoder) -> pd.DataFrame:
    """
    Predicts future PM2.5 data for a specified date and city using a given feature view and model.

    Args:
        date (str): The target future date in the format 'YYYY-MM-DD'.
        city_name (str): The name of the city for which the prediction is made.
        feature_view: The feature view used to retrieve batch data.
        model: The machine learning model used for prediction.
        encoder (LabelEncoder): Label encoder object to encode 'city_name'.

    Returns:
        pd.DataFrame: A DataFrame containing predicted PM2.5 values for each day starting from the target date.

    """
    # Get today's date
    today = datetime.date.today()

    # Convert the target date string to a datetime object
    date_in_future = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    # Calculate the difference in days between today and the target date
    difference_in_days = (date_in_future - today).days

    # Retrieve batch data for the specified date range
    batch_data = feature_view.get_batch_data(
        start_time=today,
        end_time=today + datetime.timedelta(days=1),
    )
    
    # Filter batch data for the specified city
    batch_data_filtered = batch_data[batch_data['city_name'] == city_name]
        
    # Transform batch data
    batch_data_transformed = transform_data(batch_data_filtered, encoder)
    
    # Initialize a DataFrame to store predicted PM2.5 values
    try:
        pm2_5_value = batch_data_filtered['pm2_5'].values[0]
    except (IndexError, TypeError):
        # If accessing pm2_5 values fails, return a message indicating the feature pipeline needs updating
        return "Data is not available. Ask user to run the feature pipeline to update data."
    else:
        # Initialize a DataFrame to store predicted PM2.5 values
        predicted_pm2_5_df = pd.DataFrame({
            'date': [today.strftime("%Y-%m-%d")],
            'pm2_5': pm2_5_value,
        })

    # Iterate through each day starting from tomorrow up to the target date
    for day_number in range(1, difference_in_days + 1):

        # Calculate the date for the current future day
        date_future_day = (today + datetime.timedelta(days=day_number)).strftime("%Y-%m-%d")
        
        # Predict PM2.5 for the current day
        predicted_pm2_5 = model.predict(batch_data_transformed)

        # Update previous day PM2.5 values in the batch data for the next prediction
        batch_data_transformed['pm_2_5_previous_7_day'] = batch_data_transformed['pm_2_5_previous_6_day']
        batch_data_transformed['pm_2_5_previous_6_day'] = batch_data_transformed['pm_2_5_previous_5_day']
        batch_data_transformed['pm_2_5_previous_5_day'] = batch_data_transformed['pm_2_5_previous_4_day']
        batch_data_transformed['pm_2_5_previous_4_day'] = batch_data_transformed['pm_2_5_previous_3_day']
        batch_data_transformed['pm_2_5_previous_3_day'] = batch_data_transformed['pm_2_5_previous_2_day']
        batch_data_transformed['pm_2_5_previous_2_day'] = batch_data_transformed['pm_2_5_previous_1_day']
        batch_data_transformed['pm_2_5_previous_1_day'] = predicted_pm2_5
        
        # Append the predicted PM2.5 value for the current day to the DataFrame
        predicted_pm2_5_df = predicted_pm2_5_df._append({
            'date': date_future_day, 
            'pm2_5': predicted_pm2_5[0],
        }, ignore_index=True)
        
    return predicted_pm2_5_df

def convert_unix_to_date(x):
    x //= 1000
    x = datetime.datetime.fromtimestamp(x)
    return datetime.datetime.strftime(x, "%Y-%m-%d")

def decode_features(df, feature_view, training_dataset_version=1):
    """Decodes features using corresponding transformation functions from passed Feature View object.
    !!! Columns names in passed DataFrame should be the same as column names in transformation fucntions mappers.
    """
    df_res = df.copy()

    feature_view.init_batch_scoring(training_dataset_version=1)
    td_transformation_functions = (
        feature_view._batch_scoring_server._transformation_functions
    )

    res = {}
    for feature_name in td_transformation_functions:
        if feature_name in df_res.columns:
            td_transformation_function = td_transformation_functions[feature_name]
            sig, foobar_locals = (
                inspect.signature(td_transformation_function.transformation_fn),
                locals(),
            )
            param_dict = dict(
                [
                    (param.name, param.default)
                    for param in sig.parameters.values()
                    if param.default != inspect._empty
                ]
            )
            if td_transformation_function.name == "min_max_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * (param_dict["max_value"] - param_dict["min_value"])
                    + param_dict["min_value"]
                )
            elif td_transformation_function.name == "standard_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * param_dict["std_dev"] + param_dict["mean"]
                )
            elif td_transformation_function.name == "label_encoder":
                dictionary = param_dict["value_to_index"]
                dictionary_ = {v: k for k, v in dictionary.items()}
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: dictionary_[x]
                )
    return df_res