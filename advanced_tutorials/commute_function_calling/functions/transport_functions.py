import pandas as pd
from typing import Any, Dict, List
import datetime
import pandas as pd
import time
import inspect

feature_view_batch = None

#
# Predict late departures at a given time
#

def predict_late_departure_at_scheduled_time(scheduled_time: str, feature_view, model_deployment) -> pd.DataFrame:
    """Take a scheduled time and predict whether the departures at that time will be late."""

    depatures = feature_view.get_batch_data(primary_keys=True)
    departure_key = depatures[depatures['departure_id'].str.startswith(f"9295-{scheduled_time}")]["departure_id"].iloc[-1]
    
    print(f"Scheduled time: {scheduled_time}")
    print("Departure key:", departure_key)
    
    response = model_deployment.predict(inputs=[departure_key])
    response = {"lateness_probability": 1.0 - response["predictions"][0], "scheduled_time": scheduled_time}
    
    return response

#
# Get historical data about departures within a time range
#


def get_departures_history_in_date_range(date_start: str, date_end: str, feature_view, model_deployment) -> pd.DataFrame:
    """Take start and end dates and get the departures information history scheduled in that date range."""
    
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
    
    # Rename columns with more descriptive names
    batch_data = batch_data[["scheduled", "expected", "deviations_count", "deviations_importance_max", "d6h_state_issue_count", "d6h_late_count"]]
    batch_data = batch_data.rename(columns={"deviations_importance_max": "deviations_severity", "d6h_state_issue_count": "issues_count", "d6h_late_count": "late_count"})
    
    return pd.concat([batch_data.head(3), batch_data.tail(3)]).sort_values('scheduled').reset_index(drop=True)
