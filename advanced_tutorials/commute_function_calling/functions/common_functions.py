import datetime


def convert_date_to_unix(x):
    """
    Convert datetime to unix time in milliseconds.
    """
    dt_obj = datetime.datetime.fromisoformat(str(x))
    dt_obj = int(dt_obj.timestamp() * 1000)
    return dt_obj
