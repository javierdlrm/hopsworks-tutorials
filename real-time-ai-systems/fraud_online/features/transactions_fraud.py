from math import radians
import numpy as np
import pandas as pd
from typing import Union

def haversine(long: pd.Series, lat: pd.Series, shift: int) -> np.ndarray:
    """
    Compute Haversine distance between each consecutive coordinate in (long, lat).

    Parameters:
    - long: pandas Series, longitude values
    - lat: pandas Series, latitude values
    - shift: int, the number of positions to shift for calculating distances

    Returns:
    - numpy array, Haversine distances
    """
    long_shifted = long.shift(shift)
    lat_shifted = lat.shift(shift)
    long_diff = long_shifted - long
    lat_diff = lat_shifted - lat

    a = np.sin(lat_diff/2.0)**2
    b = np.cos(lat) * np.cos(lat_shifted) * np.sin(long_diff/2.0)**2
    c = 2*np.arcsin(np.sqrt(a + b))

    return c


def time_delta(datetime_value: pd.Series, shift: int) -> pd.Series:
    """
    Compute time difference between each consecutive transaction.

    Parameters:
    - datetime_value: pandas Series, datetime values
    - shift: int, the number of positions to shift for calculating time differences

    Returns:
    - pandas Series, time differences
    """
    time_shifted = datetime_value.shift(shift)
    return time_shifted


def calculate_loc_delta_t_plus_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate loc_delta_t_plus_1 for each group.

    Parameters:
    - group: pandas DataFrame group, grouped by 'cc_num'

    Returns:
    - pandas Series, loc_delta_t_plus_1 values
    """
    df["loc_delta_t_plus_1"] = df.groupby("cc_num").apply(
        lambda x: haversine(x["longitude"], x["latitude"], 1)
        ).reset_index(level=0, drop=True).fillna(0)
    return df


def calculate_loc_delta_t_minus_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate loc_delta_t_minus_1 for each group.

    Parameters:
    - group: pandas DataFrame group, grouped by 'cc_num'

    Returns:
    - pandas Series, loc_delta_t_minus_1 values
    """
    df["loc_delta_t_minus_1"] = df.groupby("cc_num").apply(
        lambda x: haversine(x["longitude"], x["latitude"], -1)
        ).reset_index(level=0, drop=True).fillna(0)
    return df


def calculate_time_delta_t_minus_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time_delta_t_minus_1 for each group.

    Parameters:
    - group: pandas DataFrame group, grouped by 'cc_num'

    Returns:
    - pandas Series, time_delta_t_minus_1 values
    """
    df["time_delta_t_minus_1"] = df.groupby("cc_num").apply(lambda x: time_delta(x["datetime"], -1))\
        .reset_index(level=0, drop=True)
    return df


def prepare_transactions_fraud(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare transaction data with engineered features for fraud detection.

    Parameters:
    - trans_df: pandas DataFrame, transaction data

    Returns:
    - pandas DataFrame, prepared transaction data with engineered features
    """
    # Sort values and convert latitude and longitude to radians
    trans_df.sort_values("datetime", inplace=True)
    trans_df[["longitude", "latitude"]] = trans_df[["longitude", "latitude"]].applymap(radians)

    # Calculate loc_delta_t_plus_1, loc_delta_t_minus_1, and time_delta_t_minus_1 using groupby
    trans_df = calculate_loc_delta_t_plus_1(trans_df)

    trans_df = calculate_loc_delta_t_minus_1(trans_df)

    trans_df = calculate_time_delta_t_minus_1(trans_df)

    # Normalize time_delta_t_minus_1 to days and handle missing values
    trans_df["time_delta_t_minus_1"] = (trans_df["time_delta_t_minus_1"] - trans_df["datetime"]) / np.timedelta64(1, 'D')
    trans_df["time_delta_t_minus_1"] = trans_df["time_delta_t_minus_1"].fillna(0)

    # Select relevant columns, drop duplicates, and reset index
    trans_df = trans_df[["tid", "datetime", "cc_num", "amount", "country", "fraud_label",
                         "loc_delta_t_plus_1", "loc_delta_t_minus_1", "time_delta_t_minus_1"]]
    trans_df = trans_df.drop_duplicates(subset=['cc_num', 'datetime']).reset_index(drop=True)

    return trans_df


def _amount_band(amount: float) -> str:
    """Bucket an amount into a coarse low/medium/high band for text descriptors."""
    if amount < 50:
        return "low"
    if amount < 200:
        return "medium"
    return "high"


def _daypart(hour: int) -> str:
    """Map an hour-of-day to a coarse part-of-day label for text descriptors."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def build_behavior_descriptor(group: pd.DataFrame) -> str:
    """
    Build a natural-language descriptor of a payer's long-term behavior.

    Parameters:
    - group: pandas DataFrame, all transactions of a single cc_num

    Returns:
    - str, a text summary suitable for sentence embedding
    """
    amounts = group["amount"]
    top_country = group["country"].mode()
    top_country = top_country.iloc[0] if not top_country.empty else "US"
    foreign_ratio = float((group["country"] != top_country).mean())
    night_ratio = float(group["datetime"].dt.hour.between(0, 5).mean())
    span_days = max((group["datetime"].max() - group["datetime"].min()).total_seconds() / 86400.0, 1.0)
    freq = len(group) / span_days
    return (
        f"This payer makes {_amount_band(amounts.mean())}-value cash withdrawals "
        f"averaging ${amounts.mean():.0f} (max ${amounts.max():.0f}), mostly in {top_country}. "
        f"{foreign_ratio * 100:.0f}% of transactions are foreign and "
        f"{night_ratio * 100:.0f}% happen at night, "
        f"with about {freq:.1f} transactions per day."
    )


def build_sequence_descriptor(group: pd.DataFrame, n: int = 5) -> str:
    """
    Build a natural-language descriptor of a payer's most recent transactions.

    Parameters:
    - group: pandas DataFrame, all transactions of a single cc_num
    - n: int, number of most recent transactions to include

    Returns:
    - str, a text summary suitable for sentence embedding
    """
    last = group.sort_values("datetime").tail(n)
    lines = []
    prev_dt = None
    for _, row in last.iterrows():
        gap = 0.0 if prev_dt is None else (row["datetime"] - prev_dt).total_seconds() / 86400.0
        lines.append(
            f"${row['amount']:.0f}, {row['country']}, "
            f"{_daypart(row['datetime'].hour)}, {gap:.1f}d since previous"
        )
        prev_dt = row["datetime"]
    return "Recent transactions: " + "; ".join(lines)


def compute_payer_features(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into one row per payer (cc_num) with text descriptors.

    Produces the source text for the payer behavior and recent-sequence embeddings.

    Parameters:
    - trans_df: pandas DataFrame, prepared transaction data

    Returns:
    - pandas DataFrame with columns cc_num, behavior_text, sequence_text
    """
    rows = []
    for cc_num, group in trans_df.groupby("cc_num"):
        rows.append({
            "cc_num": cc_num,
            "behavior_text": build_behavior_descriptor(group),
            "sequence_text": build_sequence_descriptor(group),
        })
    return pd.DataFrame(rows)