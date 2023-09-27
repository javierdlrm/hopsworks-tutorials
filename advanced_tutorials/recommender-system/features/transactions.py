import pandas as pd
import numpy as np

### Feature functions

# @pandas_udf(StringType())
def article_id(article_ids: pd.Series) -> pd.Series:
    return article_ids.astype(str)

def t_dat(t_dat: pd.Series) -> pd.Series:
    return t_dat.apply(lambda x: pd.to_datetime(x))

@ondemand
def month_sin(t_dat: str) -> float:
    month = datetime.fromisoformat(t_dat).month - 1
    C = 2*np.pi/12
    return np.sin(month*C).item()
    
@ondemand
def month_cos(t_dat: str) -> float:
    month = datetime.fromisoformat(t_dat).month - 1
    C = 2*np.pi/12
    return np.cos(month*C).item()


### Feature data frame functions

def process(df: pd.DataFrame) -> pd.DataFrame:
    
    # process article ids
    df.article_id = article_id(df.article_id)
    
    # process t_dat
    df.t_dat = t_dat(df.t_dat)
    
    # append month features
    df["month_sin"] = df["t_dat"]
    df["month_cos"] = df["t_dat"]
    
    # convert python datetime object to unix epoch milliseconds
    trans_df.t_dat = trans_df.t_dat.values.astype(np.int64) // 10 ** 6
    
    return df


def merge_customer_ids(trans_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    return trans_df.merge(customer_subset_df["customer_id"])


### Read / extract / consume data

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


### Transformation functions

def create_transformation_fns(fs):
    fns = [fn.name for fn in fs.get_transformation_functions()]

    if "month_sin" not in fns:
        month_to_sin = fs.create_transformation_function(transactions.month_sin, output_type=float, version=1)
        month_to_sin.save()

    if "month_cos" not in fns:
        month_cos = fs.create_transformation_function(transactions.month_cos, output_type=float, version=1)
        month_cos.save()