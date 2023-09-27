import pandas as pd


### Feature functions



### Feature data frame functions

def process(df: pd.DataFrame) -> pd.DataFrame:
    
    # consider only customers with age defined.
    df.dropna(inplace=True, subset=["age"])
    
    # drop missing values 
    df.dropna(axis=1, inplace=True)
    
    return df


### Read / extract / consume data

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)