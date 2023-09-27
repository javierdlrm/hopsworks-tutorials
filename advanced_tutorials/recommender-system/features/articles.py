import pandas as pd


### Feature functions

# @pandas_udf(StringType())
def article_id(article_ids: pd.Series) -> pd.Series:
    return article_ids.astype(str)


### Feature data frame function

def process(df: pd.DataFrame) -> pd.DataFrame:
    
    # process article ids
    df.article_id = article_id(df.article_id)
    
    # drop missing values 
    df.dropna(axis=1, inplace=True)
    
    return df


### Read / extract / consume data

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)