import os
import hopsworks
import pandas as pd

from features import articles, customers, transactions


# connect to Hopsworks
project = hopsworks.login()

fs = project.get_feature_store()
dataset_api = project.get_dataset_api()


# download datasets
data_dir = "data/"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

for file in ["articles.parquet", "customers.parquet", "transactions_train.parquet"]:
    dataset_api.download(f"Resources/{file}", local_path=data_dir, overwrite=True)
    
    
# read and prepare articles data

articles_df = articles.read_parquet(data_dir + "articles.parquet")
articles_df = articles.process(articles_df)

# read and prepare customers data

customers_df = customers.read_parquet(data_dir + "customers.parquet")
customers_df = customers.process(customers_df)

# read and prepare transactions data

trans_df = transactions.read_parquet(data_dir + "transactions_train.parquet")
trans_df = transactions.process(trans_df)

N_USERS = 25_000

customer_subset_df = customers_df.sample(N_USERS, random_state=27)
trans_df = transactions.merge_customer_ids(trans_df, customer_subset_df)


# create transformation functions

transactions.create_transformation_fns(fs)


# create feature groups and insert data


articles_fg = fs.create_feature_group(
    name="articles",
    description="Fashion items data including type of item, visual description and category",
    primary_key=["article_id"],
    online_enabled=True
)
articles_fg.insert(articles_df)

customers_fg = fs.create_feature_group(
    name="customers",
    description="Customers data including age and postal code",
    primary_key=["customer_id"],
    online_enabled=True
)
customers_fg.insert(customers_df)

trans_fg = fs.create_feature_group(
    name="transactions",
    version=1,
    description="Transactions data including customer, item, price, sales channel and transaction date",
    primary_key=["customer_id", "article_id"], 
    online_enabled=True,
    event_time="t_dat",
    
    # include ondemand features (month_sin and month_cos) in the feature view.
    # these features are detected via ondemand() annotation in the feature function
    include_ondemand_features=True, # discarded
    ondemand_features={"month_sin": transactions.month_sin, "month_cos": transactions.month_cos},
    infer_from_path="features/transactions.py",
)

trans_fg.update_feature_fns(......)

# when the transactions feature group is created, the feature functions annotated with @ondemand will be stored in Hopsworks
# these will be included in the artifacts of the models trained using feature views whose queries contain this feature group
# i.e.,  @ondemand feature function  --->  feature_group ---> feature_view.query ---> train_df, test_df ---> model artifact ---> deployment
#               \_________________________________________________________________________________________________/

trans_fg.insert(trans_df)
