import os
import hopsworks
import pandas as pd

from features import articles, customers, transactions


### connect to Hopsworks
project = hopsworks.login()

fs = project.get_feature_store()
mr = project.get_model_registry()



### get feature groups

trans_fg = fs.get_feature_group("transactions", version=1)
customers_fg = fs.get_feature_group("customers", version=1)
articles_fg = fs.get_feature_group("articles", version=1)



### create feature views

customers_query = customers_fg.select_all()
customers_feature_view = fs.create_feature_view(
    name='customers',
    query=customers_query,
    
    # club_member_status is used during model inference, but not during model training
    # e.g., customers with a higher club_member_status are provided more or better recommendations (retrieve more candidates, add additional candidate filtering, ...)
    inference_helper_columns=["club_member_status"],
)


articles_query = articles_fg.select_all()
articles_feature_view = fs.create_feature_view(
    name='articles',
    query=articles_query
)

query = trans_fg.select(["customer_id", "article_id", "t_dat", "month_sin", "month_cos"])\
    .join(customers_fg.select(["age"]), on="customer_id")\
    .join(articles_fg.select(["garment_group_name", "index_group_name"]), on="article_id")

retrieval_fv = fs.create_feature_view(
    name='retrieval',
    query=query,
    
    # sales_channel_id is needed during model training, but won't be part of the model inputs
    # e.g., transactions via a specific sales channel are less relevant than the others
    training_helper_columns=["sales_channel_id"],
    
    # club_member_status is used during model inference, but not during model training
    # e.g., customers with a higher club_member_status are provided more or better recommendations (retrieve more candidates, add additional candidate filtering, ...)
    inference_helper_columns=["club_member_status"],
    
    # used when generating training data, and building model inputs in the inference pipeline
#     transformation_functions={
#         "month_sin": fs.get_transformation_function(name="month_sin", version=1),
#         "month_cos": fs.get_transformation_function(name="month_cos", version=1),
#     }
    
    
)


### create training datasets

td_version, td_job = retrieval_fv.create_train_validation_test_split(
    validation_size = 0.1, 
    test_size = 0.1,
    description = 'Retrieval dataset splits',
    data_format = 'csv',
    write_options = {'wait_for_job': True},
    coalesce = True,
    
    include_primary_keys=False,
    include_event_time=False,
)

train_df, val_df, test_df, y_train, y_val, y_test = feature_view.get_train_validation_test_split(training_dataset_version=td_version)

# -- alternatively, in-memory training data

train_df, val_df, test_df, y_train, y_val, y_test = retrieval_fv.train_validation_test_split(
    validation_size = 0.1, 
    test_size = 0.1,
    description = 'Retrieval dataset splits',
    
    include_primary_keys=False,
    include_event_time=False,
)
td_version = retrieval_fv.get_latest_training_data_version()  # we need the td_version when registering the models



### prepare training datasets

# 1. use sales_channel_id to improve the training data

def drop_transactions_via_3rd_party(df):
    df = df[df.sales_channel_id == 3]  # 3 is 3rd_party_platform


# 2. drop training_helper_columns (i.e., sales_channel_id)
train_df.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)
val_df.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)
test_df.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)
y_train.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)
y_val.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)
y_test.drop(columns=retrieval_fv.training_helper_columns, axis=1, inPlace=True)

# -- alternatively
train_df, val_df, test_df, y_train, y_val, y_test = retrieval_fv.drop_training_helper_columns(train_df, val_df, test_df, y_train, y_val, y_test)

# convert to tensorflow dataset
import tensorflow as tf
def df_to_ds(df):
    return tf.data.Dataset.from_tensor_slices({col : df[col] for col in df})

BATCH_SIZE = 2048
train_ds = df_to_ds(train_df).batch(BATCH_SIZE).cache().shuffle(BATCH_SIZE*10)
val_ds = df_to_ds(val_df).batch(BATCH_SIZE).cache()

# used by the query tower model
user_id_list = train_df["customer_id"].unique().tolist()
# used by the item tower model
item_id_list = train_df["article_id"].unique().tolist()
garment_group_list = train_df["garment_group_name"].unique().tolist()
index_group_list = train_df["index_group_name"].unique().tolist()
# TODO: prepare training dataset for ranking model ...


### train models

from models import query_tower, item_tower, two_tower, ranking_classifier

EMB_DIM = 16

query_features = ["customer_id", "age", "month_sin", "month_cos"]
candidate_features = ["article_id", "garment_group_name", "index_group_name"]

# query model
query_model = query_tower.QueryTower(EMB_DIM, user_id_list)
query_model.normalize_age(train_ds)

query_df = train_df[query_features]
query_ds = df_to_ds(query_df).batch(1)
query_model.warm_up(query_ds)  # init weights

# item model
item_model = item_tower.ItemTower(EMB_DIM, item_id_list, garment_group_list, index_group_list)

item_df = train_df[candidate_features]
item_df.drop_duplicates(subset="article_id", inplace=True)
item_ds = df_to_ds(item_df)

# two tower model
two_tower_model = TwoTowerModel(query_model, item_model, items_ds, BATCH_SIZE)
two_tower_model.set_optimizer(weight_decay=0.001, learning_rate=0.01)

two_tower_model.fit(train_ds, validation_data=val_ds, epochs=5)

# ranking model
cat_features = list(X_train.select_dtypes(include=['string', 'object']).columns)
pool_train = Pool(X_train, y_train, cat_features=cat_features)
pool_val = Pool(X_val, y_val, cat_features=cat_features)

ranking_model = ranking_classifier.RankingClassifier(
    learning_rate=0.2,
    iterations=100,
    depth=10,
    scale_pos_weight=10,
    early_stopping_rounds=5,
    use_best_model=True
)
ranking_model.fit(pool_train, eval_set=pool_val)

### save models

tf.saved_model.save(two_tower_model.query_model, export_dir="query_model")
tf.saved_model.save(two_tower_model.item_model, export_dir="candidate_model")
ranking_model.save(filename="ranking_model")


### register models in Hopsworks

# NOTE:
# Model Input Schema (features) = columns - ( primary_keys - event_time - partition_key - labels - training_helper_columns - inference_helpers_columns)
# Model Output Schema (labels)

# ranking model
input_example = X_train.sample().to_dict("records")

# NOTE: no need to build model schema manually, it will be obtained from the feature view
# input_schema = Schema(X_train)
# output_schema = Schema(y_train)
# model_schema = ModelSchema(input_schema, output_schema)

ranking_model = mr.python.create_model(
    name="ranking_model",
    description="Ranking model that scores item candidates",
    metrics=metrics,
    
    # pass feature view and td_version to the model. The following actions will be taken
    # - check ondemand functions in the feature groups contained in the feature view query
    # - if any, include them in the model artifact
    # - define Model Input Schema as: feature_view columns - (primary_keys - event_time - partition_key - labels - training_helper_columns - inference_helpers_columns)
    feature_view=retrieval_fv,
    training_dataset_version=td_version,
    
    input_example=input_example,
    #     model_schema=model_schema,  # not needed. it will obtained from the feature view
)

ranking_model.save("ranking_model.pkl")


from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# query model
query_model_input_schema = Schema(query_df)  # infer input schema from query_df
query_model_output_schema = Schema([{  # define output schema
    "name": "query_embedding",
    "type": "float32",
    "shape": [EMB_DIM]
}])
query_model_schema = ModelSchema(
    input_schema=query_model_input_schema,
    output_schema=query_model_output_schema
)
query_example = query_df.sample().to_dict("records")

mr_query_model = mr.tensorflow.create_model(
    name="query_model",
    description="Model that generates query embeddings from user and transaction features",
    
    feature_view=customers_feature_view,
    training_dataset_version=td_version,
    
    input_example=query_example,
    model_schema=query_model_schema,  # in this model, the schema differs from the feature view schema
)
mr_query_model.save("query_model")

# candidate model
candidate_model_input_schema = Schema(item_df)
candidate_model_output_schema = Schema([{
    "name": "candidate_embedding",
    "type": "float32",
    "shape": [EMB_DIM]}
])
candidate_model_schema = ModelSchema(
    input_schema=candidate_model_input_schema,
    output_schema=candidate_model_output_schema
)
candidate_example = item_df.sample().to_dict("records")

mr_candidate_model = mr.tensorflow.create_model(
    name="candidate_model",
    description="Model that generates candidate embeddings from item features",
    
    feature_view=retrieval_fv,
    training_dataset_version=td_version,
    
    input_example=candidate_example,
    model_schema=candidate_model_schema,  # in this model, the schema differs from the feature view schema
)
mr_candidate_model.save("candidate_model")
