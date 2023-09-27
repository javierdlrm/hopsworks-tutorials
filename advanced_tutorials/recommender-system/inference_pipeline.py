import os
import hopsworks

RANKING_DEPLOYMENT_NAME = "rankingdeployment"
QUERY_DEPLOYMENT_NAME = "querydeployment"

connected = False


def connect():
    if not connected:
        project = hopsworks.login()
        mr = project.get_model_registry()
        ms = project.get_model_serving()
        connected = True

    
def setup():
    connect()
    
    # get models
    ranking_model = mr.get_best_model("ranking_model", "fscore", "max")
    query_model = mr.get_model("query_model")

    # upload deployment scripts to Hopsworks file system
    uploaded_file_path = dataset_api.upload("deployments/ranking_transformer.py", "Resources", overwrite=True)
    ranking_transformer_script_path = os.path.join("/Projects", project.name, uploaded_file_path)
    uploaded_file_path = dataset_api.upload("deployments/ranking_predictor.py", "Resources", overwrite=True)
    ranking_predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)
    uploaded_file_path = dataset_api.upload("querymodel_transformer.py", "Models", overwrite=True)
    transformer_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

    # create deployments
    
    from hsml.transformer import Transformer

# Deployment schema
#     ranking_deployment : {
#       primary_keys: { customer_id : str},
#       query_emb: list(float),
#       month_sin : { t_dat: datetime }, # ondemand feature
#       month_sin : { t_dat: datetime }, # ondemand feature
#       output: { query_embedding: list(float32) },
#     }
# Inference schema
#     ranking_deployment : {
#       inputs: { customer_id : str, month_sin: float, month_cos: float, query_emb: list(float) },
#       output: { scores: list(float), article_ids: list(float) },
#     }
    ranking_transformer=Transformer(script_file=ranking_transformer_script_path, resources={"num_instances": 0})
    ranking_deployment = ranking_model.deploy(name=RANKING_DEPLOYMENT_NAME,
                                              description="Deployment that search for item candidates and scores them based on customer metadata",
                                              script_file=ranking_predictor_script_path,
                                              resources={"num_instances": 0},
                                              transformer=ranking_transformer)

# Deployment schema:
#     query_deployment : {
#       primary_keys: { customer_id: str },
#       age: int,
#       month_sin : { type: ondemand, t_dat: str }, # ondemand feature
#       month_sin : { type: ondemand, t_dat: str }, # ondemand feature
#       output: { query_embedding: list(float32) },
#     }
# Inference schema:
#     query_deployment : {
#       inputs: { customer_id : str, t_dat: str },
#       output: { query_embedding: list(float32) },
#     }
    query_model_transformer=Transformer(script_file=transformer_script_path, resources={"num_instances": 0})
    query_model_deployment = query_model.deploy(name=QUERY_DEPLOYMENT_NAME,
                                                description="Deployment that generates query embeddings from customer and item features using the query model",
                                                resources={"num_instances": 0},
                                                transformer=query_model_transformer)

    
def check_deployments():
    connect()
    return ms.get_deployment(RANKING_DEPLOYMENT_NAME) is not None and
           ms.get_deployment(QUERY_DEPLOYMENT_NAME) is not None
    
    
def start_deployments():
    connect()
    # start deployments
    ms.get_deployment(RANKING_DEPLOYMENT_NAME).start()
    ms.get_deployment(QUERY_DEPLOYMENT_NAME).start()


def run():
    connect()
    # send requests to the recommender system
    pass



if __name__ == "__main__":
    # get command args
    args = None # use args parse
    
    if args.operation == "setup":
        setup()
    
    if args.operation == "start":
        start()

    if args.operation == "run":
        run()
