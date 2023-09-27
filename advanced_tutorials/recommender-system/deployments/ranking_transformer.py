import os
import pandas as pd

import hopsworks
from opensearchpy import OpenSearch

import logging


class Transformer(object):
    
    def __init__(self):
        # connect to Hopsworks
        project = hopsworks.connection().get_project()
        
        # get feature views
        self.fs = project.get_feature_store()
        self.articles_fv = self.fs.get_feature_view("articles", 1)
        self.articles_features = [feat.name for feat in self.articles_fv.schema]
        self.customer_fv = self.fs.get_feature_view("customers", 1)

        # create opensearch client
        opensearch_api = project.get_opensearch_api()
        self.os_client = OpenSearch(**opensearch_api.get_default_py_config())
        self.candidate_index = opensearch_api.get_project_index("candidate_index")

        # get ranking model feature names
        mr = project.get_model_registry()
        model = mr.get_model(os.environ["MODEL_NAME"], os.environ["MODEL_VERSION"])
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
    
    def preprocess(self, inputs):
        inputs = inputs["instances"][0]
        customer_id = inputs["customer_id"]
        
        # search for candidates
        hits = self.search_candidates(inputs["query_emb"], k=100)
        
        # get already bought items
        already_bought_items_ids = self.fs.sql(
            f"SELECT article_id from transactions_1 WHERE customer_id = '{customer_id}'"
        ).values.reshape(-1).tolist()
        
        # build dataframes
        item_id_list = []
        item_emb_list = []
        exclude_set = set(already_bought_items_ids)
        for el in hits:
            item_id = str(el["_id"])
            if item_id in exclude_set:
                continue
            item_emb = el["_source"]["my_vector1"]
            item_id_list.append(item_id)
            item_emb_list.append(item_emb)
        item_id_df = pd.DataFrame({"article_id" : item_id_list})
        item_emb_df = pd.DataFrame(item_emb_list).add_prefix("item_emb_")
        
        # get articles feature vectors
        articles_data = []
        for article_id in item_id_list:
            try:
                article_features = self.articles_fv.get_feature_vector({"article_id" : article_id})
                articles_data.append(article_features)
            except:
                logging.info("-- not found:" + str(article_id))
                pass # article might have been removed from catalogue
        articles_df = pd.DataFrame(data=articles_data, columns=self.articles_features)
        
        # join candidates with item features
        ranking_model_inputs = item_id_df.merge(articles_df, on="article_id", how="inner")
        
        # add customer features
        customer_features = self.customer_fv.get_feature_vector({"customer_id": customer_id})
        ranking_model_inputs["age"] = customer_features[1]
        ranking_model_inputs["month_sin"] = inputs["month_sin"]
        ranking_model_inputs["month_cos"] = inputs["month_cos"]
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]
        
        return { "inputs" : [{"ranking_features": ranking_model_inputs.values.tolist(), "article_ids": item_id_list} ]}

    def postprocess(self, outputs):
        preds = outputs["predictions"]
        ranking = list(zip(preds["scores"], preds["article_ids"])) # merge lists
        ranking.sort(reverse=True) # sort by score (descending)
        return { "ranking": ranking }
    
    def search_candidates(self, query_emb, k=100):
        k = 100
        query = {
          "size": k,
          "query": {
            "knn": {
              "my_vector1": {
                "vector": query_emb,
                "k": k
              }
            }
          }
        }
        return self.os_client.search(body = query, index = self.candidate_index)["hits"]["hits"]