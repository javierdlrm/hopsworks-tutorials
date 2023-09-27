import os
import numpy as np
from datetime import datetime

import hopsworks

import logging


class Transformer(object):
    
    def __init__(self):            
        # connect to Hopsworks
        project = hopsworks.connection().get_project()
        mr = project.get_model_registry()
        
        query_model = mr.get_model(os.env["MODEL_NAME"], os.env["MODEL_VERSION"])
        
        self.customer_fv = query_model.get_feature_view(init=True)  # default init=True
        
        self.ondemand_feature_fns = query_model.get_ondemand_feature_functions
        
        
        # get feature views and transformation functions
#         fs = project.get_feature_store()
#         self.customer_fv = fs.get_feature_view("customers", 1)
#         self.month_to_sin = fs.get_transformation_function("month_sin").transformation_fn
#         self.month_to_cos = fs.get_transformation_function("month_cos").transformation_fn
        
        # get ranking deployment metadata object
        ms = project.get_model_serving()
        self.ranking_server = ms.get_deployment("rankingdeployment")
        
        
    def preprocess(self, inputs):
        inputs = inputs["instances"] if "instances" in inputs else inputs
        
        # extract month
        month_of_purchase = datetime.fromisoformat(inputs.pop("month_of_purchase"))
        
        # get customer features
        customer_features = self.customer_fv.get_feature_vector(inputs)
        
        # enrich inputs
        inputs["age"] = customer_features[1]
        inputs["month_sin"] = self.month_to_sin(month_of_purchase)
        inputs["month_cos"] = self.month_to_cos(month_of_purchase)
                
        return {"instances" : [inputs]}
    
    def postprocess(self, outputs):
        # get ordered ranking predictions
        return {"predictions": self.ranking_server.predict({ "instances": outputs["predictions"] })}
    