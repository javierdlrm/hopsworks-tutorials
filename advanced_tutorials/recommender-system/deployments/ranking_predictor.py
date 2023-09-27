import os
import joblib
import numpy as np


class Predict(object):
    
    def __init__(self):
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/ranking_model.pkl")

    def predict(self, inputs):
        features = inputs[0].pop("ranking_features")
        article_ids = inputs[0].pop("article_ids")

        scores = self.model.predict_proba(features).tolist()
        scores = np.asarray(scores)[:,1].tolist() # get scores of positive class

        return { "scores": scores, "article_ids": article_ids }