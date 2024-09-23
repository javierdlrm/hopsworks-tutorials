import os
import datetime
import joblib

class Predictor(object):
    
    
    def __init__(self, project):        
        # get feature store handle
        self.fs = project.get_feature_store()
        
        # get feature views
        self.fv = self.fs.get_feature_view("departures_agg", 1)
        
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/xgboost_regressor.pkl")
        
        
    def predict(self, inputs):
        
        print("Inputs: ", inputs)
        
        feature_vector = self.fv.get_feature_vector({"departure_id": inputs[0][0]})
        print("Feature vector", feature_vector)
        
        feature_vector[0] = self.convert_date_to_unix(feature_vector[0])
        feature_vector[1] = self.convert_date_to_unix(feature_vector[1])
        
        predictions = self.model.predict([feature_vector])
        
        return predictions.tolist()
    
            
    def convert_date_to_unix(self, x):
        """
        Convert datetime to unix time in milliseconds.
        """
        dt_obj = datetime.datetime.fromisoformat(str(x))
        dt_obj = int(dt_obj.timestamp() * 1000)
        return dt_obj