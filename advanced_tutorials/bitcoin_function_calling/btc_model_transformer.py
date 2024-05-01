
import os
import hsfs
import numpy as np

class Transformer(object):
    
    def __init__(self):        
        # get feature store handle
        fs_conn = hsfs.connection()
        self.fs = fs_conn.get_feature_store()
        
        # get feature views
        self.fv = self.fs.get_feature_view("bitcoin_feature_view", 1)
        
        # initialise serving
        self.fv.init_serving(1)

    def flat2gen(self, alist):
        for item in alist:
            if isinstance(item, list):
                for subitem in item: yield subitem
            else:
                yield item
        
    def preprocess(self, inputs):
        feature_vector = self.fv.get_feature_vector({"unix": inputs["instances"][0][0]})
        feature_vector = [*feature_vector[:9], *feature_vector[10:]]
        return { "instances" :  np.array(list(self.flat2gen(feature_vector))).reshape(-1, 1, len(feature_vector)).tolist() }

    def postprocess(self, outputs):
        return outputs
