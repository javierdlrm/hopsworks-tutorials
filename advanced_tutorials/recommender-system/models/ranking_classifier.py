from catboost import CatBoostClassifier
import joblib


class RankingClassifier(CatBoostClassifier):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def save(self, filename):
        joblib.dump(self, filename + ".pkl")
