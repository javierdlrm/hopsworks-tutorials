import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, Normalization


class QueryTower(tf.keras.Model):

    def __init__(self, EMB_DIM, user_id_list):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            StringLookup(
                vocabulary=user_id_list,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                # We add an additional embedding to account for unknown tokens.
                len(user_id_list) + 1,
                EMB_DIM
            )
        ])

        self.normalized_age = Normalization(axis=None)

        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(EMB_DIM, activation="relu"),
            tf.keras.layers.Dense(EMB_DIM)
        ])
        
    
    def warm_up(self, ds):
        self(next(iter(ds)))
    
    def normalize_age(self, ds):
        self.normalized_age.adapt(ds.map(lambda x : x["age"]))

    def call(self, inputs):
        concatenated_inputs = tf.concat([
            self.user_embedding(inputs["customer_id"]),
            tf.reshape(self.normalized_age(inputs["age"]), (-1,1)),
            tf.reshape(inputs["month_sin"], (-1,1)),
            tf.reshape(inputs["month_cos"], (-1,1))
        ], axis=1)

        outputs = self.fnn(concatenated_inputs)

        return outputs    
