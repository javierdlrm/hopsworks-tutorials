import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


class ItemTower(tf.keras.Model):

    def __init__(self, EMB_DIM, item_id_list, garment_group_list, index_group_list):
        super().__init__()
        
        self.item_id_list = item_id_list
        self.garment_group_list = garment_group_list
        self.index_group_list = index_group_list

        self.item_embedding = tf.keras.Sequential([
            StringLookup(
                vocabulary=item_id_list,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                # We add an additional embedding to account for unknown tokens.
                len(item_id_list) + 1,
                EMB_DIM
            )
        ])

        self.garment_group_tokenizer = StringLookup(vocabulary=garment_group_list, mask_token=None)
        self.index_group_tokenizer = StringLookup(vocabulary=index_group_list, mask_token=None)

        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(EMB_DIM, activation="relu"),
            tf.keras.layers.Dense(EMB_DIM)
        ])

    def call(self, inputs):
        garment_group_embedding = tf.one_hot(
            self.garment_group_tokenizer(inputs["garment_group_name"]),
            len(self.garment_group_list)
        )

        index_group_embedding = tf.one_hot(
            self.index_group_tokenizer(inputs["index_group_name"]),
            len(self.index_group_list)
        )

        concatenated_inputs = tf.concat([
            self.item_embedding(inputs["article_id"]),
            garment_group_embedding,
            index_group_embedding
        ], axis=1)

        outputs = self.fnn(concatenated_inputs)

        return outputs