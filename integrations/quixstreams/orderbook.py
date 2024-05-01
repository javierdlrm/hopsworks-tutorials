import json

from websocket import create_connection


class CoinbaseSource:
    def __init__(self, product_id):
        self.product_id = product_id
        self.ws = create_connection("wss://ws-feed.exchange.coinbase.com")
        self.ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channels": ["level2_batch"],
                }
            )
        )
        # The first msg is just a confirmation that we have subscribed.
        print(self.ws.recv())

    def next_batch(self):
        return [self.ws.recv()]

    def snapshot(self):
        return None

    def close(self):
        self.ws.close()


class CoinbaseFeed:
    def __init__(self, product_ids):
        self.product_ids = product_ids

    def list_product_ids(self) -> list:
        return self.product_ids

    def get_source_for_product_id(self, product_id) -> CoinbaseSource:
        return CoinbaseSource(product_id)
