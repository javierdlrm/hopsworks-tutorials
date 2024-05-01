import argparse
import json

from confluent_kafka import Producer

import hopsworks
import orderbook


SCHEMA_NAME = "orderbook_schema"
SCHEMA = {
    "type": "record",
    "name": SCHEMA_NAME,
    "namespace": "coinbase.orderbook",
    "fields": [
        {"name": "type", "type": ["null", "string"]},
        {"name": "product_id", "type": ["null", "string"]},
        {
            "name": "changes",
            "type": {"type": "array", "items": {"type": "array", "items": "string"}},
        },
        {
            "name": "time",
            "type": ["null", "string"],
        },
    ],
}


def connect(args):
    if args.host is None or args.project is None:
        print("Connecting to Hopsworks Serverless")
        project = hopsworks.login(api_key_value=args.api_key)
    else:
        print(f"Connecting to Hopsworks at {str(args.host)}")
        project = hopsworks.login(
            host=args.host,
            port=args.port,
            project=args.project,
            api_key_value=args.api_key,
        )
    return project.get_kafka_api()


def setup_kafka_producer(kafka_api, topic_name) -> Producer:
    if not any([t.name == topic_name for t in kafka_api.get_topics()]):
        # Create topic schema
        print(f"Creating schema '{SCHEMA_NAME}'...")
        kafka_api.create_schema(SCHEMA_NAME, SCHEMA)

        # Create kafka topic
        print(f"Creating topic '{topic_name}'...")
        _ = kafka_api.create_topic(topic_name, SCHEMA_NAME, 1, replicas=1, partitions=1)

    producer_config = kafka_api.get_default_config()
    producer_config["bootstrap.servers"] = "127.0.0.1:9092"
    return Producer(producer_config)


def delivery_callback(err, msg):
    if err:
        print("Message failed delivery: {}".format(err))
    else:
        print(
            "Message delivered to topic: {}, timestamp: {}".format(
                msg.topic(), msg.timestamp()
            )
        )


def replay_coinbase_orderbook(
    product_ids: list, batch_size: int, producer: Producer, topic: str
):
    coinbase_feed = orderbook.CoinbaseFeed(product_ids)
    btc_usd_source = coinbase_feed.get_source_for_product_id(product_ids[0])

    for _ in range(batch_size):
        batch = btc_usd_source.next_batch()
        event = json.loads(batch[0])

        if event["type"] == "snapshot":
            continue  # skip snapshot

        print("Producing...")
        producer.produce(
            topic,
            json.dumps(event),
            "key",
            callback=delivery_callback,
        )

    # trigger the sending of all messages to the brokers, 20sec timeout
    print("Flushing...")
    producer.flush(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hopsworks cluster configuration
    parser.add_argument("--host", help="Hopsworks cluster host", default=None)
    parser.add_argument(
        "--port", help="Port on which Hopsworks is listening on", default=443
    )
    parser.add_argument("--api_key", help="API key to authenticate with Hopsworks")
    parser.add_argument(
        "--project", help="Name of the Hopsworks project to connect to", default=None
    )
    parser.add_argument(
        "--topic",
        help="Name of the topic to replay orderbook events to",
        default="coinbase_orderbook",
    )
    parser.add_argument(
        "--num_events",
        help="Number of events (buy/sells) to replay",
        default=10,
        type=int,
    )

    args = parser.parse_args()

    # Setup connection to Hopsworks
    kafka_api = connect(args)

    # Setup producer and kafka topic
    producer = setup_kafka_producer(kafka_api, args.topic)

    # Replay CoinBase orderbook events
    replay_coinbase_orderbook(
        ["BTC-USD"],  # , "ETH-USD", "BTC-EUR", "ETH-EUR"],
        args.num_events,
        producer,
        args.topic,
    )
