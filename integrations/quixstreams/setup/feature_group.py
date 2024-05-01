import argparse

import hopsworks
from hsfs.feature import Feature


def connect(args):
    if args.host is None or args.project is None:
        print("Connecting to Hopsworks Serverless")
        project = hopsworks.login(api_key_value=args.api_key)
    else:
        print("Connecting to Hopsworks at " + str(args.host))
        project = hopsworks.login(
            host=args.host,
            port=args.port,
            project=args.project,
            api_key_value=args.api_key,
        )
    return project.get_feature_store()


def create_orderbook_feature_group(fs):
    features = [
        Feature(name="ticker", type="string"),
        Feature(name="timestamp", type="timestamp"),
        Feature(name="bid", type="float"),
        Feature(name="bid_size", type="float"),
        Feature(name="ask", type="float"),
        Feature(name="ask_size", type="float"),
        Feature(name="spread", type="float"),
    ]

    fg = fs.get_or_create_feature_group(
        name="orderbook",
        description="Coinbase (BTC-USD) orderbook records",
        version=1,
        primary_key=["ticker"],
        event_time="timestamp",
        statistics_config=False,
        online_enabled=False,
        features=features,
    )
    fg.save()

    return fg


def create_orderbook_agg_feature_group(fs, suffix):
    features = [
        Feature(name="ticker", type="string"),
        Feature(name="timestamp", type="timestamp"),
        Feature(name="bid_min", type="float"),
        Feature(name="bid_max", type="float"),
        Feature(name="bid_mean", type="float"),
        Feature(name="bid_sum", type="float"),
        Feature(name="bid_count", type="float"),
        Feature(name="bid_size_count", type="float"),
        Feature(name="ask_min", type="float"),
        Feature(name="ask_max", type="float"),
        Feature(name="ask_mean", type="float"),
        Feature(name="ask_sum", type="float"),
        Feature(name="ask_count", type="float"),
        Feature(name="ask_size_count", type="float"),
        Feature(name="spread", type="float"),
    ]

    fg = fs.get_or_create_feature_group(
        name="orderbook_agg_" + suffix,
        description="Coinbase (BTC-USD) orderbook aggregated records (" + suffix + ")",
        version=1,
        primary_key=["ticker"],
        event_time="timestamp",
        statistics_config=False,
        online_enabled=True,
        features=features,
    )
    fg.save()

    return fg


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
    args = parser.parse_args()

    # Setup connection to Hopsworks
    fs = connect(args)

    # Create feature groups
    fg = create_orderbook_feature_group(fs)
    fg_1m = create_orderbook_agg_feature_group(fs, "1m")
    fg_3m = create_orderbook_agg_feature_group(fs, "2m")
    fg_5m = create_orderbook_agg_feature_group(fs, "3m")

    print("\n Feature groups created successfully")
