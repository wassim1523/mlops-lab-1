"""Register the model logged by a Lab 2 or Lab 5 run."""

import argparse
import os

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--initial-champion", action="store_true")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{args.run_id}'",
        output_format="list",
    )
    (model,) = [item for item in models if item.name == "model"]
    version = mlflow.register_model(f"models:/{model.model_id}", "food11")
    if args.initial_champion:
        if "champion" in client.get_registered_model("food11").aliases:
            raise ValueError(
                "A champion exists; use promote.py to compare the new version"
            )
        client.set_registered_model_alias("food11", "champion", version.version)
    print(f"Registered food11 v{version.version} from {args.run_id}")


if __name__ == "__main__":
    main()
