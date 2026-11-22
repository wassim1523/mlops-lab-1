"""Copy a registered Lab 3 champion to a new Compose tracking server."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    source = mlflow.MlflowClient(tracking_uri=args.source)
    target = mlflow.MlflowClient(tracking_uri=args.target)
    existing = target.search_registered_models("name = 'food11'")
    if existing and "champion" in existing[0].aliases:
        raise SystemExit("Target already has a champion")
    version = source.get_model_version_by_alias("food11", "champion")
    run = source.get_run(version.run_id)
    with TemporaryDirectory() as temp:
        mlflow.set_tracking_uri(args.source)
        path = mlflow.artifacts.download_artifacts(
            artifact_uri="models:/food11@champion", dst_path=temp
        )
        if not (Path(path) / "MLmodel").is_file():
            raise ValueError("Source model is missing its MLmodel descriptor")
        mlflow.set_tracking_uri(args.target)
        mlflow.set_experiment("food11")
        with mlflow.start_run(tags={"source_run_id": run.info.run_id}) as copied:
            mlflow.log_params(run.data.params)
            mlflow.log_metrics(run.data.metrics)
            mlflow.log_artifacts(path, artifact_path="imported_model")
        saved = mlflow.register_model(
            f"runs:/{copied.info.run_id}/imported_model", "food11"
        )
        target.set_registered_model_alias("food11", "champion", saved.version)
        print(f"Imported v{version.version} as food11@champion v{saved.version}")


if __name__ == "__main__":
    main()
