import argparse
import math
import os


def gate(candidate, champion, n):
    if n < 1 or not all(0 <= p <= 1 for p in (candidate, champion)):
        raise ValueError("Expected positive test size and accuracies in [0, 1]")
    margin = 2 * math.sqrt(champion * (1 - champion) / n)
    return ("champion" if candidate >= champion + margin else "challenger"), margin


def promote(client, version, test_size=None):
    candidate = client.get_model_version("food11", version)
    champion = client.get_model_version_by_alias("food11", "champion")
    if str(champion.version) == str(candidate.version):
        print(f"Version {version} is already champion")
        return
    current = client.get_run(champion.run_id).data
    new = client.get_run(candidate.run_id).data
    if current.params.get("dataset") != new.params.get("dataset"):
        raise ValueError("Runs must use the same evaluation dataset")
    n = int(current.params.get("test_size", test_size or 0))
    if not n or (test_size is not None and n != test_size):
        raise ValueError(
            "Supply the champion's correct evaluation count using --test-size"
        )
    if int(new.params.get("test_size", n)) != n:
        raise ValueError("Evaluation split sizes differ")
    a, p = new.metrics["test_accuracy"], current.metrics["test_accuracy"]
    decision, margin = gate(a, p, n)
    for key, value in {
        "candidate_accuracy": a,
        "champion_accuracy": p,
        "test_size": n,
        "two_se": margin,
        "baseline_version": champion.version,
        "decision": decision,
    }.items():
        client.set_model_version_tag("food11", version, key, str(value))
    client.set_registered_model_alias("food11", decision, version)
    print(
        f"v{version}: {decision}; candidate={a:.4%}, champion={p:.4%}, "
        f"margin={100 * margin:.3f} points, n={n}"
    )


if __name__ == "__main__":
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--test-size", type=int)
    args = parser.parse_args()
    promote(
        MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        ),
        args.version,
        args.test_size,
    )
