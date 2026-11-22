# Lab 4 — MLflow, inference and frontend with Compose

## Bootstrapping the named volume

On the original PC, complete Lab 3 registration first. Stop the host MLflow server to free port 5000. From the Lab 2 project directory, the source SQLite database and its original artifact paths must remain available. The following commands start the new registry alone, copy the champion weights and metrics, then start the API and upload page.

```powershell
docker compose up -d --build --wait mlflow
uv run python -m src.food11.import_champion --source sqlite:///mlflow.db
docker compose up -d --build
docker compose ps
docker compose logs inference
```

Open the MLflow UI at http://127.0.0.1:5000 and Streamlit at http://127.0.0.1:8501. Upload tests/fixtures/bread.jpg. The default artifact proxy serves the weights to inference over HTTP, and the volume contains both SQLite and the artifacts.

## Questions

1. A process restart preserves the container's writable layer. Removing that container loses the database and artifacts in its layer. A fresh container starts with a fresh registry.
2. A named volume keeps data across container replacement without coupling the setup to a host path. A writable bind mount can provide the same persistence if paths and permissions are configured on that host.
3. Compose registers service names on the project network. Inference resolves mlflow through Docker DNS.
4. INFERENCE_URL allows the same frontend image to target a Compose service or an API at another address.
5. The frontend posts the uploaded file to inference:8000 on the internal network. Users reach only the UI and tracking server on published host ports.
6. Without a readiness check, model loading can fail during API startup. The Compose configuration waits for MLflow's HTTP health check. A new registry must still be populated using the commands above.
7. The resolved Compose configuration publishes 127.0.0.1:5000 for MLflow and 127.0.0.1:8501 for Streamlit. Inference has no host binding. Runtime output from docker compose ps still requires a Docker daemon.
8. Refreshing the browser still calls the running API's loaded version. Reassign the champion alias in MLflow and run `docker compose restart inference`.
9. The new process downloads the current champion during startup. The existing image contains everything needed to load it.
10. Down/up retains the named volume. The -v option deletes it and its registry. Test the destructive variant only with a disposable Compose project and database.
11. Three replicas require routing across healthy instances. Multi-host recovery also requires scheduling and a durable backend and artifact store outside the failed host. A single-host Compose project cannot recover after loss of that host.

The MLflow model-copy path was exercised between isolated registries. Compose configuration validation and runtime deployment are separate checks.

## Starting from the reproduced experiment snapshot

The accompanying food11-registry.tar.gz contains a portable database and artifacts for the three independently trained runs. To try the demo on a machine without the original MLflow directory, use a fresh Compose project and empty volume. Stop any other service occupying port 5000 or 8501. From the repo directory in PowerShell:

```powershell
mkdir food11-snapshot
tar -xzf "$HOME/Downloads/food11-registry.tar.gz" -C food11-snapshot
docker compose -p food11demo create --build mlflow
$registry = docker compose -p food11demo ps -aq mlflow
docker cp "food11-snapshot/." "${registry}:/mlflow-data/"
docker compose -p food11demo up -d --build
```

The snapshot already includes the champion alias. Open http://127.0.0.1:8501 and upload a food photo.
