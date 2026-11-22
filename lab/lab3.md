# Lab 3 — Containerizing Food-11

The API uses the same RGB size, ImageNet normalization and class order as the Lab 2 ResNet18. It converts the 11 output logits to probabilities with softmax.

## Reproduce on the Lab 2 PC

Start the tracking server from the original project directory, using the same database and artifact location as Lab 2. In another PowerShell window:

```powershell
uv sync --locked
uv run python -m src.food11.register 84ab21a8d12c44f99a06359b23bb277e --initial-champion
uv run uvicorn src.food11.serve:app --port 8000
```

In a third terminal:

```powershell
curl.exe -F "file=@tests/fixtures/bread.jpg" http://127.0.0.1:8000/predict
```

For container inference the model artifacts must also be accessible to the container. The original Lab 2 artifact paths can refer to the Windows filesystem. Lab 4 includes an importer that copies the original model into a server with proxied artifacts. After importing, use that server for this standalone test. Stop the local API first to release port 8000. On Docker Desktop, the server must listen on a host interface accessible to the container and accept its host header.

```powershell
docker build -t food11-api:latest .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

## Questions

1. A logged model belongs to a particular training run. Registration gives the model a shared name and numbered versions across runs. The independently reproduced baseline received version 1 in a fresh registry. The version number in the original PC registry depends on its history.
2. Aliases are user-defined names such as `champion` and `challenger`. A version preserves lineage. Changing the alias changes the deployment choice independently of the training run.
3. The URI resolves the chosen version and its MLflow model format. Move the alias to a new version and restart the API to pick it up.
4. Docker reuses the dependency layer when the lock and project metadata are unchanged. Editing the API invalidates the source-copy layer and subsequent layers.
5. The final stage contains the environment and source. The builder also holds uv and its download cache. To measure on a Docker host, build a single-stage comparison by using the builder instructions followed by the runtime source copy, PATH, EXPOSE and CMD. Run `docker image ls` and `docker history` for both image tags. Image sizes were not measured here.
6. The ignore file removes local datasets, tracking files, the Git history and virtual environment from the build context. Sending these paths alone does not normally break a build. Copying a host virtual environment into a Linux image can overwrite incompatible binaries. Narrow COPY instructions also keep unreferenced files out of the final image.
7. Container loopback belongs to the container. On Docker Desktop, host.docker.internal identifies the host gateway. Compose services use the network and DNS names described in Lab 4.
8. A new container resolves the alias again at startup. The server, artifacts and alias must be available. The image includes code and dependencies.
9. Publish the tested image to a registry with a commit-specific tag. Record the registry digest when exact content identity is required. Lab 5 implements publication to GHCR.

## Local endpoint check

With the reproduced ResNet18 registered as champion, Uvicorn returned HTTP 200 from both endpoints. Uploading the 128×128 JPEG tests/fixtures/bread.jpg returned category Bread with confidence 0.9962521. The image comes from the training split and serves as a request-format check.
