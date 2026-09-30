# Lab 2 — Model training and experiment tracking with MLflow

## Objective

The objective of this lab was to train a Food-11 image classification
model and track experiments using MLflow. I recorded training parameters,
performance metrics and trained models, then compared four configurations.

## 1. Environment setup

I continued working in the Lab 1 project and installed the required libraries:

```powershell
python -m uv add mlflow torch torchvision scikit-learn
```

To use the CPU-only versions of PyTorch and torchvision, I configured
the following in `pyproject.toml` before installation:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

### Question 1 — Changes to pyproject.toml and uv.lock

`pyproject.toml` contains the new project dependencies: MLflow, torch,
torchvision and scikit-learn. It also contains the CPU index configuration.

`uv.lock` records the exact resolved dependency versions and their sources,
allowing the environment to be reproduced.

## 2. MLflow server

I started a local MLflow server with:

```powershell
python -m uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

The interface was available at `http://127.0.0.1:5000`.
I kept the server running in one terminal and used a second terminal
to execute the training commands.

### Question 2 — Backend store and artifact root

The backend store, `sqlite:///mlflow.db`, stores metadata such as
experiments, runs, parameters and metrics in a SQLite database.

The artifact root, `./mlruns`, specifies the storage location for
artifacts such as models and supporting files.

### Question 3 — Why ignore MLflow outputs?

Git tracks the source code, dependency configuration and report.
DVC tracks the dataset from Lab 1, while MLflow manages the experiment
results and saved models.

I used these Git ignore rules:

```gitignore
mlflow.db
mlflow.db-*
mlruns/
mlartifacts/
```

These generated files are not committed to Git or added to DVC in this
lab. Keeping them out of Git avoids storing a changing database and large
model binaries in the repository.

The MLflow results remain local and are not backed up by a Git push.

### Question 4 — Experiment creation

The training script uses:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
```

If the `food11` experiment does not exist, MLflow creates it.
Otherwise, it reuses the existing experiment. Each execution creates
a new run inside this experiment.

## 3. Dataset and training script

The script is located at `src/food11/train.py`.

I used `data/food11_processed_mini`, prepared in Lab 1.
The images belong to 11 food categories and are divided into:

- `training`: updates the model parameters.
- `validation`: measures performance after each epoch.
- `evaluation`: measures final test performance.

The script loads images using `ImageFolder` and creates batches using
`DataLoader`. Images are resized to 128 × 128 pixels, converted to tensors
and normalized using the ImageNet mean and standard deviation.

The model is a pretrained ResNet18. Its final classification layer is
replaced with a linear layer containing 11 outputs. All model parameters
are fine-tuned using Adam and cross-entropy loss.

Training runs on CPU, with a random seed of 42.

The script supports the following arguments:

- `--dataset`: selects the mini or full processed dataset.
- `--epochs`: specifies the number of training epochs.
- `--lr`: specifies the learning rate.
- `--batch-size`: specifies the number of images per batch.

### Question 5 — Parameters, metrics and step

Parameters describe the run configuration. Examples include learning
rate, batch size, dataset and number of epochs.

Metrics describe measured performance. The script records:

- `train_loss`: average training loss per epoch.
- `val_loss`: average validation loss per epoch.
- `val_accuracy`: proportion of correctly classified validation images.
- `test_accuracy`: proportion of correctly classified test images after training.

The `step` argument identifies the epoch of each logged measurement.
This allows MLflow to display metric evolution across epochs.

### Question 6 — Metric charts and saved model

MLflow provides charts for the logged metrics. Training and validation
loss curves help assess learning progress and possible overfitting.
Validation accuracy shows how classification performance changes
during training.

The script saves the final model using:

```python
mlflow.pytorch.log_model(
    model,
    name="model",
    input_example=example,
)
```

The model files are stored in the configured artifact storage.
Their exact location is identified by the model's artifact URI.
MLflow links the model to its originating run.

The script also records `class_to_idx.json`, which maps category names
to numerical class indices.

The saved model corresponds to the final epoch, rather than an
automatically selected best-epoch checkpoint.

## 4. Experiments and results

I ran four experiments, each with five epochs on the mini dataset.
Each execution started from the pretrained ResNet18 weights.

```powershell
python -m uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

```powershell
python -m uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32
```

```powershell
python -m uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
```

```powershell
python -m uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 64
```

The observed test results were:

| Run | Learning rate | Batch size | Epochs | Test accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.001 | 32 | 5 | 40.69% |
| 2 | 0.01 | 32 | 5 | 12.96% |
| 3 | 0.0001 | 32 | 5 | 76.09% |
| 4 | 0.001 | 64 | 5 | 59.95% |

The recorded validation accuracy for Run 3 was **0.7171**, or **71.71%**.
Validation accuracy and test accuracy are measured on separate datasets
and should not be confused.

### Question 7 — Effect of the learning rate

The selected configuration used a learning rate of 0.0001 and a batch
size of 32. It achieved 71.71% validation accuracy and 76.09% test accuracy.

A higher learning rate did not necessarily improve performance.
With batch size 32, increasing the learning rate to 0.01 produced
only 12.96% test accuracy, compared with 40.69% at 0.001.

A possible explanation is that larger parameter updates disrupted
the useful pretrained weights. A smaller learning rate allowed
more gradual adaptation to Food-11.

### Question 8 — Comparing configurations

A parallel coordinates plot with `lr`, `batch_size` and `val_accuracy`
allows the relationship between the configuration and validation
performance to be examined. Each line represents one run.

The test results show that, at a learning rate of 0.001, increasing
the batch size from 32 to 64 improved test accuracy from 40.69%
to 59.95%. The highest test accuracy was achieved with learning rate
0.0001 and batch size 32.

These observations concern the recorded test scores. The other three
validation scores were not transcribed into this report, so they do
not establish a complete ranking by validation accuracy or a verified
pattern in the validation parallel coordinates plot.

### Question 9 — Selected run and Run ID

The selected run has the following configuration and results:

- Model: pretrained ResNet18, fine-tuned for 11 classes.
- Dataset: food11_processed_mini.
- Learning rate: 0.0001.
- Batch size: 32.
- Epochs: 5.
- Validation accuracy: 71.71%.
- Test accuracy: 76.09%.
- Run ID: `84ab21a8d12c44f99a06359b23bb277e`.

This Run ID identifies the experiment execution and provides access
to its parameters, metric history and associated model.

It has the highest test accuracy among the four recorded configurations.
To confirm that it is also the best validation run, all four runs must
be compared by `val_accuracy`.

## Conclusion

This lab demonstrated how to train an image classifier and track its
experiments with MLflow. The four configurations produced different
results, showing that learning rate and batch size affect performance.

The selected configuration achieved 71.71% validation accuracy and
76.09% test accuracy. MLflow keeps the configuration, metrics and model
associated with each run, making the experiments easier to inspect
and compare.