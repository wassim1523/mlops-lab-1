# Lab 1 — Git, DVC and data preparation

## Objective

The objective of this lab is to understand how Git and DVC work together in a
machine-learning project. Git versions the source code, configuration files and
DVC pointer files, while DVC versions the datasets without storing the image
files directly in GitHub.

## Adopted solution for DVC storage

I used the recommended **local DVC remote** because uploading the complete
Food-11 dataset to DagsHub was difficult. The storage directory was created
outside the Git repository and configured as the default DVC remote. Therefore,
the Git repository contains only the code, DVC configuration and data pointer,
while the actual dataset is stored in the external local directory.

Example configuration on Windows:

```bash
dvc remote add -d localstorage "C:/Users/<username>/dvc-storage/mlops-lab-1"
git add .dvc/config
git commit -m "Configure local DVC remote"
git push
```

Example configuration on Linux or macOS:

```bash
mkdir -p ../dvc-storage/mlops-lab-1
dvc remote add -d localstorage ../dvc-storage/mlops-lab-1
git add .dvc/config
git commit -m "Configure local DVC remote"
git push
```

The chosen storage folder must remain outside `mlops-lab-1`.

## Project setup

```bash
git clone https://github.com/<github-username>/mlops-lab-1.git
cd mlops-lab-1

pip install uv
uv init
uv add dvc pillow

dvc init
git add .dvc .dvcignore
git commit -m "Initialize git and dvc"
git push
```

### Question 1 — Files created by `uv init`

Depending on the installed `uv` version and selected project type, `uv init`
creates files such as:

- `pyproject.toml`: contains project metadata, the required Python version and
  project dependencies.
- `main.py`: a small initial Python entry point in the default application
  layout.
- `README.md`: provides a description and instructions for the project.
- `.python-version`: indicates the Python version to use for the project.
- `.gitignore`: tells Git which generated or local files must not be tracked.
- `uv.lock`: created after dependency resolution; records the exact resolved
  dependency versions to make the environment reproducible.

The precise initial list can vary slightly according to the version of `uv`.

### Question 2 — Files created by `dvc init`

`dvc init` creates:

- `.dvc/config`: the repository-level DVC configuration, including remotes that
  are safe to share.
- `.dvc/.gitignore`: prevents Git from tracking DVC cache files and temporary
  files stored inside `.dvc`.
- `.dvcignore`: allows files or paths to be ignored by DVC, similarly to
  `.gitignore`.

The `.dvc` directory also hosts internal/cache content during later operations.
The tracked configuration and ignore files should be pushed to Git, but cache
contents and temporary files should not. The `dvc init` command prepares the
correct ignore rules automatically.

### Question 3 — Credentials and configuration scopes

With `--global`, DVC writes configuration to the user's global DVC config,
normally under the user's configuration directory rather than inside the Git
repository.

The main alternatives are:

- no scope option: repository configuration in `.dvc/config`;
- `--local`: private repository configuration in `.dvc/config.local`;
- `--system`: machine-wide configuration;
- `--global`: user-wide configuration.

Credentials must never be pushed to GitHub. A good practice is to store the
remote URL in `.dvc/config` and secrets in `.dvc/config.local`, environment
variables, or a secure credential store. DVC ignores `.dvc/config.local`.

Because this submission uses a local remote, no DagsHub password is required.

## Adding and versioning the raw data

The downloaded files were placed in the following structure:

```text
data/food11_raw/training/
data/food11_raw/evaluation/
data/food11_raw/validation/
```

The data was then tracked and stored with:

```bash
dvc add data
git add data.dvc .gitignore
git commit -m "Track data folder with dvc"
git push
dvc push
```

### Question 4 — Modification of `.gitignore`

After `dvc add data`, DVC adds `/data` to `.gitignore`. This prevents Git from
tracking and uploading the actual dataset. The dataset is handled by DVC,
whereas Git tracks the small `data.dvc` pointer file.

### Question 5 — Contents of `data.dvc`

Yes, DVC creates `data.dvc`. It is a small YAML pointer file that describes the
tracked output. It contains information such as the data hash, total size,
number of files and path:

```yaml
outs:
- md5: <directory-hash>.dir
  size: <total-size>
  nfiles: <number-of-files>
  hash: md5
  path: data
```

The exact values depend on the local dataset. The hash identifies the precise
version of the data stored in the DVC cache/remote.

### Question 6 — GitHub and DVC remote contents

On GitHub, the source code and `data.dvc` are visible, but the image files under
`data/` are not present because that directory is ignored by Git. The
`data.dvc` file points indirectly to the correct dataset version through its
content hash.

After `dvc push`, the actual data objects are present in the configured local
DVC remote. Unlike a DagsHub remote, this local storage is visible only on the
computer or shared filesystem where it was created.

### Question 7 — Retrieving the data after cloning

A new Git clone contains the code and pointer files, but not the actual data
directory. After installing the dependencies, the data is retrieved with:

```bash
dvc pull
```

For a local remote, the cloned repository must be able to access the same
external storage path. If its relative location differs, the remote can be
overridden privately in the clone:

```bash
dvc remote modify --local localstorage url "/absolute/path/to/dvc-storage/mlops-lab-1"
dvc pull
```

## Data preparation

The script `src/food11/data.py` performs the following work:

1. Reads the images from `data/food11_raw`.
2. Extracts the class number from the beginning of each filename.
3. Converts each image to RGB and resizes it to `128 × 128` pixels.
4. Saves it in the correct category directory under `food11_processed`.
5. Copies at most 100 images for every category and split into
   `food11_processed_mini`.

It is run from the repository root with:

```bash
uv run python ./src/food11/data.py
```

The new version of the data is then stored with:

```bash
dvc add data
git add data.dvc
git commit -m "Add food11_processed and food11_processed_mini"
git push
dvc push
```

## Switching between code and data versions

```bash
git log --oneline -- data.dvc
git checkout <old-commit-hash>
dvc checkout
```

### Question 8 — Result after checking out the old commit

After checking out the commit that only referenced the raw dataset and running
`dvc checkout`, `food11_processed` and `food11_processed_mini` disappear from
the working data directory. This happens because the old `data.dvc` pointer
describes the earlier version of the complete `data` directory.

The latest version is restored with:

```bash
git checkout main
dvc checkout
```

The processed directories then reappear, provided that their corresponding
objects are available in the DVC cache or configured remote.

## Conclusion

This lab shows the separation of responsibilities between Git and DVC. Git
stores the reproducible project structure, code and lightweight pointers. DVC
stores and restores the large datasets matching each Git commit. As a result,
it is possible to move between two project versions while keeping both the code
and its corresponding data synchronized.
