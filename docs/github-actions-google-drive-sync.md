# GitHub Actions sync to Google Drive for Colab

This repository syncs Colab notebooks and runtime source files to Google Drive whenever code is pushed to `main`.

Workflow file:

- `.github/workflows/sync-colab-drive.yml`

The workflow uploads:

- `src/**/*.ipynb`
- `src/**/*.py`
- `config/**` if the directory exists
- `requirements*.txt`
- `pyproject.toml`

It skips Jupyter checkpoint files and Python cache directories.

## GitHub repository settings

Open your GitHub repository, then go to:

`Settings` -> `Secrets and variables` -> `Actions`

### Repository Secrets

Create this secret:

| Name | Value |
| --- | --- |
| `GDRIVE_SERVICE_ACCOUNT_JSON` | The full JSON key for a Google Cloud service account that has access to the target Google Drive folder.. |

### Repository Variables

Create this variable:

| Name | Value |
| --- | --- |
| `GDRIVE_FOLDER_ID` | The Google Drive folder ID where the repo files should be synced. |

Example Drive folder URL:

```text
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz
```

For that URL, set:

```text
GDRIVE_FOLDER_ID=1AbCdEfGhIjKlMnOpQrStUvWxYz
```

## Google Drive setup`

1. In Google Cloud Console, create or choose a project.
2. Enable the Google Drive API for that project.
3. Create a service account.
4. Create a JSON key for the service account.
5. Copy the whole JSON file content into the GitHub secret `GDRIVE_SERVICE_ACCOUNT_JSON`.
6. In Google Drive, create a folder for this project.
7. Share that folder with the service account email, for example:

```text
github-colab-sync@your-project.iam.gserviceaccount.com
```

Give it `Editor` access.

## How Colab receives the new files

Google Colab reads notebooks directly from Google Drive. After a push to `main` completes:

1. Wait for the GitHub Actions workflow `Sync Colab notebooks to Google Drive` to finish successfully.
2. Open Colab.
3. Choose `File` -> `Open notebook` -> `Google Drive`.
4. Open the notebook from the synced Drive folder.

If a notebook is already open in Colab, close and reopen it from Drive after the workflow finishes. Colab runtimes do not automatically reload notebook source code that changed in Drive while the runtime is already active.

For Python imports from the synced `src` directory, mount Drive in the notebook and add the synced project path to `sys.path`, for example:

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
PROJECT_DIR = '/content/drive/MyDrive/path-to-synced-folder'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if f'{PROJECT_DIR}/src' not in sys.path:
    sys.path.insert(0, f'{PROJECT_DIR}/src')
```

Replace `path-to-synced-folder` with the visible path of your Google Drive sync folder.

## Manual sync

The workflow also supports manual runs:

`Actions` -> `Sync Colab notebooks to Google Drive` -> `Run workflow`

