# Marcelle - Python Package

A python package for interacting with a Marcelle backend from python.

> See: [http://marcelle.dev](http://marcelle.dev)

## Status ⚠️

Marcelle is still experimental and is currently under active development. Breaking changes are expected.

## Installing

```shell
pip install marcelle
```

## Basic Usage

### Keras Callback

```py
from marcelle import MarcelleCallback

mrc_callback = MarcelleCallback(
    backend_root="http://localhost:3030",
    runs_path="runs",
    disk_save_formats=["h5", "tfjs"],
    remote_save_format="tfjs",
    model_checkpoint_freq=1,
    base_log_dir="marcelle-logs",
    run_params={
      "learning_rate": 1e-3,
      "some_param": "custom_value",
    }
)

model.fit(
  # ...
  callbacks = [
    mrc_callback,
    # other callbacks
  ]
)
```

### Batch upload

Useful when training was done offline or connection to server failed during the training.

```py
from glob import glob
import os
from marcelle import MarcelleRemote, MarcelleUploader


if __name__ == "__main__":
    LOG_DIR = "marcelle-logs"
    uploader = MarcelleUploader(
        MarcelleRemote(
            backend_root="http://localhost:3030",
            runs_path="runs",
            models_path="tfjs-models",
            source="keras",
        )
    )
    runs = [d for d in glob(os.path.join(LOG_DIR, "*")) if os.path.isdir(d)]
    for run in runs:
        uploader.upload(run)

```

## ✍️ Authors

- [@JulesFrancoise](https://github.com/JulesFrancoise/)
- [@bcaramiaux](https://github.com/bcaramiaux/)
