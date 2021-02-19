# Marcelle - Python Package

A python package for interacting with a Marcelle backend from python.

> See: [http://marcelle.dev](http://marcelle.dev)

## Status ⚠️

Marcelle is still experimental and is currently under active development. Breaking changes are expected.

## Installing

```shell
pip install .
```

## Basic Usage

### Keras Callback

```py
from marcelle import MarcelleCallback

mrc_callback = KerasCallback(
    backend_root="http://localhost:3030",
    disk_save_formats=["h5", "tfjs"],
    remote_save_format="tfjs",
    model_checkpoint_freq=None,
    base_log_dir="marcelle-logs",
    run_params={},
)

model.fit(
  # ...
  callbacks = [
    mrc_callback,
    # other callbacks
  ]
)
```

### Writer (for custom training loops)

```py
from marcelle import Writer

writer = Writer(
    backend_root="http://localhost:3030",
    disk_save_formats=["h5", "tfjs"],
    remote_save_format="tfjs",
    base_log_dir="marcelle-logs",
    source="keras",
)

writer.create_run(model, params, loss.name)
writer.train_begin(epochs)

for epoch in range(epochs):
  # ...
  logs = {
    "loss": 1.3,
    "accuracy": 0.7,
    "val_loss": 2.3,
    "val_accuracy": 0.52,
  }
  assets = ["path/to/asset1.wav", "path/to/asset2.wav"]
  self.writer.save_epoch(epoch, logs=logs, save_checkpoint=True, assets=assets)

writer.train_end(save_checkpoint=True)
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
            save_format="tfjs",
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
