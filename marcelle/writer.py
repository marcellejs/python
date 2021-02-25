from datetime import datetime
import os
import json
import subprocess
from tensorflow import saved_model
import tensorflowjs as tfjs
import keras2onnx
from .remote import Remote
from .utils import conform_dict, get_model_info


class Writer:
    def __init__(
        self,
        backend_root="http://localhost:3030",
        disk_save_format="h5",
        remote_save_format="tfjs",
        base_log_dir="marcelle-logs",
        source="keras",
    ):
        self.base_log_dir = base_log_dir
        self.log_folder = None
        self.disk_save_format = disk_save_format
        self.remote_save_format = remote_save_format
        self.remote = Remote(
            backend_root=backend_root,
            save_format=remote_save_format,
            source=source,
        )
        self.source = source
        self.run_data = None

    def create_run(self, model, run_params={}, loss=None):
        begin_date = datetime.now()
        self.run_data = conform_dict(
            {
                "run_start_at": begin_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": self.source,
                "status": "idle",
                "params": run_params,
                "model": get_model_info(model, self.source, loss),
                "logs": {},
                "checkpoints": [],
                "assets": [],
            }
        )
        self.remote.create(self.run_data)
        self.model = model
        self.__init_disk()
        self.__write_to_disk()

    def train_begin(self, epochs):
        self.run_data["status"] = "start"
        self.run_data["epoch"] = 0
        self.run_data["epochs"] = conform_dict(epochs)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def save_epoch(self, epoch, logs=None, save_checkpoint=False, assets=[]):
        self.run_data["status"] = "epoch"
        self.run_data["epoch"] = epoch
        if len(self.run_data["logs"]) == 0:
            [self.run_data["logs"].setdefault(x, []) for x in list(logs.keys())]
        for k in list(logs.keys()):
            self.run_data["logs"][k].append(logs[k])

        if save_checkpoint:
            self.save_checkpoint(epoch)
        if len(assets) > 0:
            for asset in assets:
                self.save_asset(epoch, asset)

        self.run_data = conform_dict(self.run_data)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def save_checkpoint(self, epoch, model=None, metadata={}):
        if model is not None:
            self.model = model
        model_path = self.__write_models_to_disk(epoch)
        checkpoint_meta = {
            "epoch": epoch,
            "local_path": model_path,
            "local_format": self.disk_save_format,
            **metadata,
        }
        remote_checkpoint = self.remote.upload_model(
            model_path,
            local_format=self.disk_save_format,
            metadata=checkpoint_meta,
        )
        checkpoint = {**checkpoint_meta, **remote_checkpoint}

        self.run_data["checkpoints"].append(checkpoint)

        self.run_data = conform_dict(self.run_data)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def save_asset(self, epoch, asset_path, metadata={}):
        asset_meta = {"epoch": epoch, "local_path": asset_path, **metadata}
        remote_asset = self.remote.upload_asset(asset_path, asset_meta)
        asset = {**asset_meta, **remote_asset}

        self.run_data["assets"].append(asset)

        self.run_data = conform_dict(self.run_data)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def train_end(self, logs=None, save_checkpoint=False):
        self.run_data["status"] = "success"
        if save_checkpoint:
            self.save_checkpoint("final")

        self.remote.update(self.run_data)
        self.__write_to_disk()

    def __init_disk(self):
        # check if logs folder exist in marcelle-logs folder
        subprocess.call(["mkdir", "-p", self.base_log_dir])
        self.log_folder = os.path.join(
            self.base_log_dir, "-".join(self.run_data["run_start_at"].split(":"))
        )
        subprocess.call(["mkdir", "-p", self.log_folder])

    def __write_to_disk(self):
        with open(os.path.join(self.log_folder, "run_data.json"), "w") as f:
            json.dump(self.run_data, f)

    def __write_models_to_disk(self, epoch):
        basename = (
            f"model_checkpoint_epoch={epoch}"
            if type(epoch) == int
            else "model_checkpoint_final"
        )
        model_path = os.path.join(self.log_folder, basename)
        if self.disk_save_format == "h5":
            model_path += ".h5"
            self.model.save(model_path)
        elif self.disk_save_format == "saved_model":
            saved_model.save(self.model, model_path)
        if self.disk_save_format == "tfjs":
            tfjs.converters.save_keras_model(self.model, model_path)
        if self.disk_save_format == "onnx":
            model_path += ".onnx"
            onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
            keras2onnx.save_model(onnx_model, model_path)
        return model_path
