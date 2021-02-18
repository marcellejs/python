import os
import json
import subprocess
import tensorflowjs as tfjs
from .remote import Remote
import keras2onnx


class Writer:
    def __init__(
        self,
        backend_root="http://localhost:3030",
        runs_path="runs",
        disk_save_formats=["h5", "tfjs"],
        remote_save_format="tfjs",
        base_log_dir="marcelle-logs",
    ):
        self.base_log_dir = base_log_dir
        self.log_folder = None
        self.disk_save_formats = disk_save_formats
        if remote_save_format not in disk_save_formats:
            self.disk_save_formats.append(remote_save_format)
        self.remote_save_format = remote_save_format
        self.remote = Remote(backend_root, runs_path, remote_save_format)

    def create_run(self, model, run_params={}):
        self.remote.create_run(model, run_params)
        self.model = model
        self.__init_disk()
        self.__write_to_disk()

    def train_begin(self, epochs):
        self.remote.train_begin(epochs)
        self.__write_to_disk()

    def epoch_end(self, epoch, logs=None, save_checkpoint=False):
        checkpoint_data = None
        if save_checkpoint:
            tfjs_model_path = self.__write_models_to_disk(epoch)
            checkpoint_data = {"epoch": epoch, "local_path": tfjs_model_path}
            remote_checkpoint = self.remote.upload_model(tfjs_model_path)
            checkpoint_data = {**checkpoint_data, **remote_checkpoint}

        self.remote.epoch_end(epoch, logs, checkpoint_data)
        self.__write_to_disk()

    def train_end(self, logs=None, save_checkpoint=True):
        checkpoint_data = None
        if save_checkpoint:
            tfjs_model_path = self.__write_models_to_disk("final")
            checkpoint_data = {"epoch": "final", "local_path": tfjs_model_path}
            remote_checkpoint = self.remote.upload_model(tfjs_model_path)
            checkpoint_data = {**checkpoint_data, **remote_checkpoint}

        self.remote.train_end(checkpoint_data)
        self.__write_to_disk()

    def __init_disk(self):
        # check if logs folder exist in marcelle-logs folder
        subprocess.call(["mkdir", "-p", self.base_log_dir])
        self.log_folder = os.path.join(
            self.base_log_dir, "-".join(self.remote.run_data["run_start_at"].split(":"))
        )
        subprocess.call(["mkdir", "-p", self.log_folder])

    def __write_to_disk(self):
        with open(os.path.join(self.log_folder, "run_data.json"), "w") as f:
            json.dump(self.remote.run_data, f)

    def __write_models_to_disk(self, epoch):
        basename = (
            f"model_checkpoint_epoch={epoch}"
            if type(epoch) == int
            else "model_checkpoint_final"
        )
        basepath = os.path.join(self.log_folder, basename)
        if "h5" in self.disk_save_formats:
            self.model.save(os.path.join(self.log_folder, f"{basename}.h5"))
        if "tfjs" in self.disk_save_formats:
            tfjs.converters.save_keras_model(self.model, basepath)
        if "onnx" in self.disk_save_formats:
            onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
            keras2onnx.save_model(
                onnx_model, os.path.join(self.log_folder, f"{basename}.onnx")
            )
        return basepath
