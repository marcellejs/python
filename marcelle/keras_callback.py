import os
import json
import subprocess
import tensorflow as tf
import tensorflowjs as tfjs
from .remote import MarcelleRemote
import keras2onnx


class MarcelleCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        backend_root="http://localhost:3030",
        runs_path="runs",
        disk_save_formats=["h5", "tfjs"],
        remote_save_format="tfjs",
        model_checkpoint_freq=None,
        base_log_dir="marcelle-logs",
        run_params={},
    ):
        super(MarcelleCallback, self).__init__()
        self.model_checkpoint_freq = model_checkpoint_freq
        self.base_log_dir = base_log_dir
        self.log_folder = None
        self.disk_save_formats = disk_save_formats
        if remote_save_format not in disk_save_formats:
            self.disk_save_formats.append(remote_save_format)
        self.remote_save_format = remote_save_format
        self.run_params = run_params
        self.remote = MarcelleRemote(backend_root, runs_path, remote_save_format)

    def init_disk(self):
        # check if logs folder exist in marcelle-logs folder
        subprocess.call(["mkdir", "-p", self.base_log_dir])
        self.log_folder = os.path.join(
            self.base_log_dir, "-".join(self.remote.run_data["run_start_at"].split(":"))
        )
        subprocess.call(["mkdir", "-p", self.log_folder])

    def write_to_disk(self):
        json.dump(
            self.remote.run_data,
            open(os.path.join(self.log_folder, "run_data.json"), "w"),
        )

    def write_models_to_disk(self, epoch):
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

    def on_train_begin(self, logs=None):
        self.remote.create_run(self.model, {**self.run_params, **self.params})
        self.remote.train_begin(self.params["epochs"])

        self.init_disk()
        self.write_to_disk()

    def on_epoch_end(self, epoch, logs=None):
        checkpoint = None
        if (
            self.model_checkpoint_freq is not None
            and (epoch + 1) % self.model_checkpoint_freq == 0
        ):
            tfjs_model_path = self.write_models_to_disk(epoch)
            checkpoint = {"epoch": epoch, "local_path": tfjs_model_path}
            remote_checkpoint = self.remote.upload_model(tfjs_model_path)
            checkpoint = {**checkpoint, **remote_checkpoint}

        self.remote.epoch_end(epoch, logs, checkpoint)
        self.write_to_disk()

    def on_train_end(self, logs=None):
        tfjs_model_path = self.write_models_to_disk("final")
        checkpoint = {"epoch": "final", "local_path": tfjs_model_path}
        remote_checkpoint = self.remote.upload_model(tfjs_model_path)
        checkpoint = {**checkpoint, **remote_checkpoint}

        self.remote.train_end(checkpoint)
        self.write_to_disk()
