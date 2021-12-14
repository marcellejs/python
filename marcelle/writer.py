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
        name,
        backend_root="http://localhost:3030",
        disk_save_format="h5",
        remote_save_format="tfjs",
        base_log_dir="marcelle-logs",
        source="keras",
    ):
        """The Writer class allows to save training information locally and to a backend

        Args:
            name (str): The base name for the run.
            backend_root (str, optional): The backend's root URL.
                Defaults to "http://localhost:3030".
            disk_save_format (str, optional): Format used to store the models locally.
                Can be either "saved_model" or "h5". Defaults to "h5".
            remote_save_format (str, optional): Format used to upload the models to the
                backend. Can be either "tfjs" or "onnx". Defaults to "tfjs".
            base_log_dir (str, optional): Path to the directory where runs should be
                stored. Defaults to "marcelle-logs".
            source (str, optional): Source framework name. Only "keras" is
                currently fully supported. Defaults to "keras".

        TODO: take a remote instance as argument (as for uploader), and make it optional
        TODO: make Keras model optional
        """
        self.name = name
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

    def create_run(self, model=None, run_params={}, loss=None):
        """Create a new training run

        Args:
            model (keras.Model, optional): A keras.Model instance associated with the
                training run_params (dict, optional): A dictionary of parameters
                associated with the training run (e.g. hyperparameters). Defaults to {}.
            loss (string or loss function, optional): The loss function used for
                training. Defaults to None.
        """
        begin_date = datetime.now()
        self.run_data = conform_dict(
            {
                "basename": self.name,
                "start": begin_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": self.source,
                "status": "idle",
                "params": run_params,
                "logs": {},
                "checkpoints": [],
                "assets": [],
            }
        )
        if model is not None:
            self.run_data["model"] = get_model_info(model, self.source, loss)
            self.model = model
        self.remote.create(self.run_data)
        self.__init_disk()
        self.__write_to_disk()

    def train_begin(self, epochs):
        """Signal that the training has started

        Args:
            epochs (number): The total number of expected training epochs
        """
        self.run_data["status"] = "start"
        self.run_data["epoch"] = 0
        self.run_data["epochs"] = conform_dict(epochs)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def save_epoch(self, epoch, logs=None, save_checkpoint=False, assets=[]):
        """Save the results at the end of an epoch, with optional associated
        checkpoint and assets

        Args:
            epoch (number): the epoch
            logs (dict, optional): A dictionary of log values to record for the
                current epochs. The information should only concern the current
                epoch (for instance, logs for loss values should be scalar:
                `{"loss": 3.14}`). Defaults to None.
            save_checkpoint (bool, optional): If `True`, a checkpoint will be saved
                locally and uploaded to the backend, according to the formats specified
                in the constructor. Defaults to False.
            assets (list[string], optional): A list of assets paths associated with
                the epoch to upload. Defaults to [].
        """
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
        if self.model is not None:
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
            checkpoint = {
                "metadata": checkpoint_meta,
                "name": f"{self.run_data['name']}@{epoch}",
            }
            if remote_checkpoint is not None:
                checkpoint = {
                    "id": remote_checkpoint["_id"],
                    "name": f"{self.run_data['name']}@{epoch}",
                    "service": f"{self.remote_save_format}-models",
                    "metadata": checkpoint_meta,
                }

            self.run_data["checkpoints"].append(checkpoint)

            self.run_data = conform_dict(self.run_data)
            self.remote.update(self.run_data)
            self.__write_to_disk()
        else:
            raise Exception("Model is None!")

    def save_asset(self, epoch, asset_path, metadata={}):
        asset_meta = {"epoch": epoch, "local_path": asset_path, **metadata}
        remote_asset = self.remote.upload_asset(asset_path, asset_meta)
        asset = {**asset_meta, **remote_asset}

        self.run_data["assets"].append(asset)

        self.run_data = conform_dict(self.run_data)
        self.remote.update(self.run_data)
        self.__write_to_disk()

    def train_end(self, logs=None, save_checkpoint=True):
        """Signal that the training has ended

        Args:
            logs (dict, optional): Dictionary of logs (UNUSED?). Defaults to None.
            save_checkpoint (bool, optional): If `True`, a checkpoint will be saved
                locally and uploaded to the backend, according to the formats specified
                in the constructor. Defaults to True.
        """
        self.run_data["status"] = "success"
        if save_checkpoint:
            self.save_checkpoint("final")

        self.remote.update(self.run_data)
        self.__write_to_disk()

    def __init_disk(self):
        # check if logs folder exist in marcelle-logs folder
        subprocess.call(["mkdir", "-p", self.base_log_dir])
        self.log_folder = os.path.join(
            self.base_log_dir, "-".join(self.run_data["start"].split(":"))
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
