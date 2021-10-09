import tensorflow as tf
from .writer import Writer


class KerasCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        name,
        backend_root="http://localhost:3030",
        disk_save_format="h5",
        remote_save_format="tfjs",
        model_checkpoint_freq=None,
        base_log_dir="marcelle-logs",
        run_params={},
    ):
        """A Keras Callback to store training information in a Marcelle backend and locally.

        Args:
            name (str): The base name for the run.
            backend_root (str, optional): The backend's root URL.
                Defaults to "http://localhost:3030".
            disk_save_format (str, optional): Format used to store the models locally.
                Can be either "saved_model" or "h5". Defaults to "h5".
            remote_save_format (str, optional): Format used to upload the models to the
                backend. Can be either "tfjs" or "onnx". Defaults to "tfjs".
            model_checkpoint_freq (number, optional): The frequency at which checkpoints
                should be saved (in epochs). Defaults to None.
            base_log_dir (str, optional): Path to the directory where runs should be
                stored. Defaults to "marcelle-logs".
            run_params (dict, optional): A dictionary of parameters associated with
                the training run (e.g. hyperparameters). Defaults to {}.
        """
        super(KerasCallback, self).__init__()
        self.model_checkpoint_freq = model_checkpoint_freq
        self.run_params = run_params
        self.writer = Writer(
            name,
            backend_root=backend_root,
            disk_save_format=disk_save_format,
            remote_save_format=remote_save_format,
            base_log_dir=base_log_dir,
            source="keras",
        )

    def on_train_begin(self, logs=None):
        self.writer.create_run(
            self.model,
            run_params={**self.run_params, **self.params},
        )
        self.writer.train_begin(self.params["epochs"])

    def on_epoch_end(self, epoch, logs=None):
        save_checkpoint = (
            self.model_checkpoint_freq is not None
            and (epoch + 1) % self.model_checkpoint_freq == 0
        )
        self.writer.save_epoch(epoch + 1, logs, save_checkpoint)

    def on_train_end(self, logs=None):
        self.writer.train_end(logs)
