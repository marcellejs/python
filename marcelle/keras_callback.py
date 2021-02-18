import tensorflow as tf
from .writer import Writer


class KerasCallback(tf.keras.callbacks.Callback):
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
        super(KerasCallback, self).__init__()
        self.model_checkpoint_freq = model_checkpoint_freq
        self.run_params = run_params
        self.writer = Writer(
            backend_root=backend_root,
            runs_path=runs_path,
            disk_save_formats=disk_save_formats,
            remote_save_format=remote_save_format,
            base_log_dir=base_log_dir,
        )

    def on_train_begin(self, logs=None):
        self.writer.create_run({**self.run_params, **self.params})
        self.writer.train_begin(self.params["epochs"])

    def on_epoch_end(self, epoch, logs=None):
        save_checkpoint = (
            self.model_checkpoint_freq is not None
            and (epoch + 1) % self.model_checkpoint_freq == 0
        )

        self.writer.epoch_end(epoch, logs, save_checkpoint)

    def on_train_end(self, logs=None):
        self.writer.train_end(logs, save_checkpoint=True)
