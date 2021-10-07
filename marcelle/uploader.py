import json
import os
from tqdm import tqdm


class Uploader:
    def __init__(self, remote):
        """The Uploader class alows to upload the results of a locally
        stored training run to the backend.

        Args:
            remote (Remote): An instance of Remote class

        TODO: Implement asset uploading
        """
        super().__init__()
        self.remote = remote
        self.run_data = {}
        self.existing_checkpoints = []

    def reset(self):
        self.run_data = {}
        self.run_id = None
        self.existing_checkpoints = []
        self.existing_checkpoints_epochs = []
        self.local_checkpoints = []
        self.local_checkpoints_epochs = []

    def upload(self, run_directory, overwrite=False):
        """Upload a training run from a directory

        Args:
            run_directory (string): run directory, from a Writer or Keras callback
            overwrite (bool, optional): If True, overwrites the data on the server,
            replacing run information and checkpoints. Defaults to False.

        Raises:
            Exception: If the input directory does not exist
        """
        if not os.path.exists(run_directory) or not os.path.isdir(run_directory):
            raise Exception(f"Directory {run_directory} does not exist os ir invalid")
        self.reset()
        with open(os.path.join(run_directory, "run_data.json"), "r") as json_file:
            self.run_data = json.load(json_file)
        start = self.run_data["start"]
        print(f"Retrieving remote run '{start}'...")
        remote_run_data = self.remote.retrieve_run(start)
        if remote_run_data and overwrite:
            self.remote.remove_run(remote_run_data)
            remote_run_data = False
        if remote_run_data:
            dict_equal = True
            for key in self.run_data:
                if not (
                    key in remote_run_data
                    and self.run_data[key] == remote_run_data[key]
                ):
                    dict_equal = False
                    break
            if dict_equal:
                print(f"Run {start} is up to date on the server")
                return
            else:
                print(f"Run {start} already exists on the server, updating...")
        else:
            if not overwrite:
                print(f"Run {start} not found on the server, uploading...")
            else:
                print(f"Run {start} was removed from the server, re-uploading...")
            overwrite = True
            self.remote.create(self.run_data)
        self.__upload_new_checkpoints(overwrite=overwrite)
        self.remote.update(self.run_data)
        with open(os.path.join(run_directory, "run_data.json"), "w") as json_file:
            json.dump(self.run_data, json_file)
        print("Done")

    def __upload_new_checkpoints(self, overwrite=False):
        upload_count = 0
        for i, checkpoint in tqdm(enumerate(self.run_data["checkpoints"])):
            if "id" in checkpoint:
                if overwrite:
                    del checkpoint["id"]
                else:
                    continue
            checkpoint_meta = checkpoint["metadata"]
            remote_checkpoint = self.remote.upload_model(
                checkpoint_meta["local_path"],
                checkpoint_meta["local_format"],
                checkpoint_meta,
            )
            if remote_checkpoint is not None:
                checkpoint = {
                    **checkpoint,
                    "id": remote_checkpoint["_id"],
                    "service": f"{self.remote.save_format}-models",
                }
            self.run_data["checkpoints"][i] = checkpoint
            upload_count += 1
        print(f"{upload_count} models were uploaded to marcelle")
