import json
import os


class MarcelleUploader:
    def __init__(self, remote):
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
        if overwrite:
            raise Exception("Overwrite mode not yet implemented")
        if not os.path.exists(run_directory) or not os.path.isdir(run_directory):
            raise Exception(f"Directory {run_directory} does not exist os ir invalid")
        self.reset()
        with open(os.path.join(run_directory, "run_data.json"), "r") as json_file:
            self.run_data = json.load(json_file)
        start = self.run_data["run_start_at"]
        print(f"Retrieving remote run '{start}'...")
        run_exists = self.remote.retrieve_run(start)
        if run_exists:
            dict_equal = True
            for key in self.run_data:
                if not (
                    key in self.remote.run_data
                    and self.run_data[key] == self.remote.run_data[key]
                ):
                    dict_equal = False
                    break
            if dict_equal:
                print(f"Run {start} is up to date on the server")
                return
            else:
                print(f"Run {start} already exists on the server, updating...")
        else:
            print(f"Run {start} not found on the server, uploading...")
        self.upload_new_checkpoints()
        with open(os.path.join(run_directory, "run_data.json"), "w") as json_file:
            json.dump(self.run_data, json_file)
        if run_exists:
            self.remote.update(self.run_data)
        else:
            self.remote.create(self.run_data)
        print("Done")

    def upload_new_checkpoints(self):
        upload_count = 0
        for i, checkpoint in enumerate(self.run_data["checkpoints"]):
            if "model_id" in checkpoint:
                continue
            remote_checkpoint = self.remote.upload_model(checkpoint["local_path"])
            self.run_data["checkpoints"][i] = {**checkpoint, **remote_checkpoint}
            upload_count += 1
        print(f"{upload_count} models were uploaded to marcelle")
