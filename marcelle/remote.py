from datetime import datetime
from glob import glob
import os
import requests
from numpy import sum
from tensorflow.keras.backend import count_params
from .utils import conform_dict


class Remote:
    def __init__(
        self,
        backend_root="http://localhost:3030",
        runs_path="runs",
        save_format="tfjs",
        source="keras",
    ):
        super().__init__()
        self.backend_root = backend_root + ("" if backend_root[-1] == "/" else "/")
        self.save_format = save_format
        self.runs_url = self.backend_root + runs_path
        self.models_url = self.backend_root + f"{save_format}-models"
        self.upload_url = self.backend_root + f"{save_format}-models" + "/upload"
        self.source = source
        self.run_data = {}
        self.run_id = None

    def create_run(self, model, params, loss=None):
        begin_date = datetime.now()
        self.run_id = None
        self.run_data = conform_dict(
            {
                "run_start_at": begin_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": self.source,
                "status": "idle",
                "params": params,
                "model": get_model_info(model, self.source, loss),
                "logs": {},
                "checkpoints": [],
            }
        )
        self.create()

    def train_begin(self, epochs):
        self.run_data["status"] = "start"
        self.run_data["epochs"] = epochs
        self.update()

    def epoch_end(self, epoch, logs, checkpoint=None):
        self.run_data["status"] = "epoch"
        self.run_data["epoch"] = epoch
        if len(self.run_data["logs"]) == 0:
            [self.run_data["logs"].setdefault(x, []) for x in list(logs.keys())]
        for k in list(logs.keys()):
            self.run_data["logs"][k].append(logs[k])

        if checkpoint:
            self.run_data["checkpoints"].append(checkpoint)

        self.update()

    def train_end(self, checkpoint):
        self.run_data["status"] = "success"
        if checkpoint:
            self.run_data["checkpoints"].append(checkpoint)

        self.update()

    def update(self, run_data=None):
        if run_data is not None:
            self.run_data = run_data
        self.run_data = conform_dict(self.run_data)
        if not self.run_id:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
            return
        try:
            res = requests.patch(
                self.runs_url + "/" + self.run_id,
                json=self.run_data,
            )
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))

    def create(self, run_data=None):
        if run_data is not None:
            self.run_data = run_data
        self.run_data = conform_dict(self.run_data)
        try:
            res = requests.post(
                self.runs_url,
                json=self.run_data,
            )
            if res.status_code != 201:
                print(
                    "Error: Could not create run. Improve error message."
                    f"HTTP Status Code: {res.status_code}"
                )
            else:
                self.run_id = res.json()["_id"]
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
        return self.run_data

    def upload_model(self, path_to_model):
        if self.source == "keras":
            if self.save_format == "tfjs":
                return self.upload_tfjs_model(path_to_model)
            elif self.save_format == "onnx":
                return self.upload_onnx_model(path_to_model)
            else:
                raise Exception(
                    f"Unknown save format `{self.save_format}`."
                    "Must be `tfjs` or `onnx`."
                )
        else:
            raise Exception("Only 'keras' source is implemented at the moment")

    def upload_tfjs_model(self, path_to_model):
        files = []
        json_file = open(os.path.join(path_to_model, "model.json"), "r")
        files.append(("model.json", ("model.json", json_file, "application/json")))
        model_files = glob(os.path.join(path_to_model, "*.bin"))
        bin_files = [open(model_file, "rb") for model_file in model_files]
        for i, f in enumerate(bin_files):
            files.append(
                (
                    os.path.basename(model_files[i]),
                    (os.path.basename(model_files[i]), f, "application/octet-stream"),
                )
            )
        model_url = None
        try:
            res = requests.post(self.upload_url, files=files)
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
            model_url = res.json()["model.json"]
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.upload_url)
            )
        json_file.close()
        [f.close() for f in bin_files]
        if not model_url:
            return {}
        model_id = None
        try:
            res = requests.post(
                self.models_url,
                json={
                    "modelName": self.run_data["model"]["name"],
                    "modelUrl": model_url,
                },
            )
            if res.status_code != 201:
                print("An error occured with HTTP Status Code:", res.status_code)
            model_id = res.json()["_id"]
            return {
                "model_url": model_url,
                "model_id": model_id,
                "format": self.save_format,
            }
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.upload_url)
            )
            return {}

    def upload_onnx_model(self, path_to_model):
        if ".onnx" not in path_to_model:
            path_to_model = f"{path_to_model}.onnx"
        onnx_file = open(path_to_model, "rb")
        filename = os.path.basename(path_to_model)
        files = [(filename, (filename, onnx_file, "application/octet-stream"))]
        model_url = None
        try:
            res = requests.post(self.upload_url, files=files)
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
            model_url = res.json()[filename]
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.upload_url)
            )
        onnx_file.close()
        if not model_url:
            return {}
        model_id = None
        try:
            res = requests.post(
                self.models_url,
                json={
                    "modelName": self.run_data["model"]["name"],
                    "modelUrl": model_url,
                },
            )
            if res.status_code != 201:
                print("An error occured with HTTP Status Code:", res.status_code)
            model_id = res.json()["_id"]
            return {
                "model_url": model_url,
                "model_id": model_id,
                "format": self.save_format,
            }
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.upload_url)
            )
            return {}

    def retrieve_run(self, run_start_at):
        success = False
        try:
            res = requests.get(
                self.runs_url + f"?source={self.source}&run_start_at={run_start_at}"
                "&$sort[createdAt]=-1"
            )
            if res.status_code != 200:
                print(f"An error occured with HTTP Status Code: {res.status_code}")
            else:
                res_json = res.json()
                if res_json["total"] > 0:
                    self.run_id = res_json["data"][0]["_id"]
                    self.run_data = res_json["data"][0]
                    success = True
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
        return success


def get_summary(model):
    summary_list = []
    model.summary(print_fn=lambda s: summary_list.append(s))
    return "\n".join(summary_list)


def count_model_params(model):
    trainable_count = int(sum([count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(
        sum([count_params(p) for p in model.non_trainable_weights])
    )
    return {
        "total": trainable_count + non_trainable_count,
        "trainable": trainable_count,
        "non_trainable": non_trainable_count,
    }


def get_layers_summary(model):
    # save model configuration (we can do better)
    model_summary = []
    for li, l in enumerate(model.layers):
        model_summary.append(
            {
                "layer_n": li,
                "name": l.name,
                "input_shape": l.input_shape,
                "output_shape": l.output_shape,
            }
        )
    return model_summary


def get_model_info(model, source, loss=None):
    if source == "keras":
        if loss is None and hasattr(model, "loss"):
            loss = model.loss if type(model.loss) == str else model.loss.name
        return {
            "name": model.name,
            "param_count": count_model_params(model),
            "summary": get_summary(model),
            "layers": get_layers_summary(model),
            "loss": loss or "Unknown",
        }
    else:
        raise Exception("Only 'keras' source is implemented at the moment")
