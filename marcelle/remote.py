from glob import glob
import filetype
import os
import requests


class Remote:
    def __init__(
        self,
        backend_root="http://localhost:3030",
        save_format="tfjs",
        source="keras",
    ):
        super().__init__()
        self.backend_root = backend_root + ("" if backend_root[-1] == "/" else "/")
        self.save_format = save_format
        self.runs_url = self.backend_root + "runs"
        self.models_url = self.backend_root + f"{save_format}-models"
        self.assets_url = self.backend_root + "assets"
        self.source = source
        self.run_id = None

    def create(self, run_data=None):
        try:
            res = requests.post(self.runs_url, json=run_data)
            if res.status_code != 201:
                print(
                    "Error: Could not create run. Improve error message."
                    f"HTTP Status Code: {res.status_code}"
                )
            else:
                self.run_id = res.json()["_id"]
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))

    def update(self, run_data=None):
        if not self.run_id:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
            return
        try:
            res = requests.patch(
                self.runs_url + "/" + self.run_id,
                json=run_data,
            )
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))

    def upload_model(self, path_to_model, metadata={}):
        if self.source == "keras":
            if self.save_format == "tfjs":
                return self.upload_tfjs_model(path_to_model, metadata)
            elif self.save_format == "onnx":
                return self.upload_onnx_model(path_to_model, metadata)
            else:
                raise Exception(
                    f"Unknown save format `{self.save_format}`."
                    "Must be `tfjs` or `onnx`."
                )
        else:
            raise Exception("Only 'keras' source is implemented at the moment")

    def upload_tfjs_model(self, path_to_model, metadata={}):
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
            res = requests.post(self.models_url + "/upload", files=files)
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
                print(res.json()["error"])
                return {}
            model_url = res.json()["model.json"]
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at "
                + str(self.models_url + "/upload")
            )
        json_file.close()
        [f.close() for f in bin_files]
        if not model_url:
            return {}
        try:
            res = requests.post(
                self.models_url,
                json={
                    **metadata,
                    "url": model_url,
                    "format": self.save_format,
                },
            )
            if res.status_code != 201:
                print("An error occured with HTTP Status Code:", res.status_code)
            return res.json()
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.models_url)
            )
            return {}

    def upload_onnx_model(self, path_to_model, metadata={}):
        if ".onnx" not in path_to_model:
            path_to_model = f"{path_to_model}.onnx"
        onnx_file = open(path_to_model, "rb")
        filename = os.path.basename(path_to_model)
        files = [(filename, (filename, onnx_file, "application/octet-stream"))]
        model_url = None
        try:
            res = requests.post(self.models_url + "/upload", files=files)
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
                print(res.json()["error"])
                return {}
            model_url = res.json()[filename]
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at "
                + str(self.models_url + "/upload")
            )
        onnx_file.close()
        if not model_url:
            return {}
        try:
            res = requests.post(
                self.models_url,
                json={
                    **metadata,
                    "url": model_url,
                    "format": self.save_format,
                },
            )
            if res.status_code != 201:
                print("An error occured with HTTP Status Code:", res.status_code)
            return res.json()
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.models_url)
            )
            return {}

    def upload_asset(self, path_to_asset, metadata={}):
        asset_file = open(path_to_asset, "rb")
        kind = filetype.guess(path_to_asset)
        if kind is None:
            extension = os.path.splitext(path_to_asset)[1]
            mime = "application/octet-stream"
        else:
            extension = f".{kind.extension}"
            mime = kind.mime

        filename = os.path.basename(path_to_asset)
        files = [(filename, (filename, asset_file, mime))]
        asset_url = None
        try:
            res = requests.post(self.assets_url + "/upload", files=files)
            if res.status_code != 200:
                print("An error occured with HTTP Status Code:", res.status_code)
                print(res.json()["error"])
                return {}
            asset_url = res.json()[filename]
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at "
                + str(self.assets_url + "/upload")
            )
        asset_file.close()
        if not asset_url:
            return {}
        try:
            res = requests.post(
                self.assets_url,
                json={
                    **metadata,
                    "filename": filename,
                    "extension": extension,
                    "mime": mime,
                    "url": asset_url,
                },
            )
            if res.status_code != 201:
                print("An error occured with HTTP Status Code:", res.status_code)
            return res.json()
        except requests.exceptions.RequestException:
            print(
                "Warning: could not reach Marcelle backend at " + str(self.assets_url)
            )
            return {}

    def retrieve_run(self, run_start_at):
        run_data = False
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
                    run_data = res_json["data"][0]
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
        return run_data

    def remove_run(self, run_data):
        for checkpoint in run_data["checkpoints"]:
            try:
                req_url = self.models_url + "/" + checkpoint["_id"]
                res = requests.delete(req_url)
                print("remove: res.status_code=", res.status_code)
                if res.status_code != 200:
                    print(
                        f"An error occured with HTTP Status Code: {res.status_code}\n"
                        f"Request URL: {req_url}"
                    )
            except requests.exceptions.RequestException:
                print(
                    "Warning: could not reach Marcelle backend at "
                    + str(self.models_url)
                )
        try:
            res = requests.delete(self.runs_url + "/" + run_data["_id"])
            print("remove: res.status_code=", res.status_code)
            if res.status_code != 200:
                print(f"An error occured with HTTP Status Code: {res.status_code}")
            else:
                res_json = res.json()
                start_date = res_json["run_start_at"]
                print(f"Removed run {id} from server (run start date: {start_date})")
            return True
        except requests.exceptions.RequestException:
            print("Warning: could not reach Marcelle backend at " + str(self.runs_url))
            return False
