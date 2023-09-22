
from typing import Any
from multiprocessing import  Queue, Process
import weakref
import uuid
from marcelle.data_store import DataStore
from marcelle.data_store_http import DataStoreHTTP


  
class TrainingServer():
    def __init__(self,
                 model_parameters: dict,
                 training_fonction: callable,
                 datastore_url: str,
                 main_service: str|None ="training_server",
                 worker_service: str|None =None,
                 training_service: str|None =None,
                 worker_size: int = 2,
                 worker_id : str = None
                 ) -> None:
        """Training Server is responsible to listen on a featherjs service and upon a training creation, launch it locally 

        Args:
            model_parameters (dict): the accepted parameters of this server
            training_fonction (callable): the training function of the model handled by this server
            datastore_url (str): the url of the marcelle backend e.g: http://localhost:3030"
            main_service (str | None, optional): base name of this server on the backend. Defaults to "training_server".
            worker_service (str | None, optional): Override the woker service name on marcelle backend. Defaults to None.
            training_service (str | None, optional): Override the training service name on marcelle backend. Defaults to None.
            worker_size (int, optional): The maximum number of task that can run simultaneously. Defaults to 2.
        """
        # map with the backend
        self.datastore_url = datastore_url
        self.datastore = DataStore(datastore_url)

        self.training_service_name = "{}_training_service".format(
            main_service) if not training_service else training_service
        self.worker_service_name = "{}_worker_service".format(
            main_service) if not worker_service else worker_service

        self.training_service_conn = self.datastore.service(
            self.training_service_name)
        self.worker_service_conn = self.datastore.service(
            self.worker_service_name)
        
        
        self.training_fonction = training_fonction

        # the mongo db id of the Worker
        self.mongo_id= None
        self.model_parameters = model_parameters
        # instance of the workers
        self.worker_size = worker_size
        # queue for IPC
        self.job_queue = Queue()
        # list of worker process
        self.executor_pool = []
        self.worker_id = worker_id
        
        weakref.finalize(self, self._unregister_server)

    def _register_server(self) -> None:
        """Register this server against the marcelle backend
        """    
        res = self.worker_service_conn.create({
            "worker_id": self.worker_id,
            "model_parameters": self.model_parameters,
        })
        self.mongo_id = res.get("_id")
        self.training_service_conn.on("created", self.submit_training)

    def _unregister_server(self):
        """Unregister this server on the marcelle backend
        """ 
        self.worker_service_conn.remove(self.mongo_id)
    
   
    def submit_training(self, training_message):
        """Add an incoming training in the job Queue

        Args:
            training_message (Dict): a training_event created in the training
                    {
                    _id: ObjectId('64a6c3f06cda3efb6c4dcc2a'),
                    worker_id: '6804b369ad864492994b0ffd0cd56b7e',
                    model_parameters: {
                        lr: 'float',
                        gamma: 'float',
                        epochs: 'int'
                        },
                    }
        """
        worker_id = training_message.get("worker_id")
        if worker_id == str(self.worker_id):
            training_parameters = training_message.get("training_parameters")
            training_id = training_message.get("_id")

            self.training_service_conn.patch(
                id=training_id, data={"state": "pending"})

            self.job_queue.put({"params": training_parameters,
                                "training_id": training_id, "wid": worker_id})

            print("message training {} received".format(training_id))
    
    def start(self):
        """Start the server loop and register it
        """        
        self.datastore.connect()
        for i in range(self.worker_size):
            p = TrainingWorker(self.training_fonction, self.datastore_url,
                                self.training_service_name, self.job_queue)
            self.executor_pool.append(p)
            p.start()

        if not self.worker_id:
            self.worker_id = uuid.uuid4().hex 
        print("worker ID: {}".format(self.worker_id))

        # declare this worker available
        self._register_server()
        weakref.finalize(self, self._unregister_server)

    def stop(self):
        """Unregister and stop the server loop
        """        
        self._unregister_server()   
        for p in self.executor_pool:
            p.terminate()
            p.join()
        self.datastore.disconnect()

class TrainingWorker(Process):
    def __init__(self, fn : callable, datastore_url : str, training_service : str, job_queue: Queue) -> None:
        """_summary_

        Args:
            fn (callable): The Training function of the worker 
            datastore_url (str): the url of the the datastore
            training_service (str): the name of the  marcelle training service
            job_queue (Queue): the job Queue 
        """
        Process.__init__(self)
        self.jq = job_queue
        self.fn = fn
        self.datastore = DataStoreHTTP(datastore_url)


        self.training_service_name = training_service
        self.svc = self.datastore.service(self.training_service_name)

    def run(self) -> Any:
        while True:
            # waiting for a job to start 
            task = self.jq.get(block=True)
            # patching the training to indicate that the training has begun 
            training_id = task.get("training_id")
            self.svc.patch(id=training_id, data={"state": "running"})
            # run the training funvtion with the parameters
            print(self.fn(**task.get("params")))
            # patching the training to indicate that the training has finished 
            self.svc.patch(id=training_id, data={"state": "finished"})


