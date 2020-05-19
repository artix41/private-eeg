from torch import nn
from models.mnist import MNIST_CNN
from models.eeg import EEG_CNN


class Model:
    def __init__(self, name, clients):
        self.name = name
        self.clients = clients
        self.n_clients = len(clients)

        self.get_model_method = {"MNIST_CNN": MNIST_CNN,
                                 "EEG_CNN": EEG_CNN}
        self.loss_fn_dict = {"MNIST_CNN": nn.NLLLoss(),
                             "EEG_CNN": nn.NLLLoss()}

        self.init_model = self.get_model_method[self.name]()
        self.model_client = [self.init_model.copy() for _ in range(self.n_clients)]

    def get_loss_fn(self):
        return self.loss_fn_dict[self.name]

    def send_model_to_clients(self):
        self.model_client = [self.model_client[i].send(self.clients[i]) for i in range(self.n_clients)]

    def get_model_client(self, client):
        if len(self.model_client) == 0:
            raise ValueError("You need to first send the model to the clients (using model.send_model_to_clients()")

        return self.model_client[client]
