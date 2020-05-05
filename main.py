import torch
from torch import nn, optim
import torch.nn.functional as F
import syft as sy
import numpy as np
import matplotlib.pyplot as plt
import math

from utils.data_loader import DataLoader
from models.model import Model

n_clients = 100
n_clients_round = int(n_clients * 0.2)

lr = 0.1
batch_size = 16
test_batch_size = 100


def main():
    hook = sy.TorchHook(torch)
    clients = [sy.VirtualWorker(hook, id=f"client{i}") for i in range(n_clients)]
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    data_loader = DataLoader("MNIST_iid", clients)
    data_loader.send_data_to_clients()

    model = Model("MNIST_CNN", clients)
    model.send_model_to_clients()

    trainer = FedAvg()

    trainer.train(data_loader)
    trainer.test()


if __name__ == "__main__":
    main()