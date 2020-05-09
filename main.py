import torch
import syft as sy
import os

from utils.data_loader import DataLoader
from utils.plotter import Plotter
from models.model import Model
from trainers.fedavg import FedAvg

n_rounds = 5
n_clients = 100
n_clients_round = int(n_clients * 0.1)
n_local_epochs = 1

lr = 0.1
batch_size = 16
test_batch_size = 100

test_every = 1
save_every = 2

output_folder = "output"


def main():
    # Create clients
    hook = sy.TorchHook(torch)
    clients = [sy.VirtualWorker(hook, id=f"client{i}") for i in range(n_clients)]
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    # Load data
    data_loader = DataLoader("mnist_iid", clients)
    data_loader.send_data_to_clients()

    # Load model
    model = Model("MNIST_CNN", clients)
    model.send_model_to_clients()

    # Train model
    save_folder = os.path.join(output_folder, "model")
    trainer = FedAvg(model, data_loader, crypto_provider, save_folder)
    trainer.train(n_rounds, n_local_epochs, n_clients_round, lr, batch_size, test_every, save_every)

    # Plot results
    list_test_loss_client = trainer.list_test_loss_client
    list_test_rounds = trainer.list_test_rounds

    # list_test_loss_client = [[4, 2, 1, 0.5, 0.25]] * n_clients
    # list_test_rounds = list(range(0, n_rounds*2, test_every))
    plotter = Plotter(output_folder)
    plotter.plot_learning_curve_avg(list_test_rounds, list_test_loss_client)
    plotter.plot_learning_curve_clients(list_test_rounds, list_test_loss_client, n_clients=9)


if __name__ == "__main__":
    main()
