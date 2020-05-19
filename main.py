import torch
import syft as sy
import os
import numpy as np

from utils.data_loader import DataLoader
from utils.plotter import Plotter
from models.model import Model
from trainers.fedavg import FedAvg
from models.eeg import EEG_CNN

n_rounds = 40
n_local_epochs = 1

n_clients = 1
n_clients_round = int(n_clients * 1.)

lr = 0.01
batch_size = 32
# test_batch_size = 100

test_every = 1
save_every = 10

smpc = False

output_folder = "output"

torch.manual_seed(42)
np.random.seed(42)


def main():
    # model = EEG_CNN()
    # model.forward(torch.zeros((1, 3, 224, 224)))
    # Create clients
    hook = sy.TorchHook(torch)
    clients = [sy.VirtualWorker(hook, id=f"client{i}") for i in range(n_clients)]
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    # Load data
    data_loader = DataLoader("sleep", clients)  # ["mnist", "sleep"]
    data_loader.send_data_to_clients()

    # Load model
    model = Model("EEG_CNN", clients)  # ["MNIST_CNN", "EEG_CNN"]
    if smpc:
        model.send_model_to_clients()

    # Train model
    save_folder = os.path.join(output_folder, "model")
    trainer = FedAvg(model, data_loader, crypto_provider, save_folder)
    trainer.train(n_rounds, n_local_epochs, n_clients_round, lr, batch_size, test_every, save_every, smpc=smpc)

    # Plot results
    list_test_loss_client = trainer.list_test_loss_client
    list_train_loss_client = trainer.list_train_loss_client
    list_accuracy_client = trainer.list_accuracy_client
    print(list_accuracy_client)
    list_test_rounds = trainer.list_test_rounds

    # list_test_loss_client = [[4, 2, 1, 0.5, 0.25]] * n_clients
    # list_test_rounds = list(range(0, n_rounds*2, test_every))
    plotter = Plotter(output_folder)

    # Loss learning curve
    plotter.plot_learning_curve_avg(list_test_rounds, list_test_loss_client, list_train_loss_client)
    plotter.plot_learning_curve_clients(list_test_rounds, list_test_loss_client, list_train_loss_client,
                                        n_clients=n_clients)

    # Accuracy learning curve
    plotter.plot_learning_curve_avg(list_test_rounds, list_accuracy_client, label="accuracy", filename="accuracy-avg")
    plotter.plot_learning_curve_clients(list_test_rounds, list_accuracy_client,
                                        n_clients=n_clients, label="accuracy", filename="accuracy-clients")


if __name__ == "__main__":
    main()
