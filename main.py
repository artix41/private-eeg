import torch
import syft as sy
import os
import numpy as np
import argparse

from utils.data_loader import DataLoader
from utils.plotter import Plotter
from models.model import Model
from trainers.fedavg import FedAvg
from models.eeg import EEG_CNN

torch.manual_seed(42)
np.random.seed(42)


def main():
    # ====================== Parameters ======================
    name_subtask = ""

    test_every = 1
    save_every = 5

    smpc = False
    output_folder = "output"

    # ====================== User inputs =====================

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Name of the dataset",
                        type=str, default="sleep", choices=["sleep", "mnist"])
    parser.add_argument("-c", "--clients", help="Number of clients",
                        type=int, default=2)
    parser.add_argument("-s", "--samples", help="Number of samples per clients",
                        type=int, default=1000)
    parser.add_argument("-k", help="Number of clients per round",
                        type=int, default=2)
    parser.add_argument("-r", "--rounds", help="Number of rounds",
                        type=int, default=20)
    parser.add_argument("-e", "--epochs", help="Number of local epochs (client epochs)",
                        type=int, default=1)
    parser.add_argument("-b", help="Batch size",
                        type=int, default=32)
    parser.add_argument("--lr", help="Learning rate",
                        type=float, default=0.01)

    args = parser.parse_args()

    problem_name = args.data
    n_rounds = args.rounds
    n_local_epochs = args.epochs

    n_clients = args.clients
    n_clients_round = args.k
    max_samples = args.samples

    lr = args.lr
    batch_size = args.b

    subtask_folder = os.path.join(output_folder, f"{n_clients}-clients", name_subtask)

    # ================== Create clients ======================

    hook = sy.TorchHook(torch)
    clients = [sy.VirtualWorker(hook, id=f"client{i}") for i in range(n_clients)]
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    # ===================== Load data =======================

    data_loader = DataLoader(problem_name, clients, max_samples_per_client=max_samples)
    data_loader.send_data_to_clients()

    #  ==================== Load model ======================

    model = Model("EEG_CNN", clients)  # ["MNIST_CNN", "EEG_CNN"]
    if smpc:
        model.send_model_to_clients()

    #  ==================== Train model =====================

    save_folder = os.path.join(subtask_folder, "model")
    trainer = FedAvg(model, data_loader, crypto_provider, save_folder)
    trainer.train(n_rounds, n_local_epochs, n_clients_round, lr, batch_size, test_every, save_every, smpc=smpc)

    #  =================== Plot results ======================

    list_test_loss_client = trainer.list_test_loss_client
    list_train_loss_client = trainer.list_train_loss_client
    list_accuracy_client = trainer.list_accuracy_client
    list_test_rounds = trainer.list_test_rounds

    # list_test_loss_client = [[4, 2, 1, 0.5, 0.25]] * n_clients
    # list_test_rounds = list(range(0, n_rounds*2, test_every))
    plotter = Plotter(subtask_folder)

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
