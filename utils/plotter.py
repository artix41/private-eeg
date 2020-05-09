import numpy as np
import matplotlib.pyplot as plt
import os
import math


class Plotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def plot_learning_curve_clients(self, list_rounds, list_loss_client, n_clients=None):
        output_file = os.path.join(self.output_folder, "learning-curve-clients.png")

        if n_clients is None:
            n_clients = len(list_loss_client)

        n_rows = int(math.sqrt(n_clients))
        n_cols = math.ceil(n_clients / n_rows)

        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                i_client = i_row * n_cols + i_col + 1
                if i_client <= n_clients:
                    # ax.subplot(n_rows, n_cols, i_client)
                    ax = fig.add_subplot(n_rows, n_cols, i_client)
                    ax.plot(list_rounds, list_loss_client[i_client], label="Test loss")
                    ax.set_xlabel("Round")
                    ax.set_ylabel("Loss")

        plt.tight_layout()
        print(f"Saving {output_file}")
        plt.savefig(output_file, tight_layout=True)

    def plot_learning_curve_avg(self, list_rounds, list_loss_client):
        plt.rcParams["figure.figsize"] = (8, 5)
        output_file = os.path.join(self.output_folder, "learning-curve-avg.png")

        plt.plot(list_rounds, np.mean(list_loss_client, axis=0))
        plt.xlabel("Round")
        plt.ylabel("Loss")

        print(f"Saving {output_file}")
        plt.savefig(output_file, tight_layout=True)
