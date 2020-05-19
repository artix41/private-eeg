import numpy as np
import matplotlib.pyplot as plt
import os
import math


class Plotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def plot_learning_curve_clients(self, list_rounds, list_test_curve_client, list_train_curve_client=None,
                                    n_clients=None, label="loss", filename="loss-clients"):
        output_file = os.path.join(self.output_folder, f"{filename}.png")

        if n_clients is None:
            n_clients = len(list_test_curve_client)

        n_rows = int(math.sqrt(n_clients))
        n_cols = math.ceil(n_clients / n_rows)

        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                i_client = i_row * n_cols + i_col
                if i_client <= n_clients:
                    ax = fig.add_subplot(n_rows, n_cols, i_client+1)

                    if list_train_curve_client is not None:
                        ax.plot(list_rounds, list_train_curve_client[i_client], label=f"Train {label}")
                    ax.plot(list_rounds, list_test_curve_client[i_client], label=f"Test {label}")
                    ax.set_xlabel("Round")
                    ax.set_ylabel(label.capitalize())
                    ax.legend()

        plt.tight_layout()
        print(f"Saving {output_file}")
        plt.savefig(output_file, tight_layout=True)

    def plot_learning_curve_avg(self, list_rounds, list_test_curve_client,
                                list_train_curve_client=None, label="loss", filename="loss-avg"):
        fig, ax = plt.subplots(figsize=(8, 5))
        output_file = os.path.join(self.output_folder, f"{filename}.png")

        if list_train_curve_client is not None:
            ax.plot(list_rounds, np.mean(list_train_curve_client, axis=0), label=f"Train {label}")
        ax.plot(list_rounds, np.mean(list_test_curve_client, axis=0), label=f"Test {label}")
        ax.set_xlabel("Round")
        ax.set_ylabel(label.capitalize())
        ax.legend()

        print(f"Saving {output_file}")
        plt.savefig(output_file, tight_layout=True)
