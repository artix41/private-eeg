import numpy as np
import torch
from torch import optim
import math
import os


def batch_list(l, batch_size):
    return [l[i*batch_size:(i+1)*batch_size] for i in range(int(math.ceil(len(l) / batch_size)))]


class FedAvg:
    def __init__(self, model, data_loader, crypto_provider, output_folder):
        self.model = model
        self.data_loader = data_loader
        self.clients = model.clients
        self.n_clients = len(self.clients)
        self.crypto_provider = crypto_provider

        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.n_samples_train_per_client = self.data_loader.n_samples_train_per_client
        self.n_samples_test_per_client = self.data_loader.n_samples_test_per_client

        self.loss_fn = self.model.get_loss_fn()

        self.list_test_loss_client = [[]] * self.n_clients
        self.list_test_rounds = []

    def train(self, n_rounds, n_local_epochs, n_clients_round, lr=0.1, batch_size=16, test_every=1, save_every=2):
        opt_client = [optim.SGD(params=self.model.model_client[i].parameters(), lr=lr) for i in range(self.n_clients)]
        self.list_test_loss_client = [[]] * self.n_clients
        self.list_test_rounds = []

        for i_round in range(n_rounds):
            print(f"================== Round {i_round} ==================")
            sample_indices = list(np.random.permutation(range(self.n_samples_train_per_client)))
            list_clients_round = np.random.choice(range(self.n_clients), n_clients_round, replace=False)

            for i_client, client_index in enumerate(list_clients_round):
                print(f"\nClient {i_client + 1}/{n_clients_round}")
                loss_client = 0
                i_iter = 0
                model_client = self.model.get_model_client(client_index)
                X_train_ptr, Y_train_ptr = self.data_loader.get_training_data(client_index)

                # Update client
                for i_local_epoch in range(n_local_epochs):  # n_local_epochs=E
                    list_batch_indices = batch_list(sample_indices, batch_size)
                    for i_batch, batch_indices in enumerate(list_batch_indices):
                        batch_indices_tensor = torch.tensor(batch_indices).send(self.clients[client_index])
                        X_ptr = X_train_ptr[batch_indices, :, :]
                        Y_ptr = Y_train_ptr.gather(0, batch_indices_tensor)

                        pred = model_client(X_ptr)
                        loss = self.loss_fn(pred, Y_ptr)

                        opt_client[client_index].zero_grad()
                        loss.backward()
                        opt_client[client_index].step()

                        loss_client += loss.get().data.numpy()
                        i_iter += 1

                        print(f"Batch {i_batch} / {self.n_samples_train_per_client // batch_size}: "
                              f"{loss_client / i_iter:.4f}", end="\r")

                        del batch_indices_tensor
                print("")

            # Update global model
            print("Update...")
            with torch.no_grad():
                new_param = [0.] * len(list(self.model.model_client[0].parameters()))
                for i_param, param_client in enumerate(zip(*(self.model.model_client[i].parameters()
                                                             for i in range(self.n_clients)))):
                    print(f"{i_param} ", end="\r")
                    for i_client in list_clients_round:
                        new_param[i_param] += param_client[i_client].copy().get().fix_precision().share(*self.clients,
                                                                                                        crypto_provider=self.crypto_provider)
                    new_param[i_param] = new_param[i_param].get().float_precision() / n_clients_round

                for i_client in range(self.n_clients):
                    model_client = self.model.get_model_client(i_client)
                    model_client.get()
                    for i_param, param in enumerate(model_client.parameters()):
                        param.set_(torch.tensor(new_param[i_param]))
                    model_client.send(self.clients[i_client])

            if i_round % test_every == 0:
                cur_test_loss_client = self.test()
                self.list_test_loss_client = np.hstack([self.list_test_loss_client, cur_test_loss_client])
                self.list_test_rounds.append(i_round)

            if i_round != 0 and i_round % save_every == 0:
                self.save()

    def test(self):
        test_loss_client = []
        for i_client in range(self.n_clients):
            X_test_ptr, Y_test_ptr = self.data_loader.get_test_data(i_client)
            model_client = self.model.get_model_client(i_client)
            pred = model_client(X_test_ptr)
            test_loss = self.loss_fn(pred, Y_test_ptr).get().data.numpy()
            test_loss_client.append([test_loss])

        return test_loss_client

    def save(self):
        output_file_loss = os.path.join(self.output_folder, "test_loss.npy")
        output_file_model = os.path.join(self.output_folder, "model.pt")

        print(f"Saving {output_file_loss}")
        np.save(output_file_loss, self.list_test_loss_client)

        print(f"Saving {output_file_model}")
        model = self.model.get_model_client(0)
        torch.save(model.state_dict(), output_file_model)
