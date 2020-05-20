import numpy as np
import torch
from torch import optim
import math
import os
from syft.federated.floptimizer import Optims


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

        self.loss_fn = self.model.get_loss_fn()

        self.list_test_loss_client = [[]] * self.n_clients
        self.list_accuracy_client = [[]] * self.n_clients
        self.list_train_loss_client = [[]] * self.n_clients
        self.list_test_rounds = []

    def train(self, n_rounds, n_local_epochs, n_clients_round, lr=0.1, batch_size=16, test_every=1, save_every=2, smpc=False):
        # opt_client = [optim.Adam(params=self.model.model_client[i].parameters(), lr=lr) for i in range(self.n_clients)]
        opt_client = [Optims([self.clients[i].id], optim=optim.Adam(params=self.model.model_client[i].parameters(), lr=lr))
                      for i in range(self.n_clients)]
        self.list_accuracy_client = [[]] * self.n_clients
        self.list_train_loss_client = [[]] * self.n_clients
        self.list_test_rounds = []

        for i_round in range(n_rounds):
            print(f"================== Round {i_round} ==================")
            list_clients_round = np.random.choice(range(self.n_clients), n_clients_round, replace=False)

            train_loss_client = []
            for i_client, client_index in enumerate(list_clients_round):
                print(f"\nClient {i_client + 1}/{n_clients_round}")
                loss_client = 0
                i_iter = 0
                model_client = self.model.get_model_client(client_index)
                X_train_ptr, Y_train_ptr = self.data_loader.get_training_data(client_index)
                n_samples_train = len(X_train_ptr)
                sample_indices = list(np.random.permutation(range(n_samples_train)))

                # Update client
                for i_local_epoch in range(n_local_epochs):  # n_local_epochs=E
                    list_batch_indices = batch_list(sample_indices, batch_size)
                    for i_batch, batch_indices in enumerate(list_batch_indices):
                        batch_indices_tensor = torch.tensor(batch_indices).send(self.clients[client_index])
                        X_ptr = X_train_ptr[batch_indices, :, :]
                        Y_ptr = Y_train_ptr.gather(0, batch_indices_tensor)
                        if not smpc:
                            model_client = model_client.send(self.clients[client_index])

                        pred = model_client(X_ptr)
                        loss = self.loss_fn(pred, Y_ptr)

                        opt = opt_client[client_index].get_optim(self.clients[client_index].id)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        loss_client += loss.get().data.numpy()
                        i_iter += 1

                        if not smpc:
                            model_client = model_client.get()

                        print(f"Batch {i_batch} / {n_samples_train // batch_size}: "
                              f"{loss_client / i_iter:.4f}", end="\r")

                        del batch_indices_tensor

                train_loss_client.append([loss_client / i_iter])
                print("")

            print()
            self.list_train_loss_client = np.hstack([self.list_train_loss_client, train_loss_client])

            # Update models
            with torch.no_grad():
                new_param = [0.] * len(list(self.model.model_client[0].parameters()))
                for i_param, param_client in enumerate(zip(*(self.model.model_client[i].parameters()
                                                             for i in range(self.n_clients)))):
                    print(f"{i_param} ", end="\r")

                    if smpc:
                        for i_client in list_clients_round:
                            new_param[i_param] += param_client[i_client].copy().get().fix_precision().share(*self.clients,
                                                                                                            crypto_provider=self.crypto_provider)
                        new_param[i_param] = new_param[i_param].get().float_precision() / n_clients_round
                    else:
                        new_param = torch.mean(torch.stack(param_client), axis=0)
                        for param in param_client:
                            param.set_(new_param)

                if smpc:
                    for i_client in range(self.n_clients):
                        model_client = self.model.get_model_client(i_client).get()
                        for i_param, param in enumerate(model_client.parameters()):
                            param.set_(torch.tensor(new_param[i_param]))
                        model_client.send(self.clients[i_client])

            if i_round % test_every == 0:
                print("Testing...")
                cur_test_loss_client, cur_accuracy_client = self.test(smpc=smpc)
                print(f"Test loss: {np.mean(cur_test_loss_client):.4f}")
                print(f"Accuracy: {np.mean(cur_accuracy_client)*100:.1f}%")
                self.list_test_loss_client = np.hstack([self.list_test_loss_client, cur_test_loss_client])
                self.list_accuracy_client = np.hstack([self.list_accuracy_client, cur_accuracy_client])
                self.list_test_rounds.append(i_round)
                print("")

                # print("Test on train set")
                # cur_train_loss_client = self.test(smpc=smpc, on_training=True)
                # self.list_train_loss_client = np.hstack([self.list_train_loss_client, cur_train_loss_client])

            if i_round != 0 and i_round % save_every == 0 or i_round == n_rounds - 1:
                self.save()

    def test(self, on_training=False, smpc=False, test_batch_size=50):
        test_loss_client, accuracy_client = [], []
        for i_client in range(self.n_clients):
            if on_training:
                X_test_ptr, Y_test_ptr = self.data_loader.get_training_data(i_client)
            else:
                X_test_ptr, Y_test_ptr = self.data_loader.get_test_data(i_client)
            model_client = self.model.get_model_client(i_client)

            if not smpc:
                model_client = model_client.send(self.clients[i_client])

            n_batches = int(np.ceil(len(X_test_ptr)/test_batch_size))
            test_loss = 0
            correct = 0
            for i_batch in range(n_batches):
                first_i = test_batch_size*i_batch
                last_i = test_batch_size*(i_batch+1)
                pred = model_client(X_test_ptr[first_i:last_i])
                test_loss += self.loss_fn(pred, Y_test_ptr[first_i:last_i]).get().data.numpy()

                pred_argmax = pred.argmax(1, keepdim=True)
                # print(Y_test_ptr[first_i:last_i].view_as(pred_argmax).get())
                # print(pred_argmax.get())
                correct += pred_argmax.eq(Y_test_ptr[first_i:last_i].view_as(pred_argmax)).sum().get().data.numpy()
            test_accuracy = correct / len(X_test_ptr)
            test_loss /= n_batches

            test_loss_client.append([test_loss])
            accuracy_client.append([test_accuracy])

            if not smpc:
                model_client.get()

        return test_loss_client, accuracy_client

    def save(self):
        output_file_test_loss = os.path.join(self.output_folder, "test_loss.npy")
        output_file_train_loss = os.path.join(self.output_folder, "train_loss.npy")
        output_file_test_accuracy = os.path.join(self.output_folder, "test_accuracy.npy")
        output_file_model = os.path.join(self.output_folder, "model.pt")

        print(f"Saving {output_file_test_loss}")
        np.save(output_file_test_loss, self.list_test_loss_client)

        print(f"Saving {output_file_train_loss}")
        np.save(output_file_train_loss, self.list_train_loss_client)

        print(f"Saving {output_file_test_accuracy}")
        np.save(output_file_test_accuracy, self.list_accuracy_client)

        print(f"Saving {output_file_model}")
        model = self.model.get_model_client(0)
        torch.save(model.state_dict(), output_file_model)
