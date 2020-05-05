import numpy as np
import torch


class FedAvg:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.clients = model.clients
        self.n_clients = len(self.clients)

        self.n_samples_train = self.data_loader.n_samples_train
        self.n_samples_train_per_client = self.n_samples_train // self.n_clients

        self.n_samples_test = self.data_loader.n_samples_test
        self.n_samples_test_per_client = self.n_samples_test // self.n_clients

    def train(self, n_rounds, n_local_epochs):
        for i_round in range(n_rounds):
            sample_indices = list(np.random.permutation(range(self.n_samples_train_per_client)))
            list_workers_round = np.random.choice(range(self.n_clients), n_clients_round, replace=False)

            for i_worker, worker_index in enumerate(list_workers_round):
                loss_worker = 0
                i_iter = 0

                # Update client
                for i_local_epoch in range(n_local_epochs):  # n_local_epochs=E
                    list_batch_indices = batch_list(sample_indices, batch_size)
                    for i_batch, batch_indices in enumerate(list_batch_indices):
                        batch_indices_tensor = torch.tensor(batch_indices).send(workers[worker_index])
                        X_ptr = X_train_ptr_worker[worker_index][batch_indices, :, :]
                        Y_ptr = Y_train_ptr_worker[worker_index].gather(0, batch_indices_tensor)

                        pred = model_worker[worker_index](X_ptr)
                        loss = loss_fn(pred, Y_ptr)

                        opt_worker[worker_index].zero_grad()
                        loss.backward()
                        opt_worker[worker_index].step()

                        #                 model_worker[worker_index] = model_worker[worker_index].get()
                        loss_worker += loss.get().data.numpy()
                        #                 model_worker[worker_index].get()
                        i_iter += 1

                        clear_output(wait=True)
                        print(f"Round {i_round + 1}/{n_rounds}")
                        print(f"Worker {i_worker + 1}/{n_workers_round}")
                        print(
                            f"Batch {i_batch} / {len(mnist_train.data) // n_workers // batch_size}: {loss_worker / (i_iter):.4f}")
                        if i_round >= 1:
                            print(f"Test loss {np.mean(list_loss_test_worker, axis=0)[-1]:.4f}")
                        del batch_indices_tensor

            # Update global model
            print("Update...")
            with torch.no_grad():
                new_param = [0.] * len(list(model_worker[0].parameters()))
                for i_param, param_worker in enumerate(zip(*(model_worker[i].parameters() for i in range(n_workers)))):
                    print(f"{i_param} ", end="\r")
                    for i_worker in list_workers_round:
                        new_param[i_param] += param_worker[i_worker].copy().get().fix_precision().share(*workers,
                                                                                                        crypto_provider=crypto_provider)
                    new_param[i_param] = new_param[i_param].get().float_precision() / n_workers_round

                for i_worker in range(n_workers):
                    model_worker[i_worker].get()
                    for i_param, param in enumerate(model_worker[i_worker].parameters()):
                        param.set_(torch.tensor(new_param[i_param]))
                    model_worker[i_worker].send(workers[i_worker])