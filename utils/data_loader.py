from torchvision import transforms
import torchvision.datasets as datasets
import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform


def get_mnist():
    mnist_train = datasets.MNIST(root='../data', train=True, download=True,
                                 transform=None)
    mnist_test = datasets.MNIST(root='../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
    return mnist_train, mnist_test


def transform_mnist(X_mnist):
    return X_mnist.view(-1, 28, 28).float() / 255


class DataLoader:
    def __init__(self, name, clients):
        self.name = name
        self.clients = clients
        self.n_clients = len(clients)

        # List of pointers to each client's X_train
        self.X_train_ptr_client = []
        self.Y_train_ptr_client = []
        self.X_test_ptr_client = []
        self.Y_test_ptr_client = []

        self.load_data_method = {"mnist_iid": self.load_mnist_iid,
                                 "sleep": self.load_sleep_data}

    def send_data_to_clients(self):
        self.load_data_method[self.name]()

    def get_training_data(self, client):
        if len(self.X_train_ptr_client) == 0:
            raise ValueError("You haven't sent the data to the clients. Use data_loader.send_data_to_clients()")

        return self.X_train_ptr_client[client], self.Y_train_ptr_client[client]

    def get_test_data(self, client):
        if len(self.X_train_ptr_client) == 0:
            raise ValueError("You haven't sent the data to the clients. Use data_loader.send_data_to_clients()")

        return self.X_test_ptr_client[client], self.Y_test_ptr_client[client]

    def load_mnist_iid(self):
        mnist_train, mnist_test = get_mnist()

        n_samples_train_per_client = len(mnist_train.targets) // self.n_clients
        n_samples_test_per_client = len(mnist_test.targets) // self.n_clients

        self.X_train_ptr_client = [
            transform_mnist(
                mnist_train.data[n_samples_train_per_client * i:n_samples_train_per_client * (i + 1)]
            ).send(self.clients[i]) for i in range(self.n_clients)]

        self.Y_train_ptr_client = [
            mnist_train.targets[n_samples_train_per_client * i:n_samples_train_per_client * (i + 1)].send(self.clients[i])
            for i in range(self.n_clients)]

        self.X_test_ptr_client = [
            transform_mnist(
                mnist_test.data[n_samples_test_per_client * i:n_samples_test_per_client * (i + 1)]
            ).send(self.clients[i]) for i in range(self.n_clients)]

        self.Y_test_ptr_client = [
            mnist_test.targets[n_samples_test_per_client * i:n_samples_test_per_client * (i + 1)].send(self.clients[i])
            for i in range(self.n_clients)]

    def load_sleep_data(self):
        max_samples = 1000
        proportion_train = 0.75

        data_folder = "../data/SleepEDF/"

        dict_labels = {"W": 0,
                       "1": 1,
                       "2": 2,
                       "3": 3,
                       "4": 4,
                       "M": 5,
                       "R": 6}

        # =========== Get all filenames and associated labels ===========

        print("Load images...")
        list_imfiles_for_subject = [[] for _ in range(20)]
        list_labels_for_subject = [[] for _ in range(20)]
        for subject in range(1, 21):
            n_nights = 2 if subject < 20 else 1
            for night in range(1, n_nights + 1):
                folder = os.path.join(data_folder, f"sub{subject}_n{night}_img_fpz")

                list_imfiles = glob.glob(os.path.join(folder, "*.png"))
                list_imfiles.remove(os.path.join(folder, "img_whole_night.png"))

                with open(os.path.join(folder, "labels.txt")) as f:
                    list_labels = f.read().splitlines()

                sorted_list_labels = []
                for imfile in list_imfiles:
                    img_number = int(imfile.split("_")[-1].split(".")[0])
                    sorted_list_labels.append(dict_labels[list_labels[img_number - 1]])

                list_labels_for_subject[subject - 1] += sorted_list_labels
                list_imfiles_for_subject[subject - 1] += list_imfiles

            list_labels_for_subject[subject - 1] = np.array(list_labels_for_subject[subject - 1])
            list_imfiles_for_subject[subject - 1] = np.array(list_imfiles_for_subject[subject - 1])

            # Randomized the two nights
            permutation_indices = np.random.permutation(len(list_imfiles_for_subject[subject - 1]))
            list_labels_for_subject[subject-1] = list_labels_for_subject[subject-1][permutation_indices]
            list_imfiles_for_subject[subject - 1] = list_imfiles_for_subject[subject - 1][permutation_indices]

        # =========== Transform it into a dataset ===========

        X_train_client = [[] for _ in range(self.n_clients)]
        Y_train_client = [[] for _ in range(self.n_clients)]
        X_test_client = [[] for _ in range(self.n_clients)]
        Y_test_client = [[] for _ in range(self.n_clients)]

        for subject in range(self.n_clients):
            n_samples = min(max_samples, len(list_labels_for_subject[subject]))
            n_samples_train = int(proportion_train * n_samples)
            for i_image in range(n_samples):
                image = plt.imread(list_imfiles_for_subject[subject][i_image])
                if not (image.shape[0] == 224 and image.shape[1] == 224):
                    image = skimage.transform.resize(image, (224, 224), preserve_range=True)
                image = image.reshape(3, 224, 224)

                if i_image < n_samples_train:
                    X_train_client[subject].append(image)
                    Y_train_client[subject].append(list_labels_for_subject[subject][i_image])
                else:
                    X_test_client[subject].append(image)
                    Y_test_client[subject].append(list_labels_for_subject[subject][i_image])

        print("Convert to tensors...")
        X_train_client = torch.tensor(np.array(X_train_client))
        X_test_client = torch.tensor(np.array(X_test_client))
        Y_train_client = torch.tensor(np.array(Y_train_client), dtype=torch.long)
        Y_test_client = torch.tensor(np.array(Y_test_client), dtype=torch.long)

        # print("Transform dataset to tensors...")
        # X_train_client = torch.Tensor(X_train_client)
        # Y_train_client = torch.Tensor(Y_train_client)
        # X_test_client = torch.Tensor(X_test_client)
        # Y_test_client = torch.Tensor(Y_test_client)

        # =========== Send dataset to clients ===========

        print("Send to clients...")
        self.X_train_ptr_client = [X_train_client[i].send(self.clients[i]) for i in range(self.n_clients)]
        self.Y_train_ptr_client = [Y_train_client[i].send(self.clients[i]) for i in range(self.n_clients)]
        self.X_test_ptr_client = [X_test_client[i].send(self.clients[i]) for i in range(self.n_clients)]
        self.Y_test_ptr_client = [Y_test_client[i].send(self.clients[i]) for i in range(self.n_clients)]


