from torchvision import transforms
import torchvision.datasets as datasets


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

        self.n_samples_train_per_client = -1
        self.n_samples_test_per_client = -1

        self.load_data_method = {"mnist_iid": self.load_mnist_iid}

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

        self.n_samples_train_per_client = n_samples_train_per_client
        self.n_samples_test_per_client = n_samples_test_per_client

    # def load_mnist_non_iid(self, n_digits_client=2):
    #     mnist_train, mnist_test = get_mnist()
    #
    #     n_samples_train_per_client = len(mnist_train.targets) // self.n_clients
    #     n_samples_test_per_client = len(mnist_test.targets) // self.n_clients
    #
    #     self.X_train_ptr_client = [
    #         transform_mnist(
    #             mnist_train.data[n_samples_train_per_client * i:n_samples_train_per_client * (i + 1)]
    #         ).send(self.clients[i]) for i in range(self.n_clients)]
    #
    #     self.Y_train_ptr_client = [
    #         mnist_train.targets[n_samples_train_per_client * i:n_samples_train_per_client * (i + 1)].send(self.clients[i])
    #         for i in range(self.n_clients)]
    #
    #     self.X_test_ptr_client = [
    #         transform_mnist(
    #             mnist_test.data[n_samples_test_per_client * i:n_samples_test_per_client * (i + 1)]
    #         ).send(self.clients[i]) for i in range(self.n_clients)]
    #
    #     self.Y_test_ptr_client = [
    #         mnist_test.targets[n_samples_test_per_client * i:n_samples_test_per_client * (i + 1)].send(self.clients[i])
    #         for i in range(self.n_clients)]
