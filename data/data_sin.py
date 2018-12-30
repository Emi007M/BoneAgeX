from data.IData import Data
import numpy as np


class DataSin(Data):

    def __init__(self, x=None, y=None):
        super().__init__(x, y)

    def get_all_data(self, amount):
        return self.__get_data(amount)

    def get_random_data(self, batch_size):
        n_samples = self.data.x.shape[0]
        indices = np.random.choice(n_samples, batch_size)
        x_batch, y_batch = self.data.x[indices], self.data.y[indices]
        return Data(x_batch, y_batch)

    def __get_data(self, amount):
        x_data = np.arange(amount * 0.1, step=.1)

        y_data = x_data + 20 * np.sin(x_data / 10)

        x_data = np.reshape(x_data, (amount, 1))
        y_data = np.reshape(y_data, (amount, 1))

        self.data = Data(x_data, y_data)
        return self.data



