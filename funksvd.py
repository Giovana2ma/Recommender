import numpy as np
from utils import *


class Funksvd:
    def __init__(
        self, ratings, weight, num_users, num_items, n_factors, n_epochs, lr, alpha
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.alpha = alpha  # Regularization parameter

        self.ratings = ratings
        self.weight = weight
        self.num_users = num_users
        self.num_items = num_items

        self.global_mean = ratings["Rating"].mean()
        self.bu = None  # User bias vector                   (U,1)
        self.bi = None  # Item bias vector                   (I,1)
        self.p = None  # User latent factor matrix           (U,F)
        self.q = None  # Item latent factor matrix           (I,F)

    def shuffle(self):
        """
        Shuffle the ratings DataFrame.
        """
        self.ratings = self.ratings.sample(frac=1).reset_index(drop=True)

    def initialization(self):
        """
        Initialize bias vectors with zeros and latent factor matrices based on a
        normal distribution with mean 0 and standard deviation 0.1.
        """
        self.bu = np.zeros(self.num_users)  # (U,1)
        self.bi = np.zeros(self.num_items)  # (I,1)

        self.p = np.random.normal(0, 0.1, (self.num_users, self.n_factors))  # (U,F)
        self.q = np.random.normal(0, 0.1, (self.num_items, self.n_factors))  # (I,F)

    def predict(self, user, item):
        """
        Predict the rating of user for item.
        """
        pred = self.global_mean
        pred += self.bu[user] + self.bi[item]
        pred += np.dot(self.p[user], self.q[item])

        return pred

    def run_iteration_(self):
        """
        Run a single iteration of SGD.
        """
        for user, item, rating in zip(
            self.ratings["UserId"], self.ratings["ItemId"], self.ratings["Rating"]
        ):

            pred = self.predict(user, item)
            err = rating - pred

            # Atualização das matrizes de viés
            self.bu[user] += self.lr * (err - self.alpha * self.bu[user])
            self.bi[item] += self.lr * (err - self.alpha * self.bi[item])

            puf = self.p[user]
            qif = self.q[item]

            # Atualização das matrizes de fatores latentes
            self.p[user] += self.lr * (err * qif - self.alpha * puf)
            self.q[item] += self.lr * (err * puf - self.alpha * qif)

    def run_sgd(self):
        """
        Run the optimization process.
        """
        self.initialization()
        for _ in range(self.n_epochs):
            self.shuffle()
            self.run_iteration_()

    def weighted_prediction(self, user, item):
        """
        Predict the rating of user for item.
        """
        pred = self.global_mean
        pred += self.bu[user] + self.bi[item]
        pred += np.dot(self.p[user], self.q[item])
        weight = (
            float(self.weight[item].replace(",", ""))
            if self.weight[item] != "N/A"
            else 3.0
        )

        return pred * weight

    def generate_predictions(self, data):
        """
        Predict the ratings for the data.
        """
        predictions = np.array(
            [
                self.weighted_prediction(user, item)
                for user, item in zip(data["UserId"], data["ItemId"])
            ]
        )
        return predictions

    def get_initial_predictions(self, targets):
        """
        Get the initial predictions for the targets.
        """
        predictions = {}
        for user in targets["UserId"]:
            for item in targets["ItemId"]:
                print(user, item)
                pred = self.predict(user, item)
                predictions[user].append((item, pred))
        return predictions
