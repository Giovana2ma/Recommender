import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import sys


class Rocchio:

    def __init__(self, content, ratings, weight, alpha, beta, gamma):
        self.content = content
        self.ratings = ratings
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def tfidf(self):
        """
        Apply TF-IDF to the content.
        """
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(self.content)

    def compute_user_representations(self):
        """
        Compute the user representations.
        """
        self.user_vectors = {}
        for user_id in self.ratings["UserId"].unique():
            user_data = self.ratings[self.ratings["UserId"] == user_id]
            ratings = csr_matrix(user_data["Rating"].values.reshape(-1, 1))
            item_indices = user_data["ItemId"]

            terms = self.tfidf_matrix[item_indices].T
            weighted_vectors = terms @ ratings

            user_vector = weighted_vectors / len(item_indices)
            self.user_vectors[user_id] = user_vector

        return

    def predict(self, user_id, item_index):
        """
        Predict the rating of a user for an item.
        """
        user_vector = self.user_vectors[user_id].toarray().flatten()
        item_vector = self.tfidf_matrix[item_index].toarray().flatten()

        similarity_score = cosine_similarity(
            user_vector.reshape(1, -1), item_vector.reshape(1, -1)
        )[0][0]

        weight = (
            float(self.weight[item_index].replace(",", ""))
            if self.weight[item_index] != "N/A"
            else 3.0
        )

        pred_scaled = similarity_score * weight

        return pred_scaled

    def generate_predictions(self, data):
        """
        Compute the predictions for the data.
        """
        predictions = np.array(
            [
                self.predict(user, item)
                for user, item in zip(data["UserId"], data["ItemId"])
            ]
        )
        return predictions

    def update_user_profiles(self, user_feedback):
        """
        Update the user profiles using Rocchio's algorithm.
        """
        for user_id, feedback in user_feedback.items():
            relevant_items = feedback["relevant"]
            non_relevant_items = feedback["non_relevant"]
            terms = self.tfidf_matrix.T

            # Relevant items' vectors
            relevant_vectors = terms[:, relevant_items].sum(axis=1)
            # Non-relevant items' vectors
            non_relevant_vectors = terms[:, non_relevant_items].sum(axis=1)

            # Update user vector using Rocchio's formula
            self.user_vectors[user_id] = (
                self.alpha * self.user_vectors[user_id]
                + self.beta * relevant_vectors
                - self.gamma * non_relevant_vectors
            )
