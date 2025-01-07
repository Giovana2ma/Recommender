import random
random.seed(42)
from utils import *
from funksvd import *
import sys
from rocchio import *


def main():
    ratings_file = sys.argv[1]
    content_file = sys.argv[2]
    targets_file = sys.argv[3]

    target, users, items, ratings, targets, num_users, num_items, contents = (
        load_and_index_datasets(ratings_file, targets_file, content_file)
    )

    reverse_users = {v: k for k, v in users.items()}
    reverse_items = {v: k for k, v in items.items()}

    n_factors = 10
    n_epochs = 30
    lr = 0.005
    alpha = 0.1

    content = combine_textual_info(contents)
    rocchio = Rocchio(content, ratings, contents["imdbVotes"])
    rocchio.tfidf()
    rocchio.compute_user_representations()

    # svd = Funksvd(
    #     ratings,
    #     contents["imdbVotes"],
    #     num_users,
    #     num_items,
    #     n_factors,
    #     n_epochs,
    #     lr,
    #     alpha,
    # )
    # svd.run_sgd()

    # user_ratings = {}
    # for user in users:
    #     user_ratings[user] = len(ratings[ratings["UserId"] == user])

    targets["Rocchio"] = targets.apply(
        lambda x: rocchio.predict(x["UserId"], x["ItemId"]), axis=1
    )

    # targets["FunkSVD"] = targets.apply(
    #     lambda x: svd.weighted_prediction(x["UserId"], x["ItemId"]), axis=1
    # )

    targets["UserId"] = targets["UserId"].map(reverse_users)
    targets["ItemId"] = targets["ItemId"].map(reverse_items)

    targets = targets.sort_values(["UserId", "Rocchio"])

    targets[["UserId", "ItemId"]].to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
