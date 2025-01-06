import numpy as np
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

    n_factors = 10
    n_epochs = 30
    lr = 0.005
    alpha = 0.1

    content = combine_textual_info(contents)
    recomender = Rocchio(content, ratings, contents["imdbVotes"])
    recomender.tfidf()
    recomender.compute_user_representations()

    model = Funksvd(
        ratings,
        contents["imdbVotes"],
        num_users,
        num_items,
        n_factors,
        n_epochs,
        lr,
        alpha,
    )
    model.run_sgd()

    predictions = model.generate_predictions(targets)
    print(predictions)
    target["Rating"] = predictions

    target = ranking(target, users, items)
    target = target.drop("Rating", axis=1)

    target.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
