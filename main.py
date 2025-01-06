import numpy as np
from utils import *
from funksvd import *
import sys
from rocchio import *


def main():

    # Leitura dos arquivos de entrada
    ratings_file = sys.argv[1]
    content_file = sys.argv[2]
    targets_file = sys.argv[3]

    target, users, items, ratings, targets, num_users, num_items, contents = (
        load_and_index_datasets(ratings_file, targets_file, content_file)
    )
    # optimize(num_users,num_items,ratings,contents)

    n_factors = 10
    n_epochs = 30
    lr = 0.005
    alpha = 0.1

    # contents['imdbRating'] = pd.to_numeric(contents['imdbRating'], errors='coerce')
    # contents['imdbVotes'] = pd.to_numeric(contents['imdbVotes'].str.replace(',', ''), errors='coerce')

    # C = contents['imdbRating'].mean()

    # # Fill missing ratings with the average and missing votes with 0
    # contents['imdbRating'].fillna(C, inplace=True)
    # contents['imdbVotes'].fillna(0, inplace=True)

    # # Calculate the 60th percentile of votes
    # m = contents['imdbVotes'].quantile(0.8)

    # # Calculate the weighted IMDb score
    # contents['weighted_imdb_score'] = (
    #     (contents['imdbVotes'] / (contents['imdbVotes'] + m)) * contents['imdbRating'] +
    #     (m / (contents['imdbVotes'] + m)) * C
    # )
    content = combine_textual_info(contents)
    recomender = Rocchio(content, ratings, contents["imdbVotes"])
    recomender.tfidf()
    recomender.user_vector()

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

    # Predição das avaliações para o arquivo "targets.csv"
    predictions = model.recommend(targets)
    print(predictions)
    target["Rating"] = predictions
    # target = target.drop(['UserId','ItemId'],axis=1)

    target = ranking(target, users, items)
    target = target.drop("Rating", axis=1)

    target.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
