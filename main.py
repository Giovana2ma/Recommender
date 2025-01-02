import numpy as np
from utils import *
from funksvd import *
import sys


def main():

    # Leitura dos arquivos de entrada
    ratings_file = sys.argv[1]
    content_file = sys.argv[2]
    targets_file = sys.argv[3]

    target,users, items, ratings, targets, num_users, num_items,contents = process_files(ratings_file, targets_file,content_file)
    
    n_factors = 10
    n_epochs = 30
    lr = 0.001
    alpha = 0.1

    model = Funksvd(ratings,num_users,num_items,n_factors,n_epochs,lr,alpha)
    model.run_sgd()

    # Predição das avaliações para o arquivo "targets.csv"
    predictions = model.recommend(targets)
    print(predictions)
    target['Rating'] = predictions
    # target = target.drop(['UserId','ItemId'],axis=1)

    target = ranking(target,users,items)
    target = target.drop('Rating',axis = 1)

    target.to_csv('predictions.csv',index=False)


if __name__ == "__main__":
    main()