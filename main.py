import numpy as np
from utils import *
from funksvd import *
import sys


def main():

    # Leitura dos arquivos de entrada
    ratings_file = sys.argv[1]
    targets_file = sys.argv[2]

    rates = read_data(ratings_file)
    target = read_data(targets_file)

    # Processamento dos dados iniciais
    users,items = get_keys(rates,target)
    rates = filter_by_time(rates)
    ratings = process_data(rates,users,items)
    targets = process_data(target,users,items)

    num_users = len(users)
    num_items = len(items)
    
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
    target.drop('Rating',axis = 1)

    target.to_csv('predictions.csv',index=False)

if __name__ == "__main__":
    main()