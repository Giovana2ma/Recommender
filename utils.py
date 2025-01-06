import numpy as np
import pandas as pd
from funksvd import *
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def read_data(file):
    rate = pd.read_json(file,lines=True) if file[-5:] == 'jsonl' else pd.read_csv(file)
    return rate

def get_keys(rates,content):
    '''
    Constroe uma chave única para cada usuário e item
    '''
    unique_users = set(rates['UserId'])
    unique_items = content['ItemId']

    users = {idx:user_id  for idx, user_id in enumerate(unique_users)}
    items = {idx: item_id for idx, item_id in enumerate(unique_items)}

    return users,items

def process_data(data,users,items):
    '''
    Processa os dados de entrada para substituir identificadores de usuários e itens por índices numéricos.
    '''
    user_map = {v: k for k, v in users.items()}
    item_map = {v: k for k, v in items.items()}

    if 'UserId' in data.columns:
        data['UserId'] = data['UserId'].map(user_map).values.astype(int)
    data['ItemId'] = data['ItemId'].map(item_map).values.astype(int)


    return data

def process_files(ratings_file, targets_file,content_file):
    rates = read_data(ratings_file)
    target = read_data(targets_file)
    content = read_data(content_file)

    # Processamento dos dados iniciais
    users,items = get_keys(rates,content)
    ratings = process_data(rates,users,items)
    targets = process_data(target,users,items)
    contents = process_data(content,users,items)

    num_users = len(users)
    num_items = len(items)

    return target,users,items,ratings,targets,num_users,num_items,contents

def preprocess_document(doc):
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    """
    Preprocess a document 
    """
    # Lowercase the document
    doc = doc.lower()
    
    # Remove special characters and numbers
    doc = re.sub(r"[^a-zA-Z\s]", "", doc)
    
    # Tokenize the document
    tokens = nltk.word_tokenize(doc)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the tokens back into a single string
    preprocessed_doc = " ".join(tokens)
    return preprocessed_doc

def extract_info(content_data):
    info = content_data.apply(lambda x: ' '.join([
        preprocess_document(x['Plot']),
        preprocess_document(x['Title']),
        preprocess_document(x['Genre']),
        preprocess_document(x['Director']),
        preprocess_document(x['Writer']),
        preprocess_document(x['Actors']),
        preprocess_document(x['Language']),
        preprocess_document(x['Country'])
    ]), axis=1)
    return info

def ranking(rates,users,items):
    user_map = {k: v for k, v in users.items()}
    item_map = {k: v for k, v in items.items()}

    rates['UserId'] = rates['UserId'].map(user_map)
    rates['ItemId'] = rates['ItemId'].map(item_map)

    rates = rates.groupby('UserId', group_keys=False).apply(
    lambda group: group.iloc[(-group['Rating']).argsort()]
    )
    return rates




def relevance_grade(rating):
    if rating == 10:
        return 3
    if 8 <= rating <= 9:
        return 2
    if 6 <= rating <= 7:
        return 1
    return 0
    
def ndcg(rank_pred,rank_true, k=20):
    """
    Calculates nDCG@20 for a set of user-item ratings and predictions.
    """
    
    # Map relevance grades
    rank_pred['Relevance'] = rank_pred['Rating'].apply(relevance_grade)
    rank_true['Relevance'] = rank_true['Rating'].apply(relevance_grade)
    
    ndcgs = []
    for user_id in rank_pred['UserId'].unique():

        user_pred = rank_pred[rank_pred['UserId'] == user_id]
        user_true = rank_true[rank_true['UserId'] == user_id]
        
        # Take the top-k items
        user_pred_relevance = user_pred['Relevance'][:k].tolist()
        user_true_relevance = user_true['Relevance'][:k].tolist()
        
        # Compute DCG@k
        dcg = sum((2**rel - 1) / np.log2(i + 1) for i, rel in enumerate(user_pred_relevance, start=1))
        
        # Compute IDCG@k
        idcg = sum((2**rel - 1) / np.log2(i + 1) for i, rel in enumerate(sorted(user_true_relevance, reverse=True), start=1))
        
        # Calculate nDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    # Average nDCG across users
    return np.mean(ndcgs)

def split_data(data):
    '''
    Divide os dados em treino(0.8) e teste(0.2)
    '''
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int((1 - 0.2) * len(shuffled_data))

    train_data = shuffled_data.iloc[:split_index]
    test_data = shuffled_data.iloc[split_index:]

    return train_data,test_data

def rmse(pred,rate):
        return np.sqrt(np.mean(np.square(rate - pred)))

def optimize(num_users,num_items,data,content):
    '''
    Executa o algoritmo de FunkSvd para a busca dos melhores paramêtros
    '''
    
    n_factors_list = range(10, 201, 30)  
    n_epochs_list = range(10, 51, 10)    
    lr_list = [0.001, 0.005, 0.01]       
    reg_list = [0.01, 0.05, 0.1]  
    best_params = {}
    best_rmse = float('inf')

    train_data,test_data = split_data(data)

    for n_factors in n_factors_list:
        for n_epochs in n_epochs_list:
            for lr in lr_list:
                for reg in reg_list:
                    model = Funksvd(train_data,content['imdbVotes'],num_users,num_items,n_factors,n_epochs,lr,reg)
                    model.run_sgd()  
                    
                    y_pred = model.recommend(test_data)
                    y_true = test_data['Rating']

                    current_rmse = rmse(y_pred,y_true)

                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_params = {'n_factors': n_factors, 'n_epochs': n_epochs, 'lr': lr, 'reg': reg}

                    print(f"Params: n_factors={n_factors}, n_epochs={n_epochs}, lr={lr}, reg={reg} -> RMSE: {current_rmse:.4f}")

    print(f"\nBest Parameters: {best_params}, Best RMSE: {best_rmse:.4f}")
    return best_params, best_rmse
    