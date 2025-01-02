import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import sys



class Rocchio:

    def __init__(self,content,ratings):
        self.content = content 
        self.ratings = ratings 

    def tfidf(self):
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(self.content)

    def user_vector(self):
        self.user_vectors = {}
        # Itera sobre os usuários únicos
        for user_id in self.ratings['UserId'].unique():
            # Filtra os itens interagidos pelo usuário
            user_data = self.ratings[self.ratings['UserId'] == user_id]
            ratings = csr_matrix(user_data['Rating'].values.reshape(-1,1))
            item_indices = user_data['ItemId']
            
            # Multiplica as avaliações pelos vetores TF-IDF dos itens
            terms = self.tfidf_matrix[item_indices].T
            weighted_vectors = terms @ ratings
            
            # Soma os vetores ponderados e divide pelo número de itens avaliados
            user_vector = weighted_vectors / len(item_indices)
            # Adiciona ao dicionário o vetor do usuário
            self.user_vectors[user_id] = user_vector
            
        return 
    
    def predict(self, user_id, item_index):
        '''
        Recebe um par usuário e item e realiza a predição da avaliação.
        '''
        # Obtém o vetor do usuário
        user_vector = self.user_vectors[user_id].toarray().flatten()
        
        # Obtém o vetor do item
        item_vector = self.tfidf_matrix[item_index].toarray().flatten()
        
        # Calcula a similaridade do cosseno entre o vetor do usuário e o vetor do item
        pred = cosine_similarity(user_vector.reshape(1, -1), item_vector.reshape(1, -1))
        
        # Escala a predição para o intervalo desejado (aqui multiplicado por 10)
        pred_scaled = pred[0][0] * 10
        
        return pred_scaled
    
    def recommend(self, data):
        '''
        data: pares <usuario,item> 
        Prevê a avaliação.
        '''
        predictions = np.array([self.predict(user, item) for user, item in zip(data['UserId'], data['ItemId'])])
        return predictions
    
ratings_file = sys.argv[1]
content_file = sys.argv[2]
targets_file = sys.argv[3]

target,users, items, ratings, targets, num_users, num_items,contents = process_files(ratings_file, targets_file,content_file)
plot = contents['Plot']

recomender = Rocchio(plot,ratings)
recomender.tfidf()
recomender.user_vector()
predictions = recomender.recommend(targets)
target['Rating'] = predictions
target = ranking(target,users,items)
target = target.drop('Rating',axis = 1)

target.to_csv('predictionss.csv',index=False)