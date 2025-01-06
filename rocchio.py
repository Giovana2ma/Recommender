import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import sys



class Rocchio:

    def __init__(self,content,ratings,weight,alpha,beta,gamma):
        self.content = content 
        self.ratings = ratings 
        self.weight = weight 
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 


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
        
        weight = float(self.weight[item_index].replace(',', '')) if self.weight[item_index] != "N/A" else 3.0
        pred_scaled = pred[0][0] * weight
        
        return pred_scaled
    
    def recommend(self, data):
        '''
        data: pares <usuario,item> 
        Prevê a avaliação.
        '''
        predictions = np.array([self.predict(user, item) for user, item in zip(data['UserId'], data['ItemId'])])
        return predictions
    
    def update_user_profiles(self, user_feedback):
        for user_id, feedback in user_feedback.items():
            relevant_items = feedback['relevant']
            non_relevant_items = feedback['non_relevant']
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
    
# ratings_file = sys.argv[1]
# content_file = sys.argv[2]
# targets_file = sys.argv[3]

# target,users, items, ratings, targets, num_users, num_items,contents = process_files(ratings_file, targets_file,content_file)
# Convert 'imdbRating' and 'imdbVotes' to numeric, coercing errors to NaN
# contents['imdbRating'] = pd.to_numeric(contents['imdbRating'], errors='coerce')
# contents['imdbVotes'] = pd.to_numeric(contents['imdbVotes'].str.replace(',', ''), errors='coerce')

# # Calculate average rating across all items
# C = contents['imdbRating'].mean()

# # Fill missing ratings with the average and missing votes with 0
# contents['imdbRating'].fillna(C, inplace=True)
# contents['imdbVotes'].fillna(0, inplace=True)

# # Calculate the 60th percentile of votes
# m = contents['imdbVotes'].quantile(0.6)

# # Calculate the weighted IMDb score
# contents['weighted_imdb_score'] = (
#     (contents['imdbVotes'] / (contents['imdbVotes'] + m)) * contents['imdbRating'] +
#     (m / (contents['imdbVotes'] + m)) * C
# )
# print(contents['weighted_imdb_score'] )
# content = extract_info(contents)
# recomender = Rocchio(content,ratings,contents['imdbVotes'])
# recomender.tfidf()
# recomender.user_vector()
# predictions = recomender.recommend(targets)
# target['Rating'] = predictions
# target = ranking(target,users,items)
# target = target.drop('Rating',axis = 1)

# target.to_csv('predictionss.csv',index=False)