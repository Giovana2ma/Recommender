import numpy as np
from utils import *

class Funksvd:
    def __init__(self,ratings,num_users,num_items,n_factors,n_epochs,lr,alpha):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.alpha = alpha # Constante de regularizacao

        self.ratings = ratings
        self.num_users = num_users
        self.num_items = num_items

        self.global_mean = ratings['Rating'].mean()
        self.bu = None # Vetor de viés dos usuários         (U,1)
        self.bi = None # Vetor de viés dos itens            (I,1)
        self.p = None # Matriz de fator latente do usuário  (U,F)
        self.q = None # Matriz de fator latente do item     (I,F)
      

    def shuffle(self):
        '''
        Reodernação dos dados para realizar as predições
        e atualizações das matrizes de maneira aleatória.
        '''
        self.ratings = self.ratings.sample(frac=1).reset_index(drop=True)

    def initialization(self):
        '''
        Inicialização das matrizes de viéses e fatores latentes 
        dos items e usuários  
        '''
        self.bu = np.zeros(self.num_users) # (U,1)
        self.bi = np.zeros(self.num_items) # (I,1)

        self.p = np.random.normal(0, .1, (self.num_users, self.n_factors)) # (U,F)
        self.q = np.random.normal(0, .1, (self.num_items, self.n_factors)) # (I,F)
    
    def predict(self,user, item):
        '''
        Recebe um par usuário e item e realiza a predição da
        avaliação.

        A predição é realizada pela soma da média global, 
        viés do usuário e item e da multiplicação da matriz
        de fatores latentes.
        '''
        pred  = self.global_mean
        pred += self.bu[user] + self.bi[item]
        pred += np.dot(self.p[user],self.q[item])

        # Garante que a predição fique dentro do intervalo 1-5
        return max(0.0, min(10.0, pred))

    def run_iteration_(self):
        '''
        Realiza uma iteração do algoritmo. 
        Para cada par usuário e item dos dados de teste 
        faz a predição e atualiza as matrizes.
        '''
        for user, item, rating in zip(self.ratings['UserId'],self.ratings['ItemId'],self.ratings['Rating']):

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
        # O primeiro passo do algoritmo é inicializar os vetores aleatórios
        self.initialization()
        # Com as matrizes e vetores inicializados, eh possivel fazer o processo de otimização
        for _ in range(self.n_epochs):
            self.shuffle()
            self.run_iteration_()
    
    def recommend(self, data):
        '''
        data: pares <usuario,item> 
        Prevê a avaliação.
        '''
        predictions = np.array([self.predict(user, item) for user, item in zip(data['UserId'], data['ItemId'])])
        return predictions
    
    

    

