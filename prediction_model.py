import os
import joblib
import pandas as pd
import pickle
import string
import sklearn
import scipy
import lime
from pickle import load
import dill
import gdown

# ----------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
from lightgbm import LGBMClassifier

# ----------------------------------------------------


print('Tags transformer loaded.')
from scipy import sparse
import re
import numpy as np
from os import path
import sys
from pathlib import Path


data_dir = '/app'
model_dir = 'model'
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
list_of_dirs = []

def list_dir():
    '''for root, dirs, files in os.walk(CURRENT_DIRECTORY):
        for dir in dirs:  # for file in files:
            print(dir)  # file'''

    for root, dirs, files in os.walk(CURRENT_DIRECTORY):
        for dire in dirs:
            list_of_dirs.append(os.path.join(root, dire))
    for name in list_of_dirs:
        print(name)

    # CURRENT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # Path(__file__).resolve().parent.parent
    # os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # Afficher l'arborescence du parent pour voir ou heroku depose les scripts



def load_pickle(path): 
    result = None
    pickle_in = open(path, "rb")
    result = load(pickle_in)
    pickle_in.close()
    return result 

def load_from_google_cloud(id, output):
    # url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
    # id = "1ABIbSgQrQWn07UkJoDcEsl9vT1rbz2lk" 
    # ID tiré du lien de partage du fichier dans google cloud avec un accès général donné à tous les utilisateurs
    gdown.download(id=id, output=output, quiet=False)


class PredictionModel:
    MODEL_PATH = os.path.join(model_dir, 'model_global.pkl')
    LIME_PATH = os.path.join(model_dir, 'lime_global.pkl')
    LOCAL_FEAT_IMPORTANCE_PATH = os.path.join(model_dir, 'feature_importance_locale.txt')
    TEST_DATA_PATH = os.path.join(CURRENT_DIRECTORY, 'small_donnees_test.json') # '/app/donnees_test.json'  # donnees_test
    # TEST_DATA_PATH = os.path.join(model_dir, 'donnees_test.pkl')
    local_feat_importance = None
    test_data = None


    def __init__(self) -> None:
        self._model = self.import_predict_model()  # load lgbm
        self.explainer = self.import_lime_model()  # laod explainer
        # self.test_data = pd.read_pickle(self.TEST_DATA_PATH)  # load dataframe donnees test
        # load dataframe donnees test
        # self.test_data = load_pickle(self.TEST_DATA_PATH)
        # if not os.path.exists(self.TEST_DATA_PATH):
        # load_from_google_cloud('1ABIbSgQrQWn07UkJoDcEsl9vT1rbz2lk', self.TEST_DATA_PATH)
        # print(CURRENT_DIRECTORY)
        # print(TEST_DATA_PATH)
        list_dir()  # CURRENT_DIRECTORY
        self.test_data = pd.read_json(self.TEST_DATA_PATH)
        
        # self.load_feat_importance_local()  # self.local_feat_importance = self.load_feat_importance_local()
        pass

    def load_features(self, client_Id):
        print('Dataframe :\n', self.test_data.head())
        # Drop SK_ID_CURR pour modeliser selon le nombre de features du notebook
        return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)]
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)].to_numpy()
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)].drop(columns=['SK_ID_CURR']).to_numpy()
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == float(client_Id)].to_numpy()



    def predict(self, client_Id):
        arr_results = []
        # features = self.load_features(np.delete(client_Id, -1, axis=1))  # donnees du client retournees sans SK_ID_CURR
        features = self.load_features(client_Id)  # donnees du client retournees
        features = features.loc[:, features.columns != 'SK_ID_CURR'].to_numpy() # !!! => ajout

        # preparation input
        print('Loaded features shape: ', features.shape)

        # prediction
        arr_results = self._model.predict_proba(features)

        # prediction Local Interpretable Model-agnostic Explanations

        # client_index = self.test_data[self.test_data['SK_ID_CURR'] == float(client_Id)].index
        # print(self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)])
        print('Indexe du numero client :', np.where(self.test_data['SK_ID_CURR']==int(client_Id))[0][0])  # Afficher l'index de SK_ID_CURR
        client_index = np.where(self.test_data['SK_ID_CURR']==int(client_Id))[0][0]
        print('debug : ', self.test_data.iloc[client_index, 0:-1])


        # client_features = self.test_data.iloc[client_index]  # Recuperer les features du client
        # client_features = client_features.drop(columns=['SK_ID_CURR'])  # Drop SK_ID_CURR pour expliquer la participation des features
        # client_features = client_features.to_numpy()
        # df.drop('SK_ID_CURR', axis=1, inplace=True)
        # self.test_data.iloc[client_index].drop('SK_ID_CURR', axis=1, inplace=True)
        self.local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index, 0:-1],
                                                       self._model.predict_proba,
                                                       num_samples=100)  # passer X en format numpy array
        '''local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index],
                                                       self._model.predict_proba,
                                                       num_samples=100)  # passer X en format numpy array'''
        temp_ = self.local_feat_importance.as_list()
        print ('temp_ :\n', temp_)
        # print('local_feat_importance :\n{}'.format(local_feat_importance))

        # conversion vecteur en tags
        label = {'numéro client' : client_Id, 'label_1':'good', 'score_1':float(arr_results[0][0]),'label_2':'bad', 'score_2':float(arr_results[0][1]),\
                 'feature_importance_locale':dict(self.local_feat_importance.as_list())}
        print(f'label predicted :\n{label}')
        return label


    def model_performance(self, X):
        info_perf= ''
        return info_perf
    
    def import_predict_model(self):
        return  load_pickle(self.MODEL_PATH)

    def import_lime_model(self):
        return load_pickle(self.LIME_PATH)

if __name__ =='__main__':
    test = PredictionModel()
    input_test = {}
    input_test['txt_question'] = 'Convert a Python list of lists to a single string'
    input_test['txt_body'] = "<p>I have list of lists consisting of chars and integers that I want to convert into a single string"
    test.predict(input_test)
