import os
from flask import Flask, render_template, redirect, url_for, request, jsonify
# from flask_caching import Cache
from werkzeug.wrappers import Request, Response
# from prediction_model import PredictionModel
import base64
import pandas as pd
from pickle import load
import pickle
import numpy as np
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

# streamlit_url = os.environ.get('streamlit_url')
# https://credit-scoring-streamlit-194aaf4426c2.herokuapp.com/
# flask_url = os.environ.get('flask_url')
streamlit_url = "https://credit-scoring-streamlit-194aaf4426c2.herokuapp.com/"



@app.route('/templates')
def index():
    return render_template('index.html')
    # render_template('index.html', streamlit_url=streamlit_url)


if __name__ == '__main__':
        app.run(port=5000, debug=True)










"""
# pred_model = PredictionModel()
app = Flask(__name__)
MODEL_PATH = 'model_global.pkl'
LIME_PATH = 'lime_global.pkl'
LOCAL_FEAT_IMPORTANCE_PATH = 'feature_importance_locale.txt'
# TEST_DATA_PATH = 'donnees_test.json'

def load_pickle(path): 
    result = None
    pickle_in = open(path, "rb")
    result = load(pickle_in)
    pickle_in.close()
    return result 

_model = load_pickle(MODEL_PATH)
explainer = load_pickle(LIME_PATH)


SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
print('SITE_ROOT :', SITE_ROOT)
json_url = os.path.join(SITE_ROOT, 'small_donnees_test.json')  # fonctionnel

print('json_url :', json_url)
if os.path.exists(json_url) and os.path.getsize(json_url) > 0:
    test_data = pd.read_json(json_url)
    print('test_data :', test_data)
else:
    print(f"File {json_url} does not exist or is empty.")




def imageToString(image_path):
    b64_string = ''
    with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
    return b64_string.decode('utf-8')




@app.route("/api/predict", methods=["GET"])


def predict():
    args = request.args
    client_Id = args['sk_id_curr']
    print(client_Id)
    arr_results = []
    features = test_data.loc[test_data['SK_ID_CURR'] == int(client_Id)]  # donnees du client retournees
    features = features.loc[:, features.columns != 'SK_ID_CURR'].to_numpy()
    

    # preparation input
    print('Loaded features shape: ', features.shape)

    # prediction
    arr_results = _model.predict_proba(features)

    # prediction Local Interpretable Model-agnostic Explanations
    print('Indexe du numero client :', np.where(test_data['SK_ID_CURR']==int(client_Id))[0][0])  # Afficher l'index de SK_ID_CURR
    client_index = np.where(test_data['SK_ID_CURR']==int(client_Id))[0][0]
    print('debug : ', test_data.iloc[client_index, 0:-1])

    local_feat_importance = explainer.explain_instance(test_data.iloc[client_index, 0:-1],
                                                    _model.predict_proba,
                                                    num_samples=100)  # passer X en format numpy array
    '''local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index],
                                                    self._model.predict_proba,
                                                    num_samples=100)  # passer X en format numpy array'''
    temp_ = local_feat_importance.as_list()
    print ('temp_ :\n', temp_)

    # conversion vecteur en tags
    label = {'numero client':client_Id, 'label_1':'good', 'score_1':float(arr_results[0][0]),'label_2':'bad', 'score_2':float(arr_results[0][1]),\
                'feature_importance_locale':dict(local_feat_importance.as_list())} 
    print(f'label predicted :\n{label}')
    print(label)

    # Filtrer les donnees du client specifique
    client_data = test_data.loc[test_data['SK_ID_CURR'] == int(client_Id)]

    # Extraire les donnees necessaires du client specifique
    other_client_info = {
                         'CODE_GENDER': [client_data['CODE_GENDER'].values[0].item(), 'Genre'],
                         'FLAG_OWN_CAR': [client_data['FLAG_OWN_CAR'].values[0].item(),'Propriétaire de voiture'],
                         'AMT_INCOME_TOTAL': [client_data['AMT_INCOME_TOTAL'].values[0].item(), 'Revenu total'],
                         'AMT_CREDIT': [client_data['AMT_CREDIT'].values[0].item(), 'Montant du prêt'],
                         'AMT_ANNUITY': [client_data['AMT_ANNUITY'].values[0].item(), 'Montant de l\'annuité'],
                         'DAYS_BIRTH': [client_data['DAYS_BIRTH'].values[0].item(), 'Date de naissance'],
                         'DAYS_EMPLOYED_PERC': [client_data['DAYS_EMPLOYED_PERC'].values[0].item(),
                                                'Nombre de jours travaillé avant la de demande de prêt'],
                         'DAYS_REGISTRATION': [client_data['DAYS_REGISTRATION'].values[0].item(),
                                               'Nombre de jours entre le dernier enregistrement et la demande de prêt'],
                         'NAME_FAMILY_STATUS_Married': [client_data['NAME_FAMILY_STATUS_Married'].values[0].item(), 'Situation Maritale'],
                         'NAME_EDUCATION_TYPE_Secondary / secondary special': [
                             client_data['NAME_EDUCATION_TYPE_Secondary / secondary special'].values[0].item(),
                             'Nombre de diplômés du secondaire']
                        }

    print(f'\n \n \n **** \nOther client information :\n{other_client_info}')

    return jsonify({
                    'status': 'ok',
                    'data': label,
                    'other_data': other_client_info
                    })

@app.route("/api/model_performance", methods=["GET"])

def get_model_performance():
    return jsonify({
                    'status': 'ok',
                    'features_importances' :  imageToString('images/Feature_Importance_Globale.png'),  # localhost:5000/images/Feature_Importance_Globale.png
                    })




@app.route("/api/client_comparison", methods=["GET"])

def get_client_comparison():
    return jsonify({
                    'status': 'ok',
                    'CODE_GENDER' : imageToString('images/CODE_GENDER.png'),
                    'FLAG_OWN_CAR' : imageToString('images/FLAG_OWN_CAR.png'),
                    'AMT_INCOME_TOTAL' : imageToString('images/AMT_INCOME_TOTAL.png'),
                    'AMT_CREDIT' : imageToString('images/AMT_CREDIT.png'),
                    'AMT_ANNUITY' : imageToString('images/AMT_ANNUITY.png'),
                    'DAYS_BIRTH' : imageToString('images/DAYS_BIRTH.png'),
                    'DAYS_EMPLOYED_PERC' : imageToString('images/DAYS_EMPLOYED_PERC.png'),
                    'DAYS_REGISTRATION' : imageToString('images/DAYS_REGISTRATION.png'),
                    'NAME_FAMILY_STATUS_Married' : imageToString('images/NAME_FAMILY_STATUS_Married.png'),
                    'NAME_EDUCATION_TYPE_Secondary_secondary_special' : imageToString('images/NAME_EDUCATION_TYPE_Secondary_secondary_special.png')
                    })  


# cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# @app.route('/clear_cache')
# def clear_cache():
    # cache.clear()
    # return 'Cache has been cleared'


if __name__ == "__main__":
    app.run(port=5000, debug=True)
"""