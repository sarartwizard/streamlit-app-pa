from json import encoder

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import model_from_json
from sklearn import preprocessing

st.write('''
# Depression Detector
cette application diagnostique la presence de trouble depressif chez une personne de plus de 15ans
''')

st.sidebar.header("les parametres d'entrée")


def user_input():
    genre = st.sidebar.slider('Irritabilité  note', 0, 10, 5)
    Age = st.sidebar.slider('Irritabilitéjk  note', 0, 10, 5)
    Trouble_du_Sommeil = st.sidebar.slider('sommeil note', 0, 10, 5)
    Fatigue_intense = st.sidebar.slider('Fatigue intense note',0,10,5)
    Ralentissement_psychomoteur_général= st.sidebar.slider('manque denergie note',0,10,5)
    Perte_de_confianceen_soi = st.sidebar.slider('estime de soi note', 0, 10, 5)
    Anxiété= st.sidebar.slider('Anxiété note',0,10,5)
    Irritabilite_frustration = st.sidebar.slider('frustee  note', 0, 10, 5)
    Troubles_de_la_mémoire = st.sidebar.slider('trouble de la memoire note',0,10,5)
    Douleur_physique_sans_causes= st.sidebar.slider('Douleur_physique_sans_causes physique note',0,10,5)
    envies_suicidaires= st.sidebar.slider('envies_suicidaires suicidaire note',0,10,5)
    modififcation_de_lappetit = st.sidebar.slider('modification de lappetit note', 0, 10, 5)
    Fausses_croyances= st.sidebar.slider('Fausses_croyances',0,10,5)
    Hallucination= st.sidebar.slider('Hallucination de lappetit note',0,10,5)
    interval_de_temps = st.sidebar.slider('interval_de_temps  note', 0, 2, 1)
    variablededepre = st.sidebar.slider('variablededepre  note', 0, 2, 1)
    Hyperactivité = st.sidebar.slider('Fatigue Hyperactivité note', 0, 10, 5)
    bonheur_intense = st.sidebar.slider('bonheur_intense bonheur_intense note', 0, 10, 5)
    estime_de_soi_démesuré = st.sidebar.slider('estime_de_soi_démesuré denergie note', 0, 10, 5)
    accéleration_de_la_pensé = st.sidebar.slider('accéleration_de_la_pensé note', 0, 10, 5)
    grande_distraction = st.sidebar.slider('grande_distraction physique note', 0, 10, 5)
    comportement_a_risque = st.sidebar.slider('comportement_a_risque de la memoire note', 0, 10, 5)
    energie_debordante = st.sidebar.slider('trouble energie_debordante la memoire note', 0, 10, 5)
    dimunition_du_besoin_de_dormir = st.sidebar.slider('dimunition_du_besoin_de_dormir de la memoire note', 0, 10, 5)
    variableB = st.sidebar.slider('variableB de la memoire note', 0, 2, 1)
    interval_de_temps2 = st.sidebar.slider('interval_de_temps2 de la memoire note', 0, 10, 1)




    data = {
            'genre': genre,
            'Age':Age,
            'Trouble_du_Sommeil': Trouble_du_Sommeil,
            'Fatigue_intense': Fatigue_intense,
            'Ralentissement_psychomoteur_général': Ralentissement_psychomoteur_général,
            'Perte_de_confianceen_soi': Perte_de_confianceen_soi,
            'envies_suicidaires':envies_suicidaires,
            'Anxiété':Anxiété,
            'Irritabilite_frustration': Irritabilite_frustration,
            'Troubles_de_la_mémoire': Troubles_de_la_mémoire,
            'Douleur_physique_sans_causes': Douleur_physique_sans_causes,
            'modififcation_de_lappetit': modififcation_de_lappetit,
            'Fausses_croyances' : Fausses_croyances,
            'Hallucination' : Hallucination,
            'interval_de_temps' : interval_de_temps,
            'variablededepre' : variablededepre,
            'Hyperactivité' : Hyperactivité,
            'bonheur_intense' : bonheur_intense,
            'estime_de_soi_démesuré' : estime_de_soi_démesuré,
            'accéleration_de_la_pensé' : accéleration_de_la_pensé,
            'grande_distraction' : grande_distraction,
            'comportement_a_risque' : comportement_a_risque,
            'energie_debordante' : energie_debordante,
            'dimunition_du_besoin_de_dormir' : dimunition_du_besoin_de_dormir,
            'variableB' : variableB,
            'interval_de_temps2': interval_de_temps

    }


    parametres_depression =pd.DataFrame(data,index=[0])
    return parametres_depression

df=user_input()
nmp=df.to_numpy()
print(df)

st.subheader('Veuillez évaluer vos symptomes sur une échelle de 1 à 10. Plus le chiffre est élevé, plus le symptome est intense.')
st.subheader(' 0 ->  Jamais')
st.subheader(' 1 et 3 ->  Rarement')
st.subheader('entre 3 et 5 -> Souvent')
st.subheader('entre 5 et 8 -> Tres souvent')
st.subheader('entre 8 et 10 -> Tout le temps')



st.write(df)

# charger le modele pour faire des prédictions sur des nouvelles données
json_file = open("model_MLPCLASSIFER"+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_MLPCLASSIFER"+".h5")
print(" -------  The model is  loaded from disk  -------")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Tester sur de nouvelles données (données de test)
from sklearn.datasets import make_blobs
xnew, _ = make_blobs(n_samples=1, centers=2, n_features=26, random_state=1)
ynew = model.predict_proba(xnew)


for i in range(len(xnew)):
	print("X=%s, Predicted=%s" % (xnew[i], ynew[i]))


ynew = np.argmax(ynew, axis= 1)

for i in range(len(xnew)):
	print("X=%s, Predicted=%s" % (xnew[i], ynew[i]))


Xnew = nmp
ynew = model.predict(Xnew)
print (ynew)
ynew = np.argmax(ynew, axis= 1)
res = encoder.inverse_transform(np.array([3]).reshape(1, -1))

st.write(res)

# depression = pd.read_excel('C:/Users/nadou/OneDrive/Documents/Depression.xlsx')
# rfc = RandomForestClassifier(n_estimators=100)
#
#
# train, test = train_test_split(depression, test_size=0.2)
# target_name = train["Diagnostique"]
# train_feat = train.iloc[:,:10]
# train_targ = train["Diagnostique"]
#
# test_feat = test.iloc[:,:10]
# test_targ = test["Diagnostique"]
#
# rfc.fit(train_feat, train_targ)
#
# prediciton = rfc.predict(df)
#
# st.subheader("votres état mental:")
# st.write(prediciton)