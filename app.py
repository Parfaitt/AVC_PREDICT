import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.sav', 'rb'))
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.sidebar.header("Informations")
st.sidebar.write('''
# Application prédiction de L'accident vasculaire cérébral (AVC)
Cet ensemble de données est utilisé pour prédire si un patient est susceptible de subir un AVC

Auteur: Parfait Tanoh N'goran
''')


st.title("Application de prédiction :ship:")


def main():
    def load_data():
        data = pd.read_csv("healthcare-dataset-stroke-data.csv")
        return data

    df = load_data()
    df_sample = df.sample(5)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeux de données brutes")
        st.write(df_sample)

    gender = st.selectbox("Genre: 0 -> Femme, 1 -> Garçon", [0, 1])
    age = st.text_input("Entrer votre age")
    hypertension = st.selectbox("Hypertension: 0 -> Non, 1 -> Oui", [0, 1])
    heart_disease = st.selectbox("Diabètes: 0 -> Non, 1 -> Oui", [0, 1])
    ever_married = st.text_input("Marié-e: 0 -> Non, 1 -> Oui")
    Residence_type = st.text_input("Résidence: 0 -> Urbain, 1 -> Rural")
    avg_glucose_level = st.text_input("Taux de glucose")
    bmi = st.text_input("(BMI) indique le rapport entre votre poids et votre taille", '25.5')
    smoking_status = st.text_input("Fumeur: 0 -> Jamais fumé, 1 -> inconnu, 2 -> Ancien fumeur, 3 -> Fumeur")

    def predict():
        if not age or not ever_married or not Residence_type or not avg_glucose_level or not bmi or not smoking_status:
            st.error("Veuillez remplir tous les champs obligatoires.")
            return

        data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status,
        }

        X = pd.DataFrame(data, index=[0])
        prediction = model.predict(X)
        if prediction[0] == 1:
            st.success("Le patient aura un AVC :thumbsdown:")
        else:
            st.error("Le patient n'aura pas d'AVC :thumbsup:")

    trigger = st.button('Predict', on_click=predict)


if __name__ == '__main__':
    main()
