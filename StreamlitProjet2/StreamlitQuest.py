
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
@st.cache
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv")
    return data

# Analyse de corrélation
def correlation_analysis(data):
    st.subheader("Analyse de corrélation")
    data_numeric = data.select_dtypes(include=['float64', 'int64'])
    corr = data_numeric.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Distribution des données
def distribution_analysis(data):
    st.subheader("Distribution des données")
    sns.pairplot(data, diag_kind='kde')
    st.pyplot()

# Filtrage par continent
def filter_by_continent(data):
    st.subheader("Filtrage par continent")
    continents = data['continent'].unique()
    selected_continent = st.selectbox("Sélectionnez un continent :", continents)
    filtered_data = data[data['continent'] == selected_continent]
    return filtered_data

# Chargement des données
data = load_data()

# Titre de l'application
st.title("Analyse des données des voitures")

# Analyse de corrélation et de distribution
correlation_analysis(data)
distribution_analysis(data)

# Filtrage par continent
filtered_data = filter_by_continent(data)

# Affichage des données filtrées
st.subheader("Données filtrées par continent")
st.write(filtered_data)
