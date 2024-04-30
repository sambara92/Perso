import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
data = pd.read_csv(url)

# Gérer l'avertissement PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Fonction d'analyse de corrélation
def correlation_analysis(data):
    st.subheader("Analyse de corrélation")
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    st.pyplot()

# Fonction d'analyse de distribution
def distribution_analysis(data):
    st.subheader("Analyse de distribution")
    sns.pairplot(data, diag_kind='kde')
    plt.tight_layout()  # Assurez-vous que la disposition est serrée
    st.pyplot()

# Filtrer les données par région
def filter_by_region(data, region):
    filtered_data = data[data['continent'] == region]
    return filtered_data

# Interface utilisateur Streamlit
def main():
    st.title("Analyse des voitures en fonction de la région")
    
    # Afficher les données brutes
    st.write("Aperçu des données brutes :")
    st.write(data.head())

    # Analyse de corrélation
    correlation_analysis(data)

    # Analyse de distribution
    distribution_analysis(data)

    # Filtrer par région
    regions = data['continent'].unique()
    selected_region = st.selectbox("Sélectionner une région :", regions)
    filtered_data = filter_by_region(data, selected_region)

    # Afficher les données filtrées
    st.subheader(f"Données pour la région : {selected_region}")
    st.write(filtered_data.head())

if __name__ == "__main__":
    main()
