# 📦 Streamlit App – Cas pratique E-commerce Ameublement

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

st.title("🛒 Recatégorisation & Extraction - Marketplace Ameublement")
st.markdown("""
Ce projet permet de :
1. Identifier les lignes mal catégorisées dans un fichier de ventes e-commerce
2. Les recatégoriser automatiquement à partir des catégories existantes (colonne "nature")
3. Extraire les dimensions et couleurs des produits à partir de leur description
""")

# --- Upload du fichier Excel ---
uploaded_file = st.file_uploader("Chargez le fichier à recatégoriser", type=["xlsx", "xls", "xlsm"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    st.write(xls.sheet_names)  # Afficher les noms des feuilles disponibles

    # Charger une feuille par son nom
    df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_name[0], engine='openpyxl')  # Exemple de chargement de la première feuille
    st.dataframe(df.head())

    #st.write(df.sheet_names)
    st.write("Aperçu du fichier :")
    #st.dataframe(df.head())
    st.success("Fichier chargé avec succès !")
    st.write(df.head())

    # --- Nettoyage et préparation ---
    st.subheader("🔍 Recatégorisation automatique")

    if 'nature' not in df.columns or 'description' not in df.columns:
        st.error("Le fichier doit contenir au minimum les colonnes 'nature' et 'description'")
    else:
        # Nettoyage des données
        df = df.dropna(subset=['nature', 'description'])
        df['description_clean'] = df['description'].str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)

        # Encodage des catégories
        le = LabelEncoder()
        y = le.fit_transform(df['nature'])

        # Modèle de prédiction
        X_train, X_test, y_train, y_test = train_test_split(df['description_clean'], y, test_size=0.2, random_state=42)

        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])

        model.fit(X_train, y_train)
        df['predicted_nature'] = le.inverse_transform(model.predict(df['description_clean']))
        df['is_misclassified'] = df['nature'] != df['predicted_nature']

        st.write("Lignes mal catégorisées :")
        st.dataframe(df[df['is_misclassified']][['description', 'nature', 'predicted_nature']].head(10))

        csv_misclassified = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger le fichier recatégorisé", csv_misclassified, file_name="recategorised_products.csv")

    # --- Extraction de dimensions et couleurs ---
    st.subheader("🎯 Extraction des dimensions et couleurs")

    def extract_dimensions(text):
        matches = re.findall(r'(\d{2,3})[x×](\d{2,3})', text)
        return ' x '.join(matches[0]) if matches else None

    def extract_colors(text):
        couleurs_connues = ["blanc", "noir", "gris", "rouge", "bleu", "beige", "vert", "jaune", "marron"]
        for color in couleurs_connues:
            if color in text:
                return color
        return None

    df['dimension'] = df['description_clean'].apply(extract_dimensions)
    df['couleur'] = df['description_clean'].apply(extract_colors)

    st.write("Aperçu des dimensions et couleurs extraites :")
    st.dataframe(df[['description', 'dimension', 'couleur']].head(10))

    csv_full = df.to_csv(index=False).encode('utf-8')
    st.download_button("📦 Télécharger le fichier enrichi", csv_full, file_name="products_enriched.csv")
