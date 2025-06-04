import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import os
import plotly.express as px

@st.cache_data
def load_and_preprocess_future_data():
    """
    Charge et prétraite le fichier Future_temp.csv.
    Cette fonction ne s'exécutera qu'une seule fois.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # /mount/src/pages/
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # /mount/src/
    csv_path = os.path.join(ROOT_DIR, "dataset_streamlit", "Future_temp.csv")

    df_future_temp = pd.read_csv(csv_path, index_col=0)

    df_future_temp['time'] = pd.to_datetime(df_future_temp['time'])
    df_future_temp.rename(columns={'time': 'Date', 'region': 'Région', 'C°': 'TMoy (°C)'}, inplace=True)

    df_future_temp['Année'] = df_future_temp['Date'].dt.year
    df_future_temp['month'] = df_future_temp['Date'].dt.month
    df_future_temp['day_of_week'] = df_future_temp['Date'].dt.dayofweek
    df_future_temp['day_of_year'] = df_future_temp['Date'].dt.dayofyear
    df_future_temp['week_of_year'] = df_future_temp['Date'].dt.isocalendar().week.astype(int) 

    df_future_temp = df_future_temp.set_index('Date')
    df_future_temp = df_future_temp[['Région', 'TMoy (°C)', 'Année', 'month', 'day_of_week', 'day_of_year', 'week_of_year']]
    
    return df_future_temp

@st.cache_resource
def load_model_for_region(model_choice, region):
    """
    Charge un modèle Joblib spécifique pour une région donnée.
    Le modèle chargé est mis en cache par Streamlit.
    """
    model_choices = {
        "Random Forest": "RF",
        "XGBoost": "XGB",
        # "Prophet": "Prophet" # Décommentez si vous avez des modèles Prophet
    }
    model_prefix = model_choices.get(model_choice)
    if not model_prefix:
        st.error(f"Modèle '{model_choice}' non reconnu pour le chargement.")
        return None

    model_name = f"{model_prefix}_{region}.joblib"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    model_path = os.path.join(ROOT_DIR, "models_predict", model_name)

    # st.write(f"Tentative de chargement du modèle depuis : {model_path}") # Pour le débogage

    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Modèle non trouvé pour la région '{region}' ({model_name}). Veuillez vérifier les fichiers dans 'models_predict'.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle '{model_name}' : {e}")
        return None

# --- DÉBUT DE L'APPLICATION STREAMLIT ---

st.title("Simulateur de Consommation Future")
st.info(""" 
            Nous récupérons un fichier des températures futures à la FREQUENCE JOUR (voir "Pré-traitement des données") ; 
            effectuons un rapide processing pour l'aligner sur la mise en forme utilisée lors de l'entrainement 
            et la génération de nos **modèles régionaux de Régression** (RF_NomRegion.joblib, XGB_NomRegion.joblib, etc)
        """)
            
st.write("C'est maintenant  à vous de jouer pour simuler une consommation future régionale 🚀 !" \
"       ")
    
st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
st.markdown('<h5 style="text-align: center; color: #4CAF50;">🔎 Votre besoin de prévision</h5>', unsafe_allow_html=True)

# Affichage des sélecteurs de date
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Date de début", datetime.today(), key="start_date")
with col2:
    end_date = st.date_input("Date de fin", datetime.today(), key="end_date")

# Affichage du sélecteur de Régions
options = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté",
           "Bretagne", "Centre-Val de Loire", "Grand Est", "Hauts-de-France",
           "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire",
           "Provence-Alpes-Côte d'Azur", "Île-de-France"]
choix_liste = st.multiselect("Sélectionnez la(les) Région(s) :", options)

# Affichage du sélecteur de Modèle
selected_model = st.selectbox("Sélectionnez le modèle entraîné :", options= ["Random Forest", "XGBoost"])

# Récap
st.write(f"Vous avez choisi de simuler la consommation pour la période du **{start_date}** au **{end_date}**. Pour la (les) région(s) suivante(s) : **{', '.join(choix_liste) if choix_liste else 'aucune'}**")
st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)

# --- LOGIQUE PRINCIPALE ---

# Appel de la fonction cachée pour charger et prétraiter les données futures
# Ceci ne se fera qu'une seule fois au premier chargement de la page simulateur
df_future_temp = load_and_preprocess_future_data()

# Filtrage des données pour la période sélectionnée
df_filtered = df_future_temp[(df_future_temp.index >= pd.to_datetime(start_date)) & 
                             (df_future_temp.index <= pd.to_datetime(end_date))].copy() # .copy() pour éviter SettingWithCopyWarning

# Vérification de la disponibilité des données et des régions sélectionnées
if df_filtered.empty:
    st.warning("⚠️ Aucune donnée disponible pour cette période. Essayez une autre date.")
elif not choix_liste:
    st.info("Veuillez sélectionner au moins une région pour lancer la simulation.")
else:
    predictions = []
    # Afficher une barre de progression pour améliorer l'expérience utilisateur
    progress_text = "Calcul des prédictions en cours..."
    my_bar = st.progress(0, text=progress_text)

    # Itération sur les régions sélectionnées
    for i, region in enumerate(choix_liste):
        my_bar.progress((i + 1) / len(choix_liste), text=f"Prédiction pour la région : **{region}**")

        df_region = df_filtered[df_filtered["Région"] == region]

        if df_region.empty:
            st.warning(f"⚠️ Aucune donnée pour la région {region} sur cette période. Cette région sera ignorée.")
            continue

        # Sélection des variables explicatives
        features = ['TMoy (°C)', 'Année', 'month', 'day_of_week', 'day_of_year', 'week_of_year']
        X_region = df_region[features]

        # Chargement du modèle via la fonction cachée (charge une seule fois par modèle/région)
        model = load_model_for_region(selected_model, region)

        if model: # S'assurer que le modèle a été chargé avec succès
            preds = model.predict(X_region)

            # Ajout des prédictions au DataFrame
            df_region.loc[:, "Consommation_Prévue (MW)"] = preds # Utilisation de .loc pour éviter SettingWithCopyWarning
            predictions.append(df_region[["Région", "Consommation_Prévue (MW)"]])
    
    my_bar.empty() # Supprimer la barre de progression une fois les prédictions terminées

    # Affichage des résultats
    if predictions:
        df_results = pd.concat(predictions)

        st.subheader("📊 Résultats de la prédiction (avec filtres)")

        # Affichage Graphique Plotly
        fig = px.line(df_results.reset_index(), x="Date", y="Consommation_Prévue (MW)", color="Région",
              title="Visualisation graphique interactive")
        
        fig.update_layout(
            title={
                'text': "📊 Visualisation graphique interactive",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'color': '#4CAF50'
                }
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Affichage du DataFrame entier avec st.data_editor
        st.markdown('<h6 style="text-align: center; color: #4CAF50;">📊 Tableau exportable des résultats</h6>', unsafe_allow_html=True)
        st.data_editor(
            df_results.reset_index(),
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Consommation_Prévue (MW)": st.column_config.NumberColumn("Prévision (MW)"),
                "Région": st.column_config.TextColumn("Région")
            },
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True
        )