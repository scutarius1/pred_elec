import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  
import datetime
import gdown

#Import des bibliothèques ML
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from prophet import Prophet


def intro():
    # ==========================================================================
    # Initialisation des clés de session_state pour tous les modèles et données
    # ==========================================================================
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        st.session_state['split_date'] = None
        st.session_state['target'] = None
        st.session_state['features'] = None
        st.session_state['rf_metrics_per_region'] = None
        st.session_state['rf_global_mean_metrics'] = None
        st.session_state['xgb_metrics_per_region'] = None
        st.session_state['xgb_global_mean_metrics'] = None

    st.write('## Classification du problème 📂')
    st.write("");st.write("") 

    st.markdown(""" <u>Type de problème et tâche de machine learning</u> : 
            Notre projet s’apparente à de la **prédiction de valeurs continues dans une suite temporelle** présentant plusieurs saisonnalités.
                L'objectif est d'anticiper la demande en énergie en fonction du temps, des conditions météorologiques et d'autres facteurs exogènes.
                """,unsafe_allow_html=True)
    st.write('#### Choix des métriques de performance 🎯')
            
    st.markdown("""La métrique **MAPE (Mean Absolute Percentage Error)** est notre métrique principale car elle est facilement interprétable et comparable avec d’autres modèles.
                Nous cherchons d’une part à pénaliser les grandes erreurs compte tenu de l’enjeu de prédiction de consommation au plus juste (**RMSE** faible), 
                tout en pouvant comparer facilement nos différents modèles sur la base de % de variation (MAPE). Enfin, la qualité globale du modèle doit aussi être élevée pour tenir compte de manière équilibrée des spécificités régionales (**Score R2**).""") 
    st.markdown("""
                Pour couvrir l’ensemble des KPI pertinents sur ce problème de régression nous allons donc récupérer chacun des indicateurs type :
                
                - Erreurs absolues et relatives (**MAE, MAPE**)
                - Erreurs quadratiques (**MSE, RMSE**)
                - Qualité d’ajustement (**R² Score**)
                """)
    st.write('#### Choix des modèles Machine Learning 🤖 ')
    st.markdown("""
                De façon plus limitée que le rapport d'étude, nous ne présenterons ici que :

                - <span style="color:blue;">**Prophet**</span> : pour challenger notamment la détection des saisonnalités et la robustesse à long terme.
                - <span style="color:blue;">**RandomForest**</span>, <span style="color:blue;">**XG Boost**</span> : 2 autres modèles, plus généralistes et simples à entraîner

                Ces modèles sont connus pour bien gérer les séries temporelles.
                """, unsafe_allow_html=True)
    st.write('#### Series temporelles ⏲️ ? Split= “Hold Out”')
    st.markdown("""
                Objectif = Éviter la fuite de données. Si les données ne sont pas triées par date et que le train_test_split est aléatoire, 
                il est possible que des observations très proches temporellement se répartissent entre Train et Test faussant l'entraînement. 
                En triant par date, les données de test et en ‘splitant’ sur la fin du jeu de données, les données de Test sont vraiment "inédites" pour le modèle. 
                """)
    st.markdown(""" Avec **Random Forest**, **XGBoost** et **Prophet**, l’encodage n'apporte pas de bénéfices majeurs par rapport à une simple variable catégorielle (ex. hour ou dayofweek). 
                De même, la normalisation des données n’a pas d’impact significatif sur la performance des modèles. Nous faisons le choix de laisser les variables sans normalisation et sans transformation variables cycliques.
            """)
    
    # Bouton pour lancer le traitement des données et l'affichage
    if st.button("Charger et Traiter les Données"):
        with st.spinner("Chargement et traitement des données en cours..."):
            # Assignation directe à 'df' et stockage sous la clé 'df' dans session_state
            df, split_date, target, features = load_process_dataset_modelisation() 
            
            st.session_state['df'] = df
            st.session_state['split_date'] = split_date
            st.session_state['target'] = target
            st.session_state['features'] = features
 
        st.success("Données chargées et prétraitées avec succès !")
        
        st.subheader("Aperçu du DataFrame après prétraitement :")
        st.dataframe(st.session_state['df'].sample(10)) 
        
        st.subheader("Paramètres de modélisation :")
        st.write(f"**Date de séparation (split_date) :** {st.session_state['split_date']}")
        st.write(f"**Variable cible (target) :** `{st.session_state['target']}`")
        st.write(f"**Variables explicatives (features) :**")
        st.write(st.session_state['features'])

    # ======================================================================
    # Nouveau bouton pour entraîner RF et XGBoost ensemble
    # ======================================================================
    # Vérifie si les données sont chargées avant d'afficher ce bouton
    if st.session_state['df'] is not None: 
        if st.button("Lancer l'entraînement et l'évaluation des modèles (RF & XGBoost)"):
            # Récupérer les données de session_state pour les passer à la fonction RF_XGB
            df = st.session_state['df'] 
            split_date = st.session_state['split_date']
            target = st.session_state['target']
            features = st.session_state['features']

            # --- Entraînement et évaluation RandomForest ---
            with st.spinner("Entraînement et évaluation du modèle RandomForest en cours..."):
                # Appel de la fonction renommée RF_XGB pour RandomForest
                rf_metrics_df_per_region, rf_global_mean_metrics = RF_XGB("RandomForest", df, split_date, target, features)
                
                st.session_state['rf_metrics_per_region'] = rf_metrics_df_per_region
                st.session_state['rf_global_mean_metrics'] = rf_global_mean_metrics
            st.success("Évaluation RandomForest terminée !")
            
            st.markdown("---") # Séparateur visuel entre les modèles

            # --- Entraînement et évaluation XGBoost ---
            with st.spinner("Entraînement et évaluation du modèle XGBoost en cours..."):
                # Appel de la fonction renommée RF_XGB pour XGBoost
                xgb_metrics_df_per_region, xgb_global_mean_metrics = RF_XGB("XGBoost", df, split_date, target, features)

                st.session_state['xgb_metrics_per_region'] = xgb_metrics_df_per_region
                st.session_state['xgb_global_mean_metrics'] = xgb_global_mean_metrics
            st.success("Évaluation XGBoost terminée !")
        
        # ======================================================================
        # Affichage séparé des résultats RF et XGBoost
        # Ces blocs s'exécutent si les résultats sont présents dans session_state
        # ======================================================================
        if st.session_state['rf_metrics_per_region'] is not None:
            st.subheader("Performances du modèle RandomForest par région :")
            st.dataframe(st.session_state['rf_metrics_per_region'].set_index('Région').style.highlight_max(axis=0, subset=['R2 Score']).highlight_min(axis=0, subset=['Mean Absolute Error', 'MAPE (%)', 'Root Mean Squared Error', 'Bias']))

            st.subheader("Moyennes des métriques d'évaluation RandomForest (Global) :")
            st.dataframe(st.session_state['rf_global_mean_metrics'].to_frame(name='Moyenne').T)

        if st.session_state['xgb_metrics_per_region'] is not None:
            st.markdown("---") 
            st.subheader("Performances du modèle XGBoost par région :")
            st.dataframe(st.session_state['xgb_metrics_per_region'].set_index('Région').style.highlight_max(axis=0, subset=['R2 Score']).highlight_min(axis=0, subset=['Mean Absolute Error', 'MAPE (%)', 'Root Mean Squared Error', 'Bias']))

            st.subheader("Moyennes des métriques d'évaluation XGBoost (Global) :")
            st.dataframe(st.session_state['xgb_global_mean_metrics'].to_frame(name='Moyenne').T)
    else:
        st.info("Cliquez sur 'Charger et Traiter les Données' pour commencer à visualiser et modéliser. " \
        "L'entrainement des modèles est ensuite proposé.")

@st.cache_data    
def load_process_dataset_modelisation():
    #Télécharge et prétraite les données depuis Google Drive."""
    file_id = "1wiXdpj6XHzB1eRxRbvcnsgE21ukVBvXs"  # Ton ID de fichier extrait
    url = f"https://drive.google.com/uc?id={file_id}"  # Lien de téléchargement direct
    output = "COMPILATION_CONSO_TEMP_POP_2.csv"
    
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Erreur lors du téléchargement du fichier : {e}")
        st.stop() 

    df= pd.read_csv(output, sep=';', on_bad_lines="skip", encoding="utf-8",low_memory=False)
    
    # Filtrer les données temporelles pour se concentrer sur une période pertinente et enlever la Corse
    df_filtered = df[(df['Date + Heure'] >= '2016-01-01') & 
                    (df['Date + Heure'] <= '2024-12-31')& (df['Région']!='Corse')] 

    # Identifier les lignes avec -0.00 dans les colonnes spécifiques
    cols_to_check = ['TMoy (°C)', 'TMin (°C)', 'TMax (°C)']
    neg_zero_mask = (df_filtered[cols_to_check] == -0.00)

    # Appliquer la correction uniquement aux valeurs identifiées en utilisant .loc
    df_filtered.loc[:, cols_to_check] = df_filtered.loc[:, cols_to_check].mask(neg_zero_mask, 0.00)

    # Remettre la colonne 'Date + Heure' en index
    df = df_filtered.set_index('Date + Heure')
    df.index = pd.to_datetime(df.index)

    # Conversion en datetime DATE pour extractions 
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    # Extraire les caractéristiques temporelles SUPPLEMENTAIRES à partir de 'DATE'
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek  # Lundi = 0, Dimanche = 6
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['PlageHoraire']= df['Heure'].str[:2].astype(int) # Extraction de l'heure
    df = df.drop(columns=['Date', 'Heure', 'Date - Heure'])

    # Récupérer toutes les colonnes du DataFrame
    all_columns = df.columns.tolist()

    # Définir la target (à exclure des features)
    target = 'Consommation (MW)'

    # Définir une liste de colonnes à exclure (en plus de la target)
    exclude_columns = ['Région']

    # Sélectionner les features en excluant la target et les colonnes à exclure
    features = [col for col in all_columns if col != target and col not in exclude_columns]

    # Définir la proportion de l'ensemble de test
    test_size = 0.20  # Pour 20%
    # Calculer la date de séparation
    split_date = df.iloc[int(len(df) * (1 - test_size))].name
    # Afficher la date de séparation
    print(f"Date de séparation pour {int(test_size * 100)}% de test : {split_date}")

    return df, split_date, target, features


def RF_XGB(model_name, df, split_date, target, features):
    """
    Entraîne un modèle (RandomForest ou XGBoost) pour chaque région et évalue ses performances.
    Args:
        model_name (str): Nom du modèle à entraîner ("RandomForest" ou "XGBoost").
        df (pd.DataFrame): DataFrame contenant les données prétraitées.
        split_date (datetime): Date de séparation pour les ensembles d'entraînement/test.
        target (str): Nom de la colonne cible.
        features (list): Liste des noms des colonnes explicatives.
    Returns:
        tuple: DataFrame des métriques par région et Series des métriques moyennes globales.
    """
    results = []
    regions = df['Région'].unique()

    # Diviser le DataFrame global en ensembles d'entraînement et de test une seule fois
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    for region in regions:
        st.write(f"Entraînement et évaluation du modèle **{model_name}** pour la région : **{region}**") 
        
        # Filtrer les données par région à partir des ensembles déjà splittés
        train_region_df = train_df[train_df['Région'] == region]
        test_region_df = test_df[test_df['Région'] == region]

        if len(train_region_df) == 0 or len(test_region_df) == 0:
            st.warning(f"Pas assez de données pour la région {region} pour le modèle {model_name}. Skipping.")
            continue
            
        X_train = train_region_df[features]
        y_train = train_region_df[target]
        X_test = test_region_df[features]
        y_test = test_region_df[target]

        # ==============================================================================
        # INSTANCIATION DU MODÈLE EN FONCTION DE SON NOM, AVEC HYPERPARAMÈTRES
        # ==============================================================================
        current_model = None
        if model_name == "RandomForest":
            current_model = RandomForestRegressor(
                n_estimators=10, 
                max_depth=10, 
                min_samples_split=2, 
                min_samples_leaf=1, 
                random_state=42,
                n_jobs=1 # Utile pour Streamlit pour la performance
            )
        elif model_name == "XGBoost":
            current_model = XGBRegressor(
                n_estimators=300,             # Nombre d'estimateurs (arbres)
                max_depth=3,                  # Profondeur maximale de l'arbre
                learning_rate=0.05,            # Taux d'apprentissage
                random_state=42,
                n_jobs=1    
            )
        else:
            st.error(f"Modèle non supporté pour l'entraînement : {model_name}")
            continue 
        
        current_model.fit(X_train, y_train)

        predictions = current_model.predict(X_test)
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100 
        r2 = r2_score(y_test, predictions)
        
        # Moyennes des valeurs réelles et prédites
        mean_y_test = np.mean(y_test)
        mean_y_pred = np.mean(predictions) 
        # Calcul du Bias
        bias = mean_y_pred - mean_y_test

        result = {
            'Région': region,
            'Moy y_test': mean_y_test,
            'Moy y_pred': mean_y_pred,
            'Bias': bias,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae,
            'MAPE (%)': mape, 
            'R2 Score': r2
        }
        
        # Ajouter les importances des features si le modèle le supporte
        if hasattr(current_model, 'feature_importances_'):
            for feature, importance in zip(X_train.columns, current_model.feature_importances_): 
                result[f'Importance {feature}'] = importance
        
        results.append(result)
        
    results_df = pd.DataFrame(results)
    
    # Calculer la moyenne des métriques globales (pour toutes les régions)
    numeric_metrics_cols = ['R2 Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'MAPE (%)', 'Bias']
    mean_metrics = results_df[numeric_metrics_cols].mean()

    return results_df, mean_metrics