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
    if 'processed_df' not in st.session_state:
        st.session_state['processed_df'] = None
        st.session_state['split_date'] = None
        st.session_state['target'] = None
        st.session_state['features'] = None
        st.session_state['df_prophet_ready'] = None
        st.session_state['rf_metrics_per_region'] = None
        st.session_state['rf_global_mean_metrics'] = None

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
            # Décomposez le tuple retourné en variables séparées
            processed_df, split_date, target, features = load_process_dataset_modelisation() 
            # CES LIGNES SONT CRUCIALES POUR LE STOCKAGE
            st.session_state['processed_df'] = processed_df
            st.session_state['split_date'] = split_date
            st.session_state['target'] = target
            st.session_state['features'] = features
            st.session_state['df_prophet_ready'] = processed_df.reset_index().rename(columns={'Date + Heure': 'ds', 'Consommation (MW)': 'y'})
    # ...
        st.success("Données chargées et prétraitées avec succès !")
        st.subheader("Aperçu du DataFrame après prétraitement :")
        st.dataframe(processed_df.sample(10)) 
        st.subheader("Paramètres de modélisation :")
        st.write(f"**Date de séparation (split_date) :** {split_date}")
        st.write(f"**Variable cible (target) :** `{target}`")
        st.write(f"**Variables explicatives (features) :**")
        st.write(features)


    if st.button("Lancer l'entraînement et l'évaluation RandomForest"):
        if 'processed_df' not in st.session_state:
            st.warning("Veuillez d'abord charger et traiter les données.")
        else:
            processed_df = st.session_state['processed_df']
            split_date = st.session_state['split_date']
            target = st.session_state['target']
            features = st.session_state['features']

            with st.spinner("Entraînement et évaluation des modèles RandomForest en cours..."):
                # Appel de la nouvelle fonction adaptée
                rf_metrics_df_per_region, rf_global_mean_metrics = random_forest (processed_df, split_date, target, features)

            st.success("Évaluation RandomForest terminée !")

            st.subheader("Performances du modèle RandomForest par région :")
            # Affichage du DataFrame avec style
            st.dataframe(rf_metrics_df_per_region.set_index('Région').style.highlight_max(axis=0, subset=['R2 Score']).highlight_min(axis=0, subset=['Mean Absolute Error', 'MAPE (%)', 'Root Mean Squared Error', 'Bias']))

            st.subheader("Moyennes des métriques d'évaluation (Global) :")
            # Affichage des moyennes globales dans un petit tableau ou en texte
            st.dataframe(rf_global_mean_metrics.to_frame(name='Moyenne').T) # Pour afficher comme un tableau horizontal
            # Ou en texte :
            # for metric, value in rf_global_mean_metrics.items():
            #     st.write(f"**{metric}**: {value:.4f}")

            # Stocker les résultats si vous voulez les utiliser ailleurs
            st.session_state['rf_metrics_per_region'] = rf_metrics_df_per_region
            st.session_state['rf_global_mean_metrics'] = rf_global_mean_metrics

    
def load_process_dataset_modelisation():
    #Télécharge et prétraite les données depuis Google Drive."""
    file_id = "1wiXdpj6XHzB1eRxRbvcnsgE21ukVBvXs"  # Ton ID de fichier extrait
    url = f"https://drive.google.com/uc?id={file_id}"  # Lien de téléchargement direct
    output = "COMPILATION_CONSO_TEMP_POP_2.csv"
    gdown.download(url, output, quiet=False)
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



def random_forest (df, split_date, target, features):
    """
    Entraîne un modèle RandomForest pour chaque région, évalue ses performances
    et retourne un DataFrame des métriques, y compris les importances des features.
    """
    results = [] # Pour stocker les métriques par région
    # all_y_test_RF et all_y_pred_RF peuvent être gérés ici si nécessaire pour des agrégations globales,
    # mais pour le tableau par région, ils ne sont pas directement nécessaires au retour.

    regions = df['Région'].unique()

    # NOTE: Dans votre script original, vous refiltrez df_region = df[df['Région'] == region]
    # puis vous faites le split sur df_region.index.
    # On peut optimiser en splittant le df global une fois, puis en filtrant par région.
    # Ou garder votre logique si elle est plus claire pour vous.
    # Je vais garder la logique que j'avais proposée qui splittait globalement puis filtrait,
    # car cela évite de recalculer split_date à chaque itération.

    # Diviser le DataFrame global en ensembles d'entraînement et de test
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    for region in regions:
        # Affichage pour Streamlit, remplace votre print()
        st.write(f"Entraînement et évaluation pour la région : **{region}**") 
        
        # Filtrer les données par région à partir des ensembles déjà splittés
        train_region_df = train_df[train_df['Région'] == region]
        test_region_df = test_df[test_df['Région'] == region]

        if len(train_region_df) == 0 or len(test_region_df) == 0:
            st.warning(f"Pas assez de données pour la région {region}. Skipping.")
            continue
            
        # Préparer les données pour le modèle
        X_train = train_region_df[features]
        y_train = train_region_df[target]
        X_test = test_region_df[features]
        y_test = test_region_df[target]

        # Initialiser et entraîner le modèle RandomForest avec VOS hyperparamètres
        model = RandomForestRegressor(
            n_estimators=10, 
            max_depth=10, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            random_state=42,
            n_jobs=-1 # Utile pour Streamlit pour la performance
        )
        model.fit(X_train, y_train)

        # Faire des prédictions
        predictions = model.predict(X_test)

        # all_y_test_RF et all_y_pred_RF pourraient être agrégés ici si vous avez besoin des métriques globales
        # pour TOUTES les régions combinées à la fin. Pour le tableau par région, ce n'est pas nécessaire.
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100 # Multiplier par 100 pour un %
        r2 = r2_score(y_test, predictions)
        
        # Moyennes des valeurs réelles et prédites
        mean_y_test = np.mean(y_test)
        mean_y_pred = np.mean(predictions) # C'est 'predictions' ici
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
            'MAPE (%)': mape, # Renommé pour correspondre au %
            'R2 Score': r2
        }
        
        # Ajouter les importances des features
        for feature, importance in zip(X_train.columns, model.feature_importances_): # Utilisez X_train.columns
            result[f'Importance {feature}'] = importance
        
        results.append(result)
        
        # Les print() sont remplacés par st.write() si vous voulez des affichages intermédiaires par région
        # Mais un tableau final est généralement préféré dans Streamlit.

    results_df = pd.DataFrame(results)
    
    # Calculer la moyenne des métriques globales (pour toutes les régions)
    # Assurez-vous que les colonnes existent et sont numériques
    numeric_metrics_cols = ['R2 Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'MAPE (%)', 'Bias']
    mean_metrics = results_df[numeric_metrics_cols].mean()

    return results_df, mean_metrics # Retourne le DF par région et les moyennes globales
    