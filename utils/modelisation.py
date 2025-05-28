import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  
import datetime
import gdown
import plotly.express as px
from utils.assets_loader import load_image
from PIL import Image

#Import des bibliothèques ML
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
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

    st.markdown(""" 
                <u>Type de problème et tâche de machine learning</u> : 

                Pour rappel, notre projet s’apparente à de la **prédiction de valeurs continues dans une suite temporelle** présentant plusieurs saisonnalités.
                L'objectif est d'anticiper la demande en énergie en fonction du temps, des conditions météorologiques et d'autres facteurs exogènes. 
                
                Nous avons donc traité et fusionné l'ensembles des données exposées précédemment dans un dataset regroupant nos variables explicatives :   
                """,unsafe_allow_html=True)
    st.write("Echantillon **.sample(10)** : ")
    # --- MODIFICATION ICI ---
    if st.session_state['df'] is not None:
        st.dataframe(st.session_state['df'].sample(5))  # Accéder à df via session_state
    else:
        st.info("Veuillez charger les données en cliquant sur le bouton 'Charger et Traiter les Données' dans la section 'Lancement' pour voir un échantillon.")
    # --- FIN MODIFICATION ---

    st.markdown(""" Pour simplifier cette restitution, nous allons entraîner puis comparer nos modèles que sur la **maille horaire**. 
                La robustesse à long terme sera limité à la fin de la période de test du jeu de données. 
                """,unsafe_allow_html=True)
    st.write("---")
    st.write('#### Choix des métriques de performance 🎯')
            
    st.markdown("""La métrique **MAPE (Mean Absolute Percentage Error)** est notre métrique principale car elle est facilement interprétable et comparable avec d’autres modèles.
                Nous cherchons d’une part à pénaliser les grandes erreurs compte tenu de l’enjeu de prédiction de consommation au plus juste (**RMSE** faible), 
                tout en pouvant comparer facilement nos différents modèles sur la base de % de variation (MAPE). Enfin, la qualité globale du modèle doit aussi être élevée pour tenir compte de manière équilibrée des spécificités régionales (**Score R2**).""") 
    st.markdown("""
                Pour couvrir l’ensemble des KPI pertinents sur ce problème de régression, nous allons donc récupérer chacun des indicateurs type :

                - Erreurs absolues et relatives : **[MAE (Mean Absolute Error)](https://en.wikipedia.org/wiki/Mean_absolute_error)**, **[MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)**
                - Erreurs quadratiques : **[MSE (Erreur quadratique moyenne)](https://fr.wikipedia.org/wiki/Erreur_quadratique_moyenne)**, **[RMSE (Racine de l'erreur quadratique moyenne)](https://fr.wikipedia.org/wiki/Racine_de_l%27erreur_quadratique_moyenne)**
                - Qualité d’ajustement : **[R² Score (Coefficient de détermination)](https://fr.wikipedia.org/wiki/Coefficient_de_d%C3%A9termination)**
                """)
    st.write("---")
    st.write('#### Choix des modèles Machine Learning 🤖 ')    
    st.markdown("""
                De façon plus limitée que le rapport d'étude, nous ne présenterons ici que :
 
                - [**Random Forest**](https://fr.wikipedia.org/wiki/For%C3%AAt_d%27arbres_d%C3%A9cisionnels), [**XGBoost**](https://en.wikipedia.org/wiki/XGBoost) : deux autres modèles, plus généralistes et simples à entraîner.
                - [**Prophet**](https://facebook.github.io/prophet/docs/quick_start.html) : pour challenger notamment la détection des saisonnalités et la robustesse à long terme.  

                Ces modèles sont connus pour bien gérer les séries temporelles.
                """)
    st.write('#### Series temporelles (hold-out)⏲️, encodage, standardisation ? ”')
    st.markdown("""
                Objectif = Éviter la fuite de données. Si les données ne sont pas triées par date et que le train_test_split est aléatoire, 
                il est possible que des observations très proches temporellement se répartissent entre Train et Test faussant l'entraînement. 
                En triant par date, les données de test et en ‘splitant’ sur la fin du jeu de données, les données de Test sont vraiment "inédites" pour le modèle. 
                """)
    st.markdown(""" Avec **Random Forest**, **XGBoost** et **Prophet**, l’encodage n'apporte pas de bénéfices majeurs par rapport à une simple variable catégorielle (ex. hour ou dayofweek). 
                De même, la normalisation des données n’a pas d’impact significatif sur la performance des modèles. Nous faisons le choix de laisser les variables sans normalisation et sans transformation variables cycliques.
            """)
    st.write("---")
    st.write('#### Fine tunning - Hyperparamètres ')
    st.write("Pour une approche méthodique dans la comparaison des 2 modèles basés sur des arbres de décisions et travailler avec les meilleurs paramètrages, " \
    "nous avons utilisé **Grid Search**. Pour alléger le besoin de puissance de calcul demandés ci-après, nous laisserons laisserons les paramètres suivants : "
    "Compromis entre une exigence de mémoire acceptable pour streamlit et des score élevés des métriques observées.")
    
    code = '''
            current_model = RandomForestRegressor(
                n_estimators=7, # Nombre d'arbres dans la forêt. Plus il y en a, plus le modèle est robuste mais lent.
                max_depth=7, # Profondeur maximale de chaque arbre. Contrôle la complexité du modèle pour éviter le surapprentissage.
                min_samples_split=2, # Nombre minimum d'échantillons requis pour diviser un nœud interne.
                min_samples_leaf=1, # Nombre minimum d'échantillons requis pour qu'un nœud soit une feuille.
                random_state=42, # Graine aléatoire pour la reproductibilité des résultats.
                n_jobs=1 # Utile pour Streamlit pour la performance

            current_model = XGBRegressor(
                n_estimators=100,    # Nombre d'estimateurs (arbres)
                max_depth=3,         # Profondeur maximale de l'arbre
                learning_rate=0.05,  # Taux d'apprentissage. Réduit la contribution de chaque arbre pour rendre le modèle plus robuste.
                random_state=42,
                n_jobs=1 
            )
        '''
    st.code(code, language='python')
        ###### image ######
    img = load_image("learning_curve_xgboost.png")
    if img:
            st.image(img, caption="A titre d'exemple, la courbe d'apprentissage XGBoost, et le score RMSE en fonction du nombre d'itérations." \
            "Et où l'on voit que n_estimator = 100 peut suffire", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
        ###### image ######
    st.write("---")
    st.write('## Traitement du dataset ')

def lancement():
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

        #combined_results_df = pd.concat([
           # st.session_state['rf_metrics_per_region'],
           # st.session_state['xgb_metrics_per_region']
        #], ignore_index=True)

        #st.session_state['combined_results_df'] = combined_results_df
        #st.session_state['features_for_plot'] = st.session_state['features']
    
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

            st.subheader("➗ Moyennes des métriques d'évaluation RandomForest (Global) :")
            st.dataframe(st.session_state['rf_global_mean_metrics'].to_frame(name='Moyenne').T)

        if st.session_state['xgb_metrics_per_region'] is not None:
            st.markdown("---") 
            st.subheader("Performances du modèle XGBoost par région :")
            st.dataframe(st.session_state['xgb_metrics_per_region'].set_index('Région').style.highlight_max(axis=0, subset=['R2 Score']).highlight_min(axis=0, subset=['Mean Absolute Error', 'MAPE (%)', 'Root Mean Squared Error', 'Bias']))

            st.subheader("➗ Moyennes des métriques d'évaluation XGBoost (Global) :")
            st.dataframe(st.session_state['xgb_global_mean_metrics'].to_frame(name='Moyenne').T)

        if (
            st.session_state['rf_metrics_per_region'] is not None
            and st.session_state['xgb_metrics_per_region'] is not None
        ):
            combined_results_df = pd.concat([
                st.session_state['rf_metrics_per_region'],
                st.session_state['xgb_metrics_per_region']
            ], ignore_index=True)
            st.session_state['combined_results_df'] = combined_results_df
            st.session_state['features_for_plot'] = st.session_state['features']
    else:
        st.info("Cliquez sur 'Charger et Traiter les Données' pour commencer à visualiser et modéliser. " \
        "L'entrainement des modèles est ensuite proposé.")

@st.cache_data    
def load_process_dataset_modelisation():
    #Télécharge et prétraite les données depuis Google Drive."""
    file_id = "1wiXdpj6XHzB1eRxRbvcnsgE21ukVBvXs"  # Ton ID de fichier extrait
    url = f"https://drive.google.com/uc?id={file_id}"  # Lien de téléchargement direct
    output = "COMPILATION_CONSO_TEMP_POP_2.csv"
    # Non reduced
    # https://drive.google.com/file/d/1wiXdpj6XHzB1eRxRbvcnsgE21ukVBvXs/view?usp=sharing
    #reduce : file_id = "1dunWvb7loR5kWYZwb8BX_lwmYMP0157q" // COMPILATION_CONSO_TEMP_POP_reduced.csv
    
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Erreur lors du téléchargement du fichier : {e}")
        st.stop() 

    df= pd.read_csv(output, sep=';',low_memory=False)
    #on_bad_lines="skip", encoding="utf-8"
    
    # Filtrer les données temporelles pour se concentrer sur une période pertinente et enlever la Corse
    # Appliquez d'abord les filtres sur le DataFrame original
    df_filtered = df[(df['Date + Heure'] >= '2016-01-01') & 
                     (df['Date + Heure'] <= '2024-12-31') & 
                     (df['Région'] != 'Corse')].copy() # Utilisez .copy() pour éviter SettingWithCopyWarning

    # Convertir 'Date + Heure' en datetime et la définir comme index pour le DataFrame filtré
    df_filtered['Date + Heure'] = pd.to_datetime(df_filtered['Date + Heure'], errors='coerce')
    df_filtered = df_filtered.set_index('Date + Heure')
    df_filtered = df_filtered.sort_index()
    #df_filtered = df_filtered.sort_values(by=[df.index.name, 'Région'])

    # Le DataFrame final à utiliser sera df_filtered
    df = df_filtered.copy() # On renomme df_filtered en df pour la cohérence avec le reste du code


    # Conversion en datetime DATE pour extractions 
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    # Extraire les caractéristiques temporelles SUPPLEMENTAIRES à partir de 'DATE'
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek  # Lundi = 0, Dimanche = 6
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    #df['PlageHoraire']= df['Heure']
    df['PlageHoraire']= df['Heure'].str[:2].astype(int) # Extraction de l'heure
    df = df.drop(columns=['Date', 'Heure', 'Date - Heure'])
    df = df.sort_index() #NOUVEAU

    # Récupérer toutes les colonnes du DataFrame
    all_columns = df.columns.tolist()

    # Définir la target (à exclure des features)
    target = 'Consommation (MW)'

    # Définir une liste de colonnes à exclure (en plus de la target)
    exclude_columns = ['Région']

    # Sélectionner les features en excluant la target et les colonnes à exclure
    features = [col for col in all_columns if col != target and col not in exclude_columns]

    # Définir la proportion de l'ensemble de test
    test_size = 0.20  # Cela signifie 20% des données pour le test et 80% pour l'entraînement.
    total_observations = len(df)
    # Calculer le nombre d'observations dans l'ensemble d'entraînement (80%)
    train_observations = int(total_observations * (1 - test_size))
    # Calculer le nombre d'observations dans l'ensemble de test (20%)
    test_observations = total_observations - train_observations
    # Calculer la date de séparation en se basant sur la 80ème percentile des observations
    # Le .name récupère la valeur de l'index (qui est la date) de cette ligne
    split_date = df.iloc[train_observations - 1].name # Utilisez train_observations - 1 car iloc est 0-indexé

    # Afficher les informations de séparation pour justifier les volumes
    print(f"Volume total des entrées (observations) : {total_observations}")
    print(f"Proportion d'entraînement désirée : {int((1 - test_size) * 100)}%")
    print(f"Nombre d'observations pour l'entraînement : {train_observations} lignes")
    print(f"Proportion de test désirée : {int(test_size * 100)}%")
    print(f"Nombre d'observations pour le test : {test_observations} lignes")
    print(f"Date de séparation : {split_date}")

    return df, split_date, target, features


def RF_XGB(model_name, df, split_date, target, features):
    """
    Entraîne un modèle (RandomForest ou XGBoost) pour chaque région et évalue ses performances.
    Returns:        tuple: DataFrame des métriques par région et Series des métriques moyennes globales.
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
            
        X_train = train_region_df[features]
        y_train = train_region_df[target]
        X_test = test_region_df[features]
        y_test = test_region_df[target]

        # =========================================
        # INSTANCIATION DU MODÈLE + HYPERPARAMÈTRES
        # =========================================
        current_model = None
        if model_name == "RandomForest":
            current_model = RandomForestRegressor(
                n_estimators=7, # Nombre d'arbres dans la forêt. Plus il y en a, plus le modèle est robuste mais lent.
                max_depth=7, # Profondeur maximale de chaque arbre. Contrôle la complexité du modèle pour éviter le surapprentissage.
                min_samples_split=2, # Nombre minimum d'échantillons requis pour diviser un nœud interne.
                min_samples_leaf=1, # Nombre minimum d'échantillons requis pour qu'un nœud soit une feuille.
                random_state=42, # Graine aléatoire pour la reproductibilité des résultats.
                n_jobs=1 # Utile pour Streamlit pour la performance
            )
        elif model_name == "XGBoost":
            current_model = XGBRegressor(
                n_estimators=100,    # Nombre d'estimateurs (arbres)
                max_depth=3,         # Profondeur maximale de l'arbre
                learning_rate=0.05,  # Taux d'apprentissage. Réduit la contribution de chaque arbre pour rendre le modèle plus robuste.
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
            'Modèle': model_name, #AJOUT
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

def plot_feature_importance(combined_results_df, features):

    # Filtrer les colonnes d'importance des features
    importance_cols = [f'Importance {f}' for f in features]
    
    # Sélectionner les colonnes pertinentes pour le graphique
    df_plot = combined_results_df[['Modèle', 'Région'] + importance_cols]
    
    # Faire un melt du DataFrame pour faciliter la visualisation avec Plotly
    # Nous voulons une ligne par (Modèle, Région, Feature, Importance)
    df_melted = df_plot.melt(id_vars=['Modèle', 'Région'], 
                             value_vars=importance_cols, 
                             var_name='Feature', 
                             value_name='Importance')
    
    # Nettoyer les noms des features (enlever 'Importance ')
    df_melted['Feature'] = df_melted['Feature'].str.replace('Importance ', '')

    # Calculer l'importance moyenne par feature et par modèle pour l'affichage global
    avg_importance_df = df_melted.groupby(['Modèle', 'Feature'])['Importance'].mean().reset_index()
    
    # Créer le barplot groupé
    fig = px.bar(avg_importance_df, 
                 x='Feature', 
                 y='Importance', 
                 color='Modèle', 
                 barmode='group', # Pour grouper les barres par feature et par modèle
                 labels={'Feature': 'Feature', 'Importance': 'Importance Moyenne'},
                 height=500)
    
    fig.update_layout(xaxis_title="Features", yaxis_title="Importance Moyenne")
    fig.update_xaxes(categoryorder='total descending') # Ordonner les features par importance totale descendante
    
    return fig

def display_modeling_results_and_plots():

    st.write("---")
    st.subheader(" 🧩 Importance des Features pour les arbre de décisions")
    if st.button("Afficher l'Importance des Features"):
        if 'combined_results_df' in st.session_state and 'features_for_plot' in st.session_state:
            if st.session_state['combined_results_df'] is not None and st.session_state['features_for_plot'] is not None:
                fig = plot_feature_importance(st.session_state['combined_results_df'], st.session_state['features_for_plot'])
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ Les données nécessaires au calcul des importances ne sont pas disponibles. Veuillez entraîner les modèles d'abord.")
        else:
            st.info("💡 Cliquez d'abord sur 'Charger et Traiter les Données' puis 'Entraîner les Modèles' avant d'afficher l’importance des features.")
    st.markdown("---")
    st.header(" 🤖 Focus Prophet")
        
    st.markdown(""" Il ressort de l'entrainement une capacité moindre à capter la variabilité de notre variable cible. Témoins les métriques légèrement inférieures à celles de RandomForest et XGBoost :
                 """)
    
    # Crée un dictionnaire avec tes données
    data = {
        'Région': [
            'Auvergne-Rhône-Alpes', 'Bretagne', 'Centre-Val de Loire', 'Grand Est',
            'Hauts-de-France', 'Normandie', 'Occitanie', 'Pays de la Loire',
            'Provence-Alpes-Côte d\'Azur', 'Île-de-France', 'Nouvelle-Aquitaine',
            'Bourgogne-Franche-Comté'
        ],
        'MSE': [618347.408440, 71293.294483, 57860.196918, 273224.467675, 496277.620232, 107251.190415, 408808.501023, 316904.940208, 196585.834993, 780407.570747, 611839.872156, 48594.971564],
        'RMSE': [786.350690, 267.008042, 240.541466, 522.708779, 704.469744, 327.492275, 639.381342, 562.943106, 443.380012, 883.406798, 782.201938, 220.442672],
        'MAE': [628.756138, 207.169993, 192.209835, 426.653469, 571.402178, 237.517623, 512.350058, 424.050043, 334.635441, 708.092105, 602.397156, 173.843995],
        'MAPE': [9.613186, 9.489713, 10.549336, 9.797167, 10.667154, 8.937283, 12.240458, 16.976135, 7.952227, 10.248624, 13.203451, 8.578102],
        'Bias': [-397.169287, 53.035106, -106.187905, -279.899089, -517.764597, -87.048697, -386.316433, -5.547958, -135.732437, -550.626619, -449.839053, -64.854448],
        'R^2': [0.638707, 0.702248, 0.720858, 0.610560, 0.409639, 0.669755, 0.466369, 0.391215, 0.629417, 0.765376, 0.507729, 0.781519]
    }
    df = pd.DataFrame(data)
    df_rounded = df.round(2)
    col1, col2 = st.columns([2, 1]) # La première colonne est deux fois plus large
    with col1:
        st.write("Voici les métriques de performance pour chaque région :")
        st.dataframe(df_rounded, hide_index=True)
    with col2:
        st.write("Moyenne Globale :")
        # Charge l'image
        try:
            img = load_image("KPI_prophet.png") # Assurez-vous que l'image est dans le même dossier ou spécifiez le chemin complet
            st.image(img, caption="On notera un BIAS moyen négatif qui sous-entend une sous-estimation contrairement aux arbres de décision qui surestiment", use_container_width=True)
        except FileNotFoundError:
            st.warning("❌ L’image est introuvable. Veuillez vérifier le chemin d'accès.")
    st.markdown("""La bonne captation par Prophet des saisonnalités est ici illustrée: tendance globale, hebdomadaire, annuelle ou journalière.)
                """)
    ###### image ######
    img1 = load_image("saisonnalités1.png")
    img2 = load_image("saisonnalité2.png")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1)
    with col2:
        st.image(img2)
    ###### image ######
    st.markdown("---")
    st.markdown("""    
                """)
    st.markdown("---")

def conclusion():
    ##################
    #INTERPRETATION FEATURES
    st.write("## ✅ Bilan ")
    st.markdown(""" Les modèles RandomForest et XGBoost s’accordent sur l’importance déterminante de la température moyenne et maximale pour prédire la consommation électrique, 
                reflétant l’impact du climat sur la demande (chauffage/climatisation). La plage horaire est également clé, capturant les variations journalières typiques. 
                Les variables calendaires jouent un rôle secondaire (XGBoost y est néanmoins plus sensible ). Tandis que la population ne variant pas à cette échelle de temps, a peu d’influence 
                """)

    st.markdown(""" 
                Comparatif des modèles :

                **XGBoost** surpasse Random Forest avec un R² supérieur de 0,07, indiquant qu’il explique significativement mieux la variabilité de la consommation électrique.
                Soit une meilleure capacité de XGBoost à capturer les variations fines et les non-linéarités. Ses erreurs (RMSE, MAE, MAPE) sont également plus faibles, traduisant des prédictions plus précises et un biais réduit (<u>moins de surestimation</u>.
                A noter : 
                - la différence de MAPE correspond à une amélioration d’environ 10% de la précision relative au profit de XG Boost
                - XG Boost porte bien son nom en ce qu'il est également plus rapide à entrainer (sur ce jeu de données).

                Cependant, Random Forest reste un modèle robuste, plus simple à paramétrer et interpréter, pouvant être préféré selon les contraintes opérationnelles. 
                Le choix dépendra donc du compromis entre performance fine et simplicité d’usage.

                """,unsafe_allow_html=True)