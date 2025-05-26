import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  
import datetime
import gdown
import plotly.graph_objects as go
import plotly.express as px

#Import des bibliothèques ML
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from prophet import Prophet

def plot_date_order(df_to_check, title="Vérification de l'ordre temporel du DataFrame"):
    """
    Crée un nuage de points pour vérifier l'ordre temporel d'un DataFrame.
    L'axe X est l'index (date), l'axe Y est la position ordinale.
    """
    if not isinstance(df_to_check.index, pd.DatetimeIndex):
        st.warning("L'index du DataFrame n'est pas de type DatetimeIndex. La vérification peut être imprécise.")
        return

    # Créer une colonne pour la position ordinale
    df_to_check['ordinal_position'] = range(len(df_to_check))

    fig = px.scatter(df_to_check,
                     x=df_to_check.index,
                     y='ordinal_position',
                     title=title,
                     labels={'x': 'Date de l\'Index', 'ordinal_position': 'Position Ordinale dans le DataFrame'},
                     template="plotly_white")
    
    if 'split_date' in st.session_state and st.session_state['split_date'] is not None:
        try:
            split_date_dt = pd.to_datetime(st.session_state['split_date'])
            split_idx_pos = df_to_check.index.get_loc(split_date_dt, method='nearest')
            if isinstance(split_idx_pos, slice):
                split_idx_pos = split_idx_pos.start if split_idx_pos.start is not None else 0

            fig.add_vline(x=split_date_dt, line_width=2, line_dash="dash", line_color="red",
                          annotation_text=f"Split Date: {split_date_dt.strftime('%Y-%m-%d %H:%M')}",
                          annotation_position="top right")
            
            if split_date_dt in df_to_check.index:
                position_at_split = df_to_check.loc[split_date_dt]['ordinal_position']
                if isinstance(position_at_split, pd.Series):
                    position_at_split = position_at_split.iloc[0]
                
                fig.add_hline(y=position_at_split, 
                              line_width=2, line_dash="dash", line_color="blue",
                              annotation_text=f"Position: {split_idx_pos}",
                              annotation_position="bottom right")

        except Exception as e:
            st.warning(f"Impossible d'ajouter la split_date au graphique: {e}")

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    df_to_check.drop(columns=['ordinal_position'], inplace=True, errors='ignore')

def plot_target_over_time_with_split(df_to_check, target_col, split_date, title="Visualisation de la série temporelle et du point de séparation"):
    """
    Trace la variable cible sur l'index de temps et marque la date de séparation.
    Affiche une portion des données pour une meilleure lisibilité.
    """
    if not isinstance(df_to_check.index, pd.DatetimeIndex):
        st.warning("L'index du DataFrame n'est pas de type DatetimeIndex.")
        return

    # Pour éviter de tracer un DataFrame trop grand (lent), on peut prendre un échantillon ou une période limitée.
    # Par exemple, les dernières 3 années ou un échantillon stratifié
    # Si le DataFrame est très grand, prenez un échantillon représentatif ou une sous-période
    df_sample = df_to_check.tail(365 * 24 * 3) # Ex: les 3 dernières années d'heures

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_sample.index, y=df_sample[target_col], mode='lines', name=target_col))

    fig.update_layout(title=title,
                      xaxis_title="Date",
                      yaxis_title=target_col,
                      template="plotly_white")

    # Ajouter une ligne verticale pour marquer la split_date
    if split_date:
        fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red",
                      annotation_text=f"Date de Séparation: {split_date.strftime('%Y-%m-%d %H:%M')}",
                      annotation_position="top right")

    st.plotly_chart(fig, use_container_width=True)

# Exemple d'utilisation dans votre `lancement()`:
# Juste après que st.session_state['df'] et split_date soient mis à jour
# if st.session_state['df'] is not None and st.session_state['split_date'] is not None:
#     st.subheader("Visualisation de la variable cible par rapport au temps")
#     plot_target_over_time_with_split(st.session_state['df'], 
#                                       st.session_state['target'], 
#                                       st.session_state['split_date'])

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
    st.write('#### Series temporelles (hold-out)⏲️, encodage, standardisation ? ”')
    st.markdown("""
                Objectif = Éviter la fuite de données. Si les données ne sont pas triées par date et que le train_test_split est aléatoire, 
                il est possible que des observations très proches temporellement se répartissent entre Train et Test faussant l'entraînement. 
                En triant par date, les données de test et en ‘splitant’ sur la fin du jeu de données, les données de Test sont vraiment "inédites" pour le modèle. 
                """)
    st.markdown(""" Avec **Random Forest**, **XGBoost** et **Prophet**, l’encodage n'apporte pas de bénéfices majeurs par rapport à une simple variable catégorielle (ex. hour ou dayofweek). 
                De même, la normalisation des données n’a pas d’impact significatif sur la performance des modèles. Nous faisons le choix de laisser les variables sans normalisation et sans transformation variables cycliques.
            """)
    
    st.write('#### Fine tunning - Hyperparamètres ')
    st.write("Pour une approche plus méthodique dans la comparaison des 2 modèles basés sur des arbres de décisions et travailler avec les meilleurs paramètrages, " \
    "nous avons utilisé **Grid Search**. Pour alléger le besoin de puissance de calcul demandés ci-après, nous laisserons laisserons les paramètres suivants."
    "C'est un compromis entre une exigence de mémoire acceptable pour ce projet streamlit en ligne et des score élevés des métriques observées ")
    
    code = '''
            current_model = RandomForestRegressor(
                n_estimators=8, # Nombre d'arbres dans la forêt. Plus il y en a, plus le modèle est robuste mais lent.
                max_depth=8, # Profondeur maximale de chaque arbre. Contrôle la complexité du modèle pour éviter le surapprentissage.
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

# --- AJOUT DES VISUALISATIONS ICI AVEC LA BONNE INDENTATION ---
        # Ces blocs s'exécutent car nous sommes déjà dans le if st.button(...)
        # et st.session_state['df'] vient d'être défini.
        st.markdown("---")
        st.subheader("Graphique 1: Vérification de l'ordre temporel (Index vs. Position Ordinale)")
        # Assurez-vous que plot_date_order et plot_target_over_time_with_split sont définies dans votre script
        # avant d'être appelées ici.
        plot_date_order(st.session_state['df'], title="Vérification de l'ordre temporel du DataFrame après prétraitement")

        st.markdown("---")
        st.subheader("Graphique 2: Visualisation de la Variable Cible et de la Date de Séparation")
        plot_target_over_time_with_split(st.session_state['df'], 
                                          st.session_state['target'], 
                                          st.session_state['split_date'])
        # --- FIN DE L'AJOUT DES VISUALISATIONS ---

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
    #test_size = 0.20  # Pour 20%
    # Calculer la date de séparation
    #split_date = df.iloc[int(len(df) * (1 - test_size))].name
    # Afficher la date de séparation
    #print(f"Date de séparation pour {int(test_size * 100)}% de test : {split_date}")
    #split_date = "2021-09-06 00:00:00"

        # Définir la proportion de l'ensemble de test
    test_size = 0.20  # Cela signifie 20% des données pour le test, et donc 80% pour l'entraînement.

    # Calculer le nombre d'observations total
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
                n_estimators=8, # Nombre d'arbres dans la forêt. Plus il y en a, plus le modèle est robuste mais lent.
                max_depth=8, # Profondeur maximale de chaque arbre. Contrôle la complexité du modèle pour éviter le surapprentissage.
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



