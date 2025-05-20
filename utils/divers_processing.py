#################################
# ⚙️ DATASET TEMPERATURES FUTURES    ⚙️#
#################################

from PIL import Image
import streamlit as st
from utils.assets_loader import load_image

def drias():
    st.header("🌡️ Données Climatiques Futures pour la Simulation")

    st.markdown("""
    Pour tester notre modèle avec des **données futures réalistes**, nous devons récupérer une simulation des **températures régionales**, 
    à la **maille fine** et sur **plusieurs années**. Après exploration, nous avons identifié sur le portail de la [**DRIAS**](https://www.drias-climat.fr) un catalogue de données de **simulations climatiques** 
    au format [**NetCDF**](https://fr.wikipedia.org/wiki/NetCDF) pouvant répondre à ce besoin.
    """)

    st.markdown("""
    <div style="background-color: #182C43 ; padding: 1em; border-radius: 5px; border-left: 5px solid #91caff;">
    <h4>⚠️ Avertissement</h4>
    <p><strong>Cet exercice de simulation des températures futures vise à illustrer des méthodes de traitement de données et nous permettre de mobiliser notre modèle de prédiction pour simuler une future consommation électrique.</strong></p>
    <ul>
    <li>Les <strong>prévisions climatiques</strong> sont issues de <strong>modèles complexes</strong> qui comportent des <strong>incertitudes</strong> importantes.</li>
    <li>Les <strong>données consolidées datent de 2020</strong>, ce qui peut introduire un <strong>biais temporel</strong>.</li>
    <li>Pour associer les données climatiques à nos <strong>régions administratives</strong>, nous avons utilisé une méthode triviale de <strong>regroupement par encadrement</strong> des <strong>coordonnées géographiques</strong>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)  
    st.write("---")

    st.markdown(" #### Choix d’un Scénario d’émission de Co2") 
    st.markdown("""
    - RCP2.6 : Scénario avec une politique climatique visant à faire baisser les concentrations en CO2
    - **RCP4.5** : Scénario avec une politique climatique visant à stabiliser les concentrations en CO2 (Scénario retenu)
    - RCP8.5 : Scénario sans politique climatique
    """)

    ###### image ######
    img = load_image("scenarios_climatiques_regions_france.png")
    if img:
            st.image(img, caption="Modèles possibles", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################

    ###### image ######
    img = load_image("Carto_safran.png")
    if img:
            st.image(img, caption="Carte des données grille SAFRAN - plus de 8000 points sur le territoire", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################


    col1, col2 = st.columns([1, 1.2])

    # COL NETCDF
    with col1:
        st.markdown("#### 🧩 De .NetCDF à un .csv exploitable...")

        code = '''
    import xarray as xr
    import pandas as pd

    # Charger le fichier netCDF
    nc_file_2 = "Explo/tasAdjust_France_CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63_rcp4.5_METEO-FRANCE_ADAMONT-France_SAFRAN_day_20240101-20351231.nc"
    ds2 = xr.open_dataset(nc_file_2)

    # Convertir en DataFrame
    df2 = ds2.to_dataframe()

    # Sauvegarder en CSV
    csv_file = "output2.csv"
    df2.to_csv(csv_file)

    print(f"Fichier CSV enregistré sous {csv_file}")
    '''
        st.code(code, language='python')

    # COL RECOUPEMENT
    with col2:
        with st.container():
            st.write(""), st.write(""), st.write(""),st.write("")
            st.write("Nous passons d’un fichier : netCDF > df de près de **84 M de lignes** (😅)")

            ###### image ######
            img = load_image("Netcdf_to_df.png")
            if img:
                # On ajoute un encadré visuel avec `st.image` dans un `st.container`
                st.image(img, caption="Nous ferons bon usage des coordonnées géographiques et températures - tasAdjust", use_container_width=True)
            else:
                st.warning("❌ L’image 'Netcdf_to_df.png' est introuvable dans le dossier `pictures/`.")
            ###### image ######

    st.write("")

    col3, col4 = st.columns([1, 1.2])

    with col3:
        st.write("Croisement de l’ensemble des données avec définition des zones géographiques (coordonnées lat/long) pour chaque région...")

        code = '''
    # Définition des zones géographiques sous forme de DataFrame
    regions = [
        ("Auvergne-Rhône-Alpes", 44, 46.5, 2, 7),
        ("Bourgogne-Franche-Comté", 46.5, 48.5, 2, 6.5),
        ("Bretagne", 47, 49, -5.5, -1),
        ("Centre-Val de Loire", 46, 48, 0, 3),
        ("Grand Est", 48, 50, 4, 8),
        ("Hauts-de-France", 49, 51, 1, 4),
        ("Normandie", 48, 50, -1.8, 1.7),
        ("Nouvelle-Aquitaine", 43, 47, -1.8, 1.5),
        ("Occitanie", 42, 45, -1, 4),
        ("Pays de la Loire", 46, 48, -2, 1),
        ("Provence-Alpes-Côte d'Azur", 43, 45, 4, 7.5),
        ("Île-de-France", 48, 49, 2, 3),
    ]

    df_regions = pd.DataFrame(regions, columns=["region", "lat_min", "lat_max", "lon_min", "lon_max"])
    df_regions
        '''
        st.code(code, language='python')

    with col4:
        with st.container():
            st.write("... et aboutir à un nouveau dataset très léger de prévisions de températures régionales journalières 2024 à 2035 ✌️")
            img = load_image("dataset_temperatures_futures.png")
            if img:
                st.image(img, caption="Fichier `Future_temps.csv` (prévisions 2024–2035)", use_container_width=True)
            else:
                st.warning("❌ L’image 'dataset_temperatures_futures.png' est introuvable dans le dossier `pictures/`.")

    st.write("---")

    st.write("La projection sur plusieurs années  semble plausible")
    ###### image ######
    img = load_image("Tendance2025_2035.png")
    if img:
            st.image(img, use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################

    st.write("Les variations des températures moyennes sur une année semblent correctes. " \
    "Ici 4 régions sont testés sur 2026 en comparaison de 2020 - qui a été une des années les plus chaudes depuis plus d'un siècle ")
    ###### image ######
    img = load_image("comparatif_2020_2026.png")
    if img:
            st.image(img, caption="test sur 4 régions", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################
    st.write("Let's go pour la modélisation 🚀 ")

#################################
# ⚙️ DATACLEANING ECO2 MIX    ⚙️#
#################################
def cleaning():

    st.markdown("###  ⚙️ Data Cleaning ✂️ Dataset Principal")

    st.markdown("""
                Le Datacleaning : a été la part la plus importante en terme de préprocessing. 
                En effet Pour avoir la timeserie la plus longue, nous avons inclu dans le dataset les données de consommation les plus récentes possible. 
                Problème : ces données (de janvier 2023 à décembre 2024) n'ayant pas été consolidées par les équipes Data de RTE contrairement à notre dataset de base. 
                nous devons procéder à un nettoyage des valeurs aberrantes après concatenation.""")
    
    ###### image ######
    img = load_image("consolide_vs_non.png")
    if img:
            st.image(img, caption="constate de l’impact de la consolidation dans la ‘propreté’ du dataset", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")

    img = load_image("outliers_eco2mix_temps_reel.png")
    if img:
            st.image(img, caption="outliers manifestement visibles", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    
    ##################
    

    st.markdown("##### Gestion des Outliers ?")
    st.markdown("""
                Concrètement à moins d'un Black out sur 100% d'un territoire, une valeur exceptionnellement basse ou nulle sur une courte durée, 
                c'est très probablement une erreur matérielle. De même pour des valeurs anormalement élevée.
                Ci-après quelques exemples visuels des valeurs anormales que l'on a du retraiter.
                Une fois les régions et périodes problématiques identifiées (avec l'aide de Plotly qui a été hyper pratique ), le remplacement des outliers s'est fait :
                - Par **Interpolation linéaire** pour les valeur nulles et les rupture manifestes de tendances
                - Par **imputation** (par la **moyenne historique**) principalement pour les valeurs anormalement basses. 
                La méthode étant la fixation d'un seuil historique minimum avec une marge +10%, puis remplacement des outlier depassant ce seuil 
                sur la plage temporelle concernée par la moyenne historique de cette même plage.

                """)
