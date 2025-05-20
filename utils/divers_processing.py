#################################
# ⚙️ DATASET TEMPERATURES FUTURES    ⚙️#
#################################

from PIL import Image
import streamlit as st
from utils.assets_loader import load_image

def drias():
    st.subheader("🌡️ Données Climatiques Futures pour la Simulation")

    st.markdown("""
    Pour tester notre modèle avec des **données futures réalistes**, nous devons récupérer une simulation des **températures régionales**, 
    à la **maille fine** et sur **plusieurs années**. Après exploration, nous avons identifié sur le portail de la [**DRIAS**](https://www.drias-climat.fr) un catalogue de données de **simulations climatiques** 
    au format [**NetCDF**](https://fr.wikipedia.org/wiki/NetCDF) pouvant répondre à ce besoin.

    Choix d’un Scénario d’émission de Co2 : 

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

    st.write("Nous passons d’un fichier : netCDF > df > .csv  avec plus de 6 Go et 84 M de lignes (!!)")

    code = '''
    import xarray as xr
    import pandas as pd

    # Charger le fichier netCDF
    nc_file_2 = "Explo/tasAdjust_France_CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63_rcp4.5_METEO-FRANCE_ADAMONT-France_SAFRAN_day_20240101-20351231.nc"
    ds2 = xr.open_dataset(nc_file_2)

    # Convertir en DataFrame (en fonction des variables présentes dans le fichier)
    df2 = ds2.to_dataframe()

    # Sauvegarder en CSV
    csv_file = "output2.csv"
    df2.to_csv(csv_file)

    print(f"Fichier CSV enregistré sous {csv_file}")
    '''
    st.code(code, language='python')

    ###### image ######
    img = load_image("Netcdf_to_df.png")
    if img:
            st.image(img, caption="Nous passons d’un fichier : netCDF > df > .csv  avec plus de 6 Go et 84 M de lignes (!!)", use_container_width=True)
    else:
            st.warning("❌ L’image est introuvable dans le dossier `pictures/`.")
    ##################
    st.write("---")

    st.write("Croisement de l’ensemble des données avec définition des zones géographiques (coordonnées lat/long) pour chaque régions :")
    
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

    st.markdown("### ⚠️ Avertissement")
    st.info("Cet exercice de simulation des températures futures vise uniquement à illustrer des méthodes de traitement de données " \
    "et nous permettre de mobiliser notre modèle de prédiction et simuler de manière triviale la future consommation électrique.")    
    st.markdown("""
    - ⚠️ Les **prévisions climatiques** sont issues de **modèles complexes** qui comportent des **incertitudes** importantes.  
    - 📅 Les **données consolidées datent de 2020**, ce qui peut introduire un **biais temporel**.  
    - 🗺️ Pour associer les données climatiques à nos **régions administratives**, nous avons utilisé une méthode simple de **regroupement par encadrement** 
      des **coordonnées géographiques**.

    ---
    """)
