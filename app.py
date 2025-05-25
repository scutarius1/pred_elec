import streamlit as st
import streamlit.components.v1 as components
import io
import os
#ajout OS + if not os.path.exists
import pandas as pd
import matplotlib.pyplot as plt
import gdown

#import Explo_Viz , considération data cleaning & seeking

from utils import Explo_Viz
from utils import divers_processing
from utils import modelisation

# #########################
# ⚙️ LOAD & PREPROCESS ⚙️ #
##########################

@st.cache_data
def load_and_preprocess_data():
    """Télécharge et prétraite les données depuis Google Drive."""
    file_id = "1aqr3QQCoeQcNp8vrnvgaDHKXzlTYbFGC"  # Ton ID de fichier extrait
    
    url = f"https://drive.google.com/uc?id={file_id}"  # Lien de téléchargement direct
    output = "eco2mix-regional_reduced.csv"
    gdown.download(url, output, quiet=False)

    if not os.path.exists(output):  # ⚠️ Evite redownload
        gdown.download(url, output, quiet=False)

    df_cons = pd.read_csv(output, sep=',', on_bad_lines="skip", encoding="utf-8",low_memory=False)  
    
    # Appliquer le prétraitement
    df_cons_preprocessed = Explo_Viz.preprocess_data(df_cons)
    df_energie = Explo_Viz.preprocess_data2(df_cons_preprocessed)
    df_temp = Explo_Viz.load_temp()  # Charger les données de température
    return df_cons_preprocessed, df_energie, df_temp #ajout de df_energie

def main():
    #st.title("Prédiction de Consommation Electrique en France")
    st.sidebar.title("⚡⚡ Prédiction Conso Electrique en France ⚡⚡")
    pages = ["📖 Contexte et Datasets", "📊 Production VS Consommation", "📉 Variabilité de la consommation", "✂️ Prétraitements des données"," 🤖 Modélisation"]
    page = st.sidebar.radio("Aller vers", pages)
    #st.sidebar.title("Modélisation")
    #st.sidebar.page_link("pages/modelisation.py", label="Processing et Modélisation")
    st.sidebar.title("Simulateur")
    st.sidebar.page_link("pages/simulateur.py", label="📈 Prédictions Régionales Futures")

    df_cons_preprocessed, df_energie, df_temp = load_and_preprocess_data() # AJOUTE



#################################
# ⚙️ CONTEXTE ET DATASETS     ⚙️#
#################################
    if page == pages[0]: 
        st.title("Prédiction de Consommation Electrique en France")
        st.write("")
        st.header("Contexte")
        st.markdown(""" L’adéquation entre la production et la consommation d’électricité est au cœur des préoccupations d’un acteur de l’énergie comme EDF. 
                 EDF, en tant que producteur et commercialisateur d’électricité est en effet un responsable d’équilibre vis-à-vis de RTE. 
                 Cela signifie qu’il se doit d’assurer à tout instant un équilibre entre sa production et la consommation de ses clients, sous peine de pénalités. 
                 Pour se faire, construire un modèle de prévision de la consommation de ses clients est une activité essentielle au bon fonctionnement de EDF.""") 
        
        st.write('**Objectif** : Constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional. ' \
            'Analyse pour en déduire une prévision de consommation (risque de black out notamment)')

        st.write("## Les jeux de données mis en oeuvre")
        data = [
        {"Objet": "Energie (Consolidé)", "Description": "Production et consommation d’énergie par type de moyen de production et régions ( 30 min)", "Période couverte": "2013-2022", "Volumétrie (lignes x colonnes)": "2.121.408 x 32", "Source": "ODRE, Open Data EDF"},
        {"Objet": "Energie (Temps Réel)", "Description": "Production et consommation d’énergie par type de moyen de production et région (15 min, non consolidé)", "Période couverte": "2023-2024", "Volumétrie (lignes x colonnes)": "796.000 x 32", "Source": "ODRE"},
        {"Objet": "Population", "Description": "Évolutions et prévisions de la population française par région", "Période couverte": "1990-2070", "Volumétrie (lignes x colonnes)": "264.951 x 7", "Source": "INSEE"},
        {"Objet": "Température", "Description": "Évolution des températures quotidiennes par région", "Période couverte": "2016-2024", "Volumétrie (lignes x colonnes)": "41.756 x 7", "Source": "Météo France"},
        {"Objet": "Température", "Description": "Simulations 'DRIAS-2020' : données corrigées quotidiennes. Grille Safran", "Période couverte": "2006-2100", "Volumétrie (lignes x colonnes)": "83.987.046 x 8", "Source": "DRIAS"},
        ]

        st.markdown("""
                    <style>
                    .stTable td:nth-child(2),.stTable td:nth-child(4) {
                    white-space: nowrap;
                    }
                    </style>
                    """, unsafe_allow_html=True)
        st.table(data)

        st.markdown(""" Les échanges avec le data scientist EDF ont confirmé notre intuition d’expliquer la variable cible **Consommation** 
                    par les variables explicatives **Température**, **Dates** et **Population**.
                    Nous pourrons en effet à travers la variable 'Date' étudier l’impact des saisons, des périodes de vacances scolaires et des week-ends notamment.
        """)
    
        st.write("### 🔎 Exploration 'Eco2Mix' - Notre dataset Principal")
        st.markdown("""
                    Ce jeu de données, rafraîchi une fois par jour, présente les données régionales consolidées depuis janvier 2021 et définitives (de janvier 2013 à décembre 2020) issues de l'application éCO2mix. 
                    Elles sont élaborées à partir des comptages et complétées par des forfaits. Les données sont dites consolidées lorsqu'elles ont été vérifiées et complétées (livraison en milieu de M+1). 
                    Vous y trouverez au pas demi-heure:
                    - La consommation réalisée.
                    - La production selon les différentes filières composant le mix énergétique.
                    - La consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP).
                    - Le solde des échanges avec les régions limitrophes.
                    """)
        st.markdown("Source : pour en savoir plus et télécharger ce dataset produit par RTE, cliquez [ICI](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJsaW5lIiwiZnVuYyI6IlNVTSIsInlBeGlzIjoiY29uc29tbWF0aW9uIiwiY29sb3IiOiJyYW5nZS1jdXN0b20iLCJzY2llbnRpZmljRGlzcGxheSI6dHJ1ZX1dLCJ4QXhpcyI6ImRhdGVfaGV1cmUiLCJtYXhwb2ludHMiOjIwMCwidGltZXNjYWxlIjoibWludXRlIiwic29ydCI6IiIsImNvbmZpZyI6eyJkYXRhc2V0IjoiZWNvMm1peC1yZWdpb25hbC1jb25zLWRlZiIsIm9wdGlvbnMiOnsiZGlzanVuY3RpdmUubGliZWxsZV9yZWdpb24iOnRydWUsImRpc2p1bmN0aXZlLm5hdHVyZSI6dHJ1ZSwic29ydCI6Ii1kYXRlX2hldXJlIn19LCJzZXJpZXNCcmVha2Rvd24iOiJsaWJlbGxlX3JlZ2lvbiJ9XSwidGltZXNjYWxlIjoiIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D)")
        st.write("---")
        st.write("Echantillon .sample(10) : ")
        st.dataframe(df_cons_preprocessed.sample(10))  # Utiliser le dataframe prétraité
        st.write("---")
        st.write("résumé statistique  .describe() : ")
        st.dataframe(df_cons_preprocessed.describe())
        st.write("---")
        st.write("Infos dataframe  .info() : ")
        # Capturer et afficher df_cons_preprocessed.info() directement avec st.text
        buffer = io.StringIO()
        df_cons_preprocessed.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

       
####################################
# ⚙️ DATAVIZ ADEQUATION PROD/CONSO    ⚙️#
####################################

    elif page == pages[1]:
        st.header("Inégalités Régionales : Mix Energétique et Capacités de Production"
        )

        st.write ("""En plus de ne pas avoir le même mix energétique, les régions sont dans une situation de disparité de leurs capacités de production pour couvrir leurs besoins : """)

#Affichage des taux de couverture/régions

        fig2 = Explo_Viz.create_barplot(df_cons_preprocessed)
        fig2.text(0.5, -0.15, "Certaines régions sont largement déficitaires en terme de phasage entre leur production et leurs besoin. Cf. Couverture 100%", ha='center', va='top', fontsize=12)
        st.pyplot(fig2)
        plt.close(fig2)
        st.write("");st.write("") 
        st.write("---")
#Affichage des besoins /régions dans le temps    

        st.header("Phasages et Echanges Inter-régionaux : Visualisation interactive 🤓 "
        )
        st.write("");st.write("") 

        st.write(""" Avec l'aide des opérateurs d'énergie, les régions procèdent toute l'année à des *échanges*.
                Le graphique interactif ci-après permet de constater quelque soit la période et la maille temporelle choisie :
                la **variabilité des besoins** des Régions au fil du temps d'une part. Le phasage entre Consommation (Ligne en pointillé noir) 
                 et Production au moyen des **échanges inter-régionaux** d'autre part.
                    """)
        st.write("") 

## ⚙️ OUTIL DE FILTRAGE ####
        st.markdown('<div class="filtre-vert">', unsafe_allow_html=True)
        st.markdown("<hr style='border: 4px solid #4CAF50;'>", unsafe_allow_html=True)
        st.markdown('<h6 style="text-align: center; color: #4CAF50;">🔎 Filtres d\'Analyse</h6>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            annee = st.selectbox("**Année** ('None' pour tout le dataset')", options=sorted(df_cons_preprocessed['DH'].dt.year.unique(), 
            reverse=True) + [None], index=sorted(df_cons_preprocessed['DH'].dt.year.unique(), reverse=True).index(2022)  # 2022 par défaut
            )
        with col2:
            if annee:
                mois = st.selectbox("**Mois** ('None' pour toute l'année)",
                options=sorted(df_cons_preprocessed[df_cons_preprocessed['DH'].dt.year == annee]['DH'].dt.month.unique()) + [None],index=0
                )
            else:
                mois = None
        with col3:
            frequence_resample = st.radio("**Fréquence** (échantillonnage)", options=['Heure', 'Jour', 'Semaine', 'Mois'],index=1  # 'Jour' par défaut
            )
#'Bretagne', 'Centre-Val de Loire', 
        regions_preselectionnees = ['Auvergne-Rhône-Alpes', "Provence-Alpes-Côte d'Azur"]
        regions = sorted(df_cons_preprocessed['Région'].unique())

        regions_selected = st.multiselect("Régions à comparer (2 maximum)", options=regions,default=regions_preselectionnees
        )
        st.markdown("<hr style='border: 4px solid #4CAF50;'>", unsafe_allow_html=True)

## ⚙️ GRAPHIQUE INTERACTIF  ####       
        fig = Explo_Viz.create_regional_plots(df_cons_preprocessed, annee, mois, None, frequence_resample, regions_selected)
        st.pyplot(fig)
        plt.close(fig)

#################################
# ⚙️ DATAVIZ CORRELATIONS  ⚙️#
#################################
    elif page == pages[2]:
        st.header("Saisonnalité et Consommation")
        
        st.write("""Ce graphique suivant montre l’évolution mensuelle de la consommation d’énergie entre 2013 et 2023. Au-delà du lien entre le mois de l'année et le niveeau de consommation, 
                 on observe une chute marquée en 2020 (ligne grise), liée à la crise du Covid-19 et ses confinements. 
        En 2022 (ligne cyan), la consommation reste globalement plus basse, traduisant l’effet des tensions énergétiques causées par la guerre en Ukraine et les efforts de sobriété""")
        
        df_st2 = Explo_Viz.compute_df_st2(df_energie)
        fig_boxplot = Explo_Viz.create_annual_plot(df_st2)
        st.pyplot(fig_boxplot)
        plt.close(fig_boxplot)
        st.write("")

        st.write("### Température et Consommation")
        st.write("""Le graphique ci-après combine des 'boxplots' de **consommation électrique (MW)** et un 'scatter plot' de **température moyenne (°C)**,
                 le tout groupé par mois sur l’entièreté de la période étudiée. Il permet d'émettre l'hypothèse d'une influence significative de la température sur la consommation électrique au niveau mensuel, 
                 tout en visualisant la distribution et la variabilité de ces deux variables clés au fil de l'année :"""
        )

        fig_boxplot,df_corr01 = Explo_Viz.create_boxplot(df_energie, df_temp)  # Appel de la fonction
        st.pyplot(fig_boxplot)  # Affichage du graphique dans Streamlit
        plt.close(fig_boxplot)  # Fermeture pour éviter les conflits de rendu

        st.write(" - **Saisonnalité de la Consommation** : Les boxplots de consommation révèlent une forte saisonnalité. " \
        "La consommation est généralement plus élevée en hiver, avec des médianes et des étendues interquartiles significativement plus hautes.")
        st.write("")
        st.write(" - **Corrélation Inverse Apparente** : En juxtaposant les deux types de données, on peut observer une corrélation inverse suggestive " \
        "entre la température moyenne et la consommation électrique.")

        #CORRELATION TEMPERATURE ET CONSO
        st.write("""Pour vérifier cette hypothèse de correlation, ci-après le résultat d'un test statistique  """
        )
        st.write(" Les hypothèses :")
        st.write(" H0 : Il n'y a pas d'influence de la température sur la consommation")
        st.write("H1 : Il y a une influence significative de la température sur la consommation")
        #####UPDATE####
        corr_results_temp, df_corr01 = Explo_Viz.Test_corr_temp(df_corr01)

        st.write("#### Résultats des tests de corrélation entre Température moyenne et Consommation")
        st.write("**Les hypothèses :**")
        st.write("- H0 : Il n'y a pas de lien entre la température moyenne et la consommation")
        st.write("- H1 : Il existe une relation significative entre température et consommation")

        st.write(f"- Corrélation de Spearman : {corr_results_temp['spearman_corr']:.3f} (p-value = {corr_results_temp['spearman_p']:.3e})")
        st.write(f"- Corrélation de Pearson : {corr_results_temp['pearson_corr']:.3f} (p-value = {corr_results_temp['pearson_p']:.3e})")

        if corr_results_temp['spearman_p'] < 0.05 or corr_results_temp['pearson_p'] < 0.05:
            st.write("➡️ Le lien entre **température** et **consommation** est **significatif**, car la p-valeur est inférieure à 0.05.")
        else:
            st.write("❗ Aucune corrélation significative détectée entre température et consommation (p-valeur > 0.05).")

        st.write("💡 Note : La température peut influencer la consommation énergétique (chauffage ou climatisation), mais cette relation peut varier selon les régions, saisons, ou plages horaires.")
        #####UPDATE####

        #CORRELATION PLAGE HORAIRE ET CONSO
        st.write("### Plage Horaire et Consommation")
        st.write("")
        st.write(""" La variabilité horaire est particulièrement marquée en hiver, tandis qu’elle reste plus stable en été, comme l’indiquent les amplitudes des boxplots.
                 La forte structuration des courbes selon l’heure suggère une corrélation claire entre consommation électrique et rythme quotidien d’activité."""
                 )

        fig_boxplot, df_st3 = Explo_Viz.create_boxplot_season(df_energie)  # Appel de la fonction
        st.pyplot(fig_boxplot)  # Affichage du graphique dans Streamlit
        plt.close(fig_boxplot)  # Fermeture pour éviter les conflits de rendu

        corr_results, df_st3 = Explo_Viz.Test_corr(df_st3)

        st.write("#### Résultats des tests de corrélation entre Plage Horaire et Consommation")
        st.write(" Les hypothèses :")
        st.write(" H0 : Il n'y a pas d'influence de la plage horaire sur la consommation")
        st.write("H1 : Il y a une influence significative de la plage horaire sur la consommation")
        
        st.write(f"- Corrélation de Spearman : {corr_results['spearman_corr']:.3f} (p-value = {corr_results['spearman_p']:.3e})")
        st.write(f"- Corrélation de Pearson : {corr_results['pearson_corr']:.3f} (p-value = {corr_results['pearson_p']:.3e})")
        st.write("le facteur “Plage_Horaire” a un effet significatif sur la consommation,\n car la p-valeur est inférieure à 0.05")
        st.write(""" Mais ces tests ne captent pas nécessairement toute la structure réelle du phénomène (comme la nature cyclique des heures)."""
                 )

#################################
# ⚙️     MODELISATIONS        ⚙️#
#################################

    elif page == pages[4]:
        st.title("Modélisation")
        modelisation.intro()
        modelisation.lancement()

        
#################################
# ⚙️     DIVERS PROCESSING        ⚙️#
#################################

    elif page == pages[3]:
        st.title("Challenges Preprocessing ")
        divers_processing.cleaning()
        divers_processing.drias()
        

if __name__ == "__main__":
    main()