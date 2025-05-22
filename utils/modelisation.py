import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  
import datetime

#Import des bibliothèques ML
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV


def intro():
    st.write('#### Classification du problème 📂')

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
    st.write('#### Choix des modèles ML 🤖 et besoins relatifs aux Séries temporelles ⏲️')
    st.markdown("""
                De façon plus limitée que le rapport d'étude, nous ne présenterons ici que :

                - Prophet : pour challenger notamment la détection des saisonnalités et la robustesse à long terme.
                - RandomForest, XG Boost : 2 autres modèles, plus généralistes et simples à entraîner

                Ces modèles sont connus pour bien gérer les séries temporelles. 
                """)
    st.markdown("""


                """)