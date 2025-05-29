
Projet de Data Analyse visant à prévoir la consommation électrique régionale en France.

Restitution via Streamlit (voir lien section "About") 

Notre dataset Principal'Eco2Mix' : Ce jeu de données, rafraîchi une fois par jour, présente les données régionales consolidées issues de l'application éCO2mix. Elles sont élaborées à partir des comptages et complétées par des forfaits. 

On y trouve :
                    - La consommation réalisée.
                    - La production selon les différentes filières composant le mix énergétique.
                    - La consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP).
                    - Le solde des échanges avec les régions limitrophes.
Source : pour en savoir plus et télécharger ce dataset produit par RTE, cliquez [ICI](http://bit.ly/4juXg9D)

J'ai exploré divers jeux de données, principalement sur l'énergie, la température et la population, pour identifier les variables clés. 
Après un nettoyage et un prétraitement approfondis des données, l'analyse s'est concentrée sur la corrélation entre la consommation et la température, ainsi que la saisonnalité et l'heure.  Plusieurs modèles de prédiction de séries temporelles (Prophet, Random Forest, XGBoost) ont été testés et optimisés, avec une comparaison des performances et une réflexion sur la granularité temporelle. 

Type de problème - Machine learning : Notre projet s’apparente à de la prédiction de valeurs continues dans une suite temporelle présentant plusieurs saisonnalités. L'objectif est d'anticiper la demande en énergie en fonction du temp et des conditions météorologiques.

#pandas 
#numpy 
#matplotlib
#seaborn
#scipy
#plotly 
#scikit-learn 
#xgboost
#prophet