import geopandas as gpd

# Charger le shapefile
shapefile = gpd.read_file('C:/Users/gaspa/Desktop/stage_CRPG/qgis/calculs_aires/a7.shp')

# Calculer l'aire de chaque polygone
shapefile['area'] = shapefile['geometry'].area

# Afficher les valeurs d'aire
print(shapefile[['area']])