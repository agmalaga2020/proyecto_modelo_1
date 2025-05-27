#!/usr/bin/env python
# coding: utf-8

# # Modelo Predictivo de Precios de Vivienda con Datos Integrados
# 
# Este notebook extiende el análisis original integrando el dataset municipal (dataset_municipio_cnae_anual_2014_2020.csv)
# con los datos de viviendas para enriquecer el modelo predictivo.

# ## 1. Importación de librerías

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Para preprocesamiento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Para modelado
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ## 2. Carga de datos

# ### 2.1 Carga de datos de viviendas

print("Cargando datos de viviendas...")
# Cargamos el archivo principal de viviendas
housing_df = pd.read_csv('spanish_houses.csv')

# Verificamos las dimensiones
print(f"Dimensiones del dataset de viviendas: {housing_df.shape}")
print(f"Primeras filas del dataset de viviendas:")
print(housing_df.head())

# ### 2.2 Carga de datos municipales

print("\nCargando datos municipales...")
# Cargamos el dataset municipal
municipal_df = pd.read_csv('dataset_municipio_cnae_anual_2014_2020.csv')

# Verificamos las dimensiones
print(f"Dimensiones del dataset municipal: {municipal_df.shape}")
print(f"Primeras filas del dataset municipal:")
print(municipal_df.head())

# ## 3. Preprocesamiento inicial

# ### 3.1 Limpieza de datos de viviendas

print("\nRealizando limpieza inicial de datos de viviendas...")

# Convertimos columnas numéricas
numeric_cols = ['price', 'room_num', 'bath_num', 'm2_real', 'm2_useful', 'air_conditioner', 
                'balcony', 'built_in_wardrobe', 'chimney', 'garden', 'lift', 'reduced_mobility', 
                'storage_room', 'swimming_pool', 'terrace']

for col in numeric_cols:
    if col in housing_df.columns:
        housing_df[col] = pd.to_numeric(housing_df[col], errors='coerce')

# Extraemos el año de obtención
housing_df['obtention_year'] = pd.to_datetime(housing_df['obtention_date'], errors='coerce').dt.year

# Extraemos la provincia de loc_zone
housing_df['provincia'] = housing_df['loc_zone'].str.split(',').str[-1].str.strip()

# Extraemos el municipio de loc_city
housing_df['municipio'] = housing_df['loc_city']

# ### 3.2 Preprocesamiento de datos municipales

print("\nRealizando preprocesamiento de datos municipales...")

# Renombramos columnas para facilitar la unión
municipal_df = municipal_df.rename(columns={
    'CPRO': 'codigo_provincia',
    'NOMBRE_MUNICIPIO': 'nombre_municipio',
    'year': 'año'
})

# Creamos un diccionario de mapeo de códigos de provincia a nombres
provincia_codes = {
    '01': 'Álava', '02': 'Albacete', '03': 'Alicante', '04': 'Almería', '05': 'Ávila',
    '06': 'Badajoz', '07': 'Balears', '08': 'Barcelona', '09': 'Burgos', '10': 'Cáceres',
    '11': 'Cádiz', '12': 'Castellón', '13': 'Ciudad Real', '14': 'Córdoba', '15': 'Coruña',
    '16': 'Cuenca', '17': 'Girona', '18': 'Granada', '19': 'Guadalajara', '20': 'Gipuzkoa',
    '21': 'Huelva', '22': 'Huesca', '23': 'Jaén', '24': 'León', '25': 'Lleida',
    '26': 'La Rioja', '27': 'Lugo', '28': 'Madrid', '29': 'Málaga', '30': 'Murcia',
    '31': 'Navarra', '32': 'Ourense', '33': 'Asturias', '34': 'Palencia', '35': 'Las Palmas',
    '36': 'Pontevedra', '37': 'Salamanca', '38': 'Santa Cruz de Tenerife', '39': 'Cantabria', '40': 'Segovia',
    '41': 'Sevilla', '42': 'Soria', '43': 'Tarragona', '44': 'Teruel', '45': 'Toledo',
    '46': 'Valencia', '47': 'Valladolid', '48': 'Bizkaia', '49': 'Zamora', '50': 'Zaragoza',
    '51': 'Ceuta', '52': 'Melilla'
}

# Añadimos el nombre de la provincia
municipal_df['provincia'] = municipal_df['codigo_provincia'].map(provincia_codes)

# Filtramos solo las filas con CNAE='Total' para evitar duplicados
municipal_df = municipal_df[municipal_df['CNAE'] == 'Total']

# ## 4. Integración de datasets

print("\nIntegrando datasets...")

# Normalizamos los nombres de municipios y provincias para facilitar la unión
def normalize_text(text):
    if pd.isna(text):
        return None
    return str(text).lower().strip()

housing_df['municipio_norm'] = housing_df['municipio'].apply(normalize_text)
housing_df['provincia_norm'] = housing_df['provincia'].apply(normalize_text)
municipal_df['municipio_norm'] = municipal_df['nombre_municipio'].apply(normalize_text)
municipal_df['provincia_norm'] = municipal_df['provincia'].apply(normalize_text)

# Aseguramos que los tipos de datos sean consistentes para la unión
housing_df['obtention_year'] = housing_df['obtention_year'].astype('Int64')
municipal_df['año'] = municipal_df['año'].astype('Int64')

# Convertimos explícitamente las columnas de unión a tipo string
housing_df['municipio_norm'] = housing_df['municipio_norm'].astype(str)
housing_df['provincia_norm'] = housing_df['provincia_norm'].astype(str)
municipal_df['municipio_norm'] = municipal_df['municipio_norm'].astype(str)
municipal_df['provincia_norm'] = municipal_df['provincia_norm'].astype(str)

# Seleccionamos las variables socioeconómicas relevantes del dataset municipal
socioeconomic_vars = [
    'año', 'municipio_norm', 'provincia_norm', 'poblacion_total', 
    'importe_total_PIE', 'idh', 'idh_salud', 'idh_educacion', 
    'idh_renta', 'proporcion_urbana', 'total_empresas'
]

municipal_subset = municipal_df[socioeconomic_vars]

# Realizamos la unión por municipio, provincia y año
# Primero, aseguramos que el año de obtención esté en el rango del dataset municipal
housing_df = housing_df[housing_df['obtention_year'].between(2014, 2020)]

# Verificamos los tipos de datos antes de la unión
print("\nTipos de datos de las columnas clave en housing_df:")
print(f"municipio_norm: {housing_df['municipio_norm'].dtype}")
print(f"provincia_norm: {housing_df['provincia_norm'].dtype}")
print(f"obtention_year: {housing_df['obtention_year'].dtype}")

print("\nTipos de datos de las columnas clave en municipal_subset:")
print(f"municipio_norm: {municipal_subset['municipio_norm'].dtype}")
print(f"provincia_norm: {municipal_subset['provincia_norm'].dtype}")
print(f"año: {municipal_subset['año'].dtype}")

# Realizamos la unión
merged_df = pd.merge(
    housing_df,
    municipal_subset,
    how='left',
    left_on=['municipio_norm', 'provincia_norm', 'obtention_year'],
    right_on=['municipio_norm', 'provincia_norm', 'año']
)

# Verificamos el resultado de la unión
print(f"\nDimensiones del dataset integrado: {merged_df.shape}")
print(f"Porcentaje de filas con datos municipales: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%")

# Para las filas sin correspondencia, intentamos unir solo por provincia
# Primero, calculamos agregados a nivel provincial
province_aggs = municipal_df.groupby(['provincia_norm', 'año']).agg({
    'poblacion_total': 'sum',
    'importe_total_PIE': 'sum',
    'idh': 'mean',
    'idh_salud': 'mean',
    'idh_educacion': 'mean',
    'idh_renta': 'mean',
    'proporcion_urbana': 'mean',
    'total_empresas': 'sum'
}).reset_index()

# Identificamos filas sin datos municipales
missing_municipal = merged_df[merged_df['poblacion_total'].isna()]

# Unimos con datos provinciales
province_merged = pd.merge(
    missing_municipal,
    province_aggs,
    how='left',
    left_on=['provincia_norm', 'obtention_year'],
    right_on=['provincia_norm', 'año']
)

# Actualizamos las columnas en el dataset original
for col in ['poblacion_total', 'importe_total_PIE', 'idh', 'idh_salud', 
            'idh_educacion', 'idh_renta', 'proporcion_urbana', 'total_empresas']:
    if col + '_y' in province_merged.columns:
        merged_df.loc[merged_df['poblacion_total'].isna(), col] = province_merged[col + '_y']

# Verificamos el resultado final
print(f"Porcentaje final de filas con datos socioeconómicos: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%")

# Guardamos el dataset integrado
merged_df.to_csv('housing_municipal_integrated.csv', index=False)
print("\nDataset integrado guardado como 'housing_municipal_integrated.csv'")

# ## 5. Análisis exploratorio del dataset integrado

print("\nRealizando análisis exploratorio del dataset integrado...")

# Verificamos las columnas disponibles
print(f"Columnas del dataset integrado:")
print(merged_df.columns.tolist())

# Estadísticas descriptivas de las variables numéricas principales
numeric_vars = ['price', 'm2_real', 'm2_useful', 'room_num', 'bath_num', 
                'poblacion_total', 'idh', 'idh_renta', 'total_empresas']
print("\nEstadísticas descriptivas de variables numéricas principales:")
print(merged_df[numeric_vars].describe())

# Matriz de correlación con el precio
corr_vars = [col for col in numeric_vars if col in merged_df.columns]
print("\nMatriz de correlación con el precio:")
correlation_matrix = merged_df[corr_vars].corr()['price'].sort_values(ascending=False)
print(correlation_matrix)

# Visualizamos la distribución del precio
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['price'].dropna(), kde=True)
plt.title('Distribución de Precios de Viviendas')
plt.xlabel('Precio (€)')
plt.ylabel('Frecuencia')
plt.savefig('price_distribution.png')

# Visualizamos la relación entre precio y metros cuadrados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='m2_real', y='price', data=merged_df.sample(min(1000, len(merged_df))))
plt.title('Relación entre Precio y Metros Cuadrados')
plt.xlabel('Metros Cuadrados')
plt.ylabel('Precio (€)')
plt.savefig('price_vs_m2.png')

# Visualizamos la relación entre precio e IDH
plt.figure(figsize=(10, 6))
sns.scatterplot(x='idh', y='price', data=merged_df.sample(min(1000, len(merged_df))))
plt.title('Relación entre Precio e Índice de Desarrollo Humano')
plt.xlabel('IDH')
plt.ylabel('Precio (€)')
plt.savefig('price_vs_idh.png')

print("\nAnálisis exploratorio completado. Visualizaciones guardadas.")

# ## 6. Preparación para modelado

print("\nPreparando datos para modelado...")

# Seleccionamos las variables para el modelo
# Variables de la vivienda
housing_features = [
    'm2_real', 'm2_useful', 'room_num', 'bath_num', 
    'air_conditioner', 'balcony', 'built_in_wardrobe', 
    'chimney', 'garden', 'lift', 'storage_room', 
    'swimming_pool', 'terrace'
]

# Variables socioeconómicas
socioeconomic_features = [
    'poblacion_total', 'idh', 'idh_salud', 'idh_educacion', 
    'idh_renta', 'proporcion_urbana', 'total_empresas'
]

# Variables categóricas
categorical_features = ['house_type', 'provincia_norm']

# Combinamos todas las variables
all_features = housing_features + socioeconomic_features + categorical_features

# Variable objetivo
target = 'price'

# Filtramos filas con valores nulos en la variable objetivo
model_df = merged_df.dropna(subset=[target])

# Filtramos outliers de precio (eliminamos el 1% superior e inferior)
lower_bound = model_df[target].quantile(0.01)
upper_bound = model_df[target].quantile(0.99)
model_df = model_df[(model_df[target] >= lower_bound) & (model_df[target] <= upper_bound)]

# Dividimos en conjuntos de entrenamiento y prueba
X = model_df[all_features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}")

# Guardamos los conjuntos para uso posterior
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nConjuntos de entrenamiento y prueba guardados.")
print("Preparación de datos completada.")

print("\nScript de integración y preparación de datos completado con éxito.")
