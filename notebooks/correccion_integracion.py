#!/usr/bin/env python
# coding: utf-8

# # Corrección de Integración de Datos Municipales y de Viviendas
# 
# Este script corrige los problemas de integración entre el dataset municipal y el de viviendas,
# enfocándose en la correcta normalización y mapeo de provincias y municipios.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz, process
import re
import unicodedata
import os

# Función para normalizar texto (eliminar acentos, convertir a minúsculas, etc.)
def normalize_text(text):
    if pd.isna(text):
        return None
    # Convertir a string si no lo es
    text = str(text)
    # Normalizar Unicode (eliminar acentos)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y espacios múltiples
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio y final
    text = text.strip()
    return text

# Cargar los datasets
print("Cargando datasets...")
# Leer el archivo CSV con encoding explícito para evitar problemas con BOM
municipal_df = pd.read_csv('dataset_municipio_cnae_anual_2014_2020.csv', encoding='utf-8-sig')
housing_df = pd.read_csv('spanish_houses.csv')

# Verificar las primeras filas y columnas
print("\nPrimeras filas del dataset municipal:")
print(municipal_df.head())
print("\nColumnas del dataset municipal:")
print(municipal_df.columns.tolist())

# Verificar los códigos de provincia únicos
unique_cpro = municipal_df['CPRO'].unique()
print(f"\nCódigos de provincia únicos en el dataset municipal: {len(unique_cpro)}")
print(f"Ejemplos de códigos de provincia: {sorted(unique_cpro)[:10]}")

# Crear un diccionario de mapeo de códigos de provincia a nombres
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

# Añadir el nombre de la provincia al dataset municipal
municipal_df['provincia'] = municipal_df['CPRO'].map(provincia_codes)

# Verificar la asignación de nombres de provincia
print("\nVerificando la asignación de nombres de provincia:")
print(municipal_df[['CPRO', 'provincia']].drop_duplicates().sort_values('CPRO').head(10))

# Contar provincias únicas después del mapeo
unique_provinces = municipal_df['provincia'].unique()
print(f"\nProvincias únicas en el dataset municipal después del mapeo: {len(unique_provinces)}")
print(f"Ejemplos de provincias: {sorted(unique_provinces)[:10]}")

# Extraer provincias y municipios de housing_df
print("\nExtrayendo provincias y municipios del dataset de viviendas...")
housing_df['provincia'] = housing_df['loc_zone'].str.split(',').str[-1].str.strip()
housing_df['municipio'] = housing_df['loc_city']

# Verificar provincias únicas en housing_df
unique_housing_provinces = housing_df['provincia'].unique()
print(f"\nProvincias únicas en el dataset de viviendas: {len(unique_housing_provinces)}")
print(f"Ejemplos de provincias: {sorted(unique_housing_provinces)[:10]}")

# Normalizar nombres de municipios y provincias
print("\nNormalizando nombres...")
housing_df['municipio_norm'] = housing_df['municipio'].apply(normalize_text)
housing_df['provincia_norm'] = housing_df['provincia'].apply(normalize_text)
municipal_df['municipio_norm'] = municipal_df['NOMBRE_MUNICIPIO'].apply(normalize_text)
municipal_df['provincia_norm'] = municipal_df['provincia'].apply(normalize_text)

# Verificar provincias normalizadas
unique_housing_provinces_norm = housing_df['provincia_norm'].unique()
unique_municipal_provinces_norm = municipal_df['provincia_norm'].unique()

print(f"\nProvincias normalizadas únicas en housing_df: {len(unique_housing_provinces_norm)}")
print(f"Provincias normalizadas únicas en municipal_df: {len(unique_municipal_provinces_norm)}")

# Crear mapeo manual para provincias con nombres diferentes
print("\nCreando mapeo manual para provincias...")
province_mapping = {
    'alava': 'alava',
    'álava': 'alava',
    'albacete': 'albacete',
    'alicante': 'alicante',
    'almeria': 'almeria',
    'asturias': 'asturias',
    'avila': 'avila',
    'badajoz': 'badajoz',
    'balears': 'balears',
    'balears illes': 'balears',
    'barcelona': 'barcelona',
    'burgos': 'burgos',
    'caceres': 'caceres',
    'cadiz': 'cadiz',
    'cantabria': 'cantabria',
    'castellon': 'castellon',
    'ciudad real': 'ciudad real',
    'cordoba': 'cordoba',
    'a coruna': 'coruna',
    'coruña': 'coruna',
    'coruna': 'coruna',
    'cuenca': 'cuenca',
    'girona': 'girona',
    'granada': 'granada',
    'guadalajara': 'guadalajara',
    'gipuzkoa': 'gipuzkoa',
    'guipuzcoa': 'gipuzkoa',
    'huelva': 'huelva',
    'huesca': 'huesca',
    'jaen': 'jaen',
    'la rioja': 'la rioja',
    'las palmas': 'las palmas',
    'leon': 'leon',
    'lleida': 'lleida',
    'lugo': 'lugo',
    'madrid': 'madrid',
    'malaga': 'malaga',
    'murcia': 'murcia',
    'navarra': 'navarra',
    'ourense': 'ourense',
    'palencia': 'palencia',
    'pontevedra': 'pontevedra',
    'salamanca': 'salamanca',
    'santa cruz de tenerife': 'santa cruz de tenerife',
    'segovia': 'segovia',
    'sevilla': 'sevilla',
    'soria': 'soria',
    'tarragona': 'tarragona',
    'teruel': 'teruel',
    'toledo': 'toledo',
    'valencia': 'valencia',
    'valladolid': 'valladolid',
    'bizkaia': 'bizkaia',
    'vizcaya': 'bizkaia',
    'zamora': 'zamora',
    'zaragoza': 'zaragoza',
    'ceuta': 'ceuta',
    'melilla': 'melilla'
}

# Aplicar mapeo manual a ambos datasets
housing_df['provincia_mapped'] = housing_df['provincia_norm'].map(lambda x: province_mapping.get(x, x))
municipal_df['provincia_mapped'] = municipal_df['provincia_norm'].map(lambda x: province_mapping.get(x, x))

# Verificar provincias después del mapeo manual
unique_housing_provinces_mapped = housing_df['provincia_mapped'].unique()
unique_municipal_provinces_mapped = municipal_df['provincia_mapped'].unique()

print(f"\nProvincias mapeadas únicas en housing_df: {len(unique_housing_provinces_mapped)}")
print(f"Provincias mapeadas únicas en municipal_df: {len(unique_municipal_provinces_mapped)}")

# Encontrar provincias comunes después del mapeo
common_provinces = set(unique_housing_provinces_mapped) & set(unique_municipal_provinces_mapped)
print(f"\nProvincias comunes después del mapeo: {len(common_provinces)}")
print(f"Provincias comunes: {sorted(common_provinces)}")

# Filtrar solo las filas con CNAE='Total' para evitar duplicados
municipal_df = municipal_df[municipal_df['CNAE'] == 'Total']

# Extraer el año de obtención
housing_df['obtention_year'] = pd.to_datetime(housing_df['obtention_date'], errors='coerce').dt.year

# Asegurar que el año de obtención esté en el rango del dataset municipal
housing_df = housing_df[housing_df['obtention_year'].between(2014, 2020)]

# Realizar la unión con el dataset municipal
print("\nRealizando unión con el dataset municipal...")
# Seleccionar las variables socioeconómicas relevantes
socioeconomic_vars = [
    'year', 'municipio_norm', 'provincia_mapped', 'poblacion_total', 
    'importe_total_PIE', 'idh', 'idh_salud', 'idh_educacion', 
    'idh_renta', 'proporcion_urbana', 'total_empresas'
]

municipal_subset = municipal_df[socioeconomic_vars]

# Realizar la unión
merged_df = pd.merge(
    housing_df,
    municipal_subset,
    how='left',
    left_on=['municipio_norm', 'provincia_mapped', 'obtention_year'],
    right_on=['municipio_norm', 'provincia_mapped', 'year']
)

# Verificar el resultado de la unión
print(f"\nDimensiones del dataset integrado: {merged_df.shape}")
print(f"Porcentaje de filas con datos municipales: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%")

# Para las filas sin correspondencia, intentar unir solo por provincia
if merged_df['poblacion_total'].notna().sum() / len(merged_df) < 0.5:
    print("\nIntentando unión solo por provincia...")
    # Calcular agregados a nivel provincial
    province_aggs = municipal_df.groupby(['provincia_mapped', 'year']).agg({
        'poblacion_total': 'sum',
        'importe_total_PIE': 'sum',
        'idh': 'mean',
        'idh_salud': 'mean',
        'idh_educacion': 'mean',
        'idh_renta': 'mean',
        'proporcion_urbana': 'mean',
        'total_empresas': 'sum'
    }).reset_index()
    
    # Identificar filas sin datos municipales
    missing_municipal = merged_df[merged_df['poblacion_total'].isna()]
    
    # Unir con datos provinciales
    province_merged = pd.merge(
        missing_municipal,
        province_aggs,
        how='left',
        left_on=['provincia_mapped', 'obtention_year'],
        right_on=['provincia_mapped', 'year']
    )
    
    # Actualizar las columnas en el dataset original
    for col in ['poblacion_total', 'importe_total_PIE', 'idh', 'idh_salud', 
                'idh_educacion', 'idh_renta', 'proporcion_urbana', 'total_empresas']:
        if col + '_y' in province_merged.columns:
            merged_df.loc[merged_df['poblacion_total'].isna(), col] = province_merged[col + '_y']
    
    # Verificar el resultado final
    print(f"Porcentaje final de filas con datos socioeconómicos: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%")

# Guardar el dataset integrado
merged_df.to_csv('housing_municipal_integrated_fixed.csv', index=False)
print("\nDataset integrado guardado como 'housing_municipal_integrated_fixed.csv'")

# Crear un informe de diagnóstico
print("\nCreando informe de diagnóstico...")
with open('diagnostico_integracion_corregido.txt', 'w') as f:
    f.write("# Informe de Diagnóstico de Integración de Datos (Corregido)\n\n")
    
    f.write("## Estadísticas de Provincias\n")
    f.write(f"- Provincias únicas en housing_df (original): {len(unique_housing_provinces)}\n")
    f.write(f"- Provincias únicas en municipal_df (original): {len(unique_provinces)}\n")
    f.write(f"- Provincias únicas en housing_df (normalizadas): {len(unique_housing_provinces_norm)}\n")
    f.write(f"- Provincias únicas en municipal_df (normalizadas): {len(unique_municipal_provinces_norm)}\n")
    f.write(f"- Provincias únicas en housing_df (mapeadas): {len(unique_housing_provinces_mapped)}\n")
    f.write(f"- Provincias únicas en municipal_df (mapeadas): {len(unique_municipal_provinces_mapped)}\n")
    f.write(f"- Provincias comunes después del mapeo: {len(common_provinces)}\n\n")
    
    f.write("## Resultados de la Integración\n")
    f.write(f"- Dimensiones del dataset integrado: {merged_df.shape}\n")
    f.write(f"- Filas con datos municipales: {merged_df['poblacion_total'].notna().sum()}\n")
    f.write(f"- Porcentaje de filas con datos municipales: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%\n")

print("Diagnóstico corregido completado. Informe guardado en 'diagnostico_integracion_corregido.txt'")

# Análisis exploratorio del dataset integrado
print("\nRealizando análisis exploratorio del dataset integrado...")

# Verificar las columnas disponibles
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

# Preparación para modelado
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
categorical_features = ['house_type', 'provincia_mapped']

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}")

# Guardamos los conjuntos para uso posterior
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_data_fixed.csv', index=False)
test_data.to_csv('test_data_fixed.csv', index=False)

print("\nConjuntos de entrenamiento y prueba guardados.")
print("Preparación de datos completada.")

print("\nScript de corrección e integración de datos completado con éxito.")
