#!/usr/bin/env python
# coding: utf-8

# # Diagnóstico de correspondencia entre municipios y provincias
# 
# Este script analiza las diferencias entre los nombres de municipios y provincias
# en los datasets de viviendas y municipal para identificar y corregir problemas de correspondencia.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz, process
import re
import unicodedata

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
housing_df = pd.read_csv('spanish_houses.csv')
municipal_df = pd.read_csv('dataset_municipio_cnae_anual_2014_2020.csv')

# Extraer provincias y municipios de housing_df
housing_df['provincia'] = housing_df['loc_zone'].str.split(',').str[-1].str.strip()
housing_df['municipio'] = housing_df['loc_city']

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

# Normalizar nombres de municipios y provincias
print("Normalizando nombres...")
housing_df['municipio_norm'] = housing_df['municipio'].apply(normalize_text)
housing_df['provincia_norm'] = housing_df['provincia'].apply(normalize_text)
municipal_df['municipio_norm'] = municipal_df['NOMBRE_MUNICIPIO'].apply(normalize_text)
municipal_df['provincia_norm'] = municipal_df['provincia'].apply(normalize_text)

# Filtrar solo las filas con CNAE='Total' para evitar duplicados
municipal_df = municipal_df[municipal_df['CNAE'] == 'Total']

# Extraer el año de obtención
housing_df['obtention_year'] = pd.to_datetime(housing_df['obtention_date'], errors='coerce').dt.year

# Asegurar que el año de obtención esté en el rango del dataset municipal
housing_df = housing_df[housing_df['obtention_year'].between(2014, 2020)]

# Analizar los valores únicos de provincias en ambos datasets
housing_provinces = housing_df['provincia_norm'].unique()
municipal_provinces = municipal_df['provincia_norm'].unique()

print(f"\nNúmero de provincias únicas en housing_df: {len(housing_provinces)}")
print(f"Número de provincias únicas en municipal_df: {len(municipal_provinces)}")

# Encontrar provincias que están en housing_df pero no en municipal_df
missing_provinces = set(housing_provinces) - set(municipal_provinces)
print(f"\nProvincias en housing_df que no están en municipal_df ({len(missing_provinces)}):")
print(sorted(missing_provinces)[:20])  # Mostrar solo las primeras 20 para no saturar la salida

# Encontrar las mejores coincidencias para las provincias faltantes
print("\nMejores coincidencias para provincias faltantes:")
for province in sorted(missing_provinces)[:10]:  # Analizar solo las primeras 10
    matches = process.extract(province, municipal_provinces, limit=3, scorer=fuzz.token_sort_ratio)
    print(f"{province}: {matches}")

# Crear un diccionario de mapeo para provincias
province_mapping = {}
for province in housing_provinces:
    if province in municipal_provinces:
        province_mapping[province] = province
    else:
        # Encontrar la mejor coincidencia
        match = process.extractOne(province, municipal_provinces, scorer=fuzz.token_sort_ratio)
        if match and match[1] > 70:  # Solo mapear si la coincidencia es buena (>70%)
            province_mapping[province] = match[0]

# Guardar el mapeo de provincias
province_mapping_df = pd.DataFrame({
    'housing_province': list(province_mapping.keys()),
    'municipal_province': list(province_mapping.values())
})
province_mapping_df.to_csv('province_mapping.csv', index=False)
print(f"\nMapeo de provincias guardado en 'province_mapping.csv'")

# Analizar los valores únicos de municipios por provincia
print("\nAnalizando municipios por provincia...")
province_municipality_counts = {}

# Seleccionar algunas provincias para análisis detallado
sample_provinces = list(set(municipal_provinces) & set(housing_provinces))[:5]

for province in sample_provinces:
    housing_municipalities = housing_df[housing_df['provincia_norm'] == province]['municipio_norm'].unique()
    municipal_municipalities = municipal_df[municipal_df['provincia_norm'] == province]['municipio_norm'].unique()
    
    province_municipality_counts[province] = {
        'housing': len(housing_municipalities),
        'municipal': len(municipal_municipalities),
        'common': len(set(housing_municipalities) & set(municipal_municipalities))
    }
    
    print(f"\nProvincia: {province}")
    print(f"  Municipios en housing_df: {len(housing_municipalities)}")
    print(f"  Municipios en municipal_df: {len(municipal_municipalities)}")
    print(f"  Municipios comunes: {province_municipality_counts[province]['common']}")
    
    # Mostrar algunos ejemplos de municipios que no coinciden
    missing_municipalities = set(housing_municipalities) - set(municipal_municipalities)
    if missing_municipalities:
        print(f"  Ejemplos de municipios en housing_df que no están en municipal_df:")
        for muni in list(missing_municipalities)[:5]:
            matches = process.extract(muni, municipal_municipalities, limit=3, scorer=fuzz.token_sort_ratio)
            print(f"    {muni}: {matches}")

# Crear un diccionario de mapeo para municipios por provincia
print("\nCreando mapeo de municipios...")
municipality_mapping = {}

for province in province_mapping.values():
    housing_municipalities = housing_df[housing_df['provincia_norm'].map(lambda x: x in province_mapping and province_mapping[x] == province)]['municipio_norm'].unique()
    municipal_municipalities = municipal_df[municipal_df['provincia_norm'] == province]['municipio_norm'].unique()
    
    for municipality in housing_municipalities:
        if pd.isna(municipality) or municipality is None:
            continue
            
        key = (province, municipality)
        
        if municipality in municipal_municipalities:
            municipality_mapping[key] = municipality
        else:
            # Encontrar la mejor coincidencia
            match = process.extractOne(municipality, municipal_municipalities, scorer=fuzz.token_sort_ratio)
            if match and match[1] > 80:  # Solo mapear si la coincidencia es buena (>80%)
                municipality_mapping[key] = match[0]

# Convertir el mapeo de municipios a DataFrame
municipality_mapping_list = []
for (province, municipality), mapped_municipality in municipality_mapping.items():
    municipality_mapping_list.append({
        'province': province,
        'housing_municipality': municipality,
        'municipal_municipality': mapped_municipality
    })

municipality_mapping_df = pd.DataFrame(municipality_mapping_list)
municipality_mapping_df.to_csv('municipality_mapping.csv', index=False)
print(f"Mapeo de municipios guardado en 'municipality_mapping.csv'")

# Aplicar el mapeo a housing_df
print("\nAplicando mapeo a housing_df...")
# Crear una función para aplicar el mapeo de provincias
def map_province(province):
    if pd.isna(province) or province is None:
        return None
    return province_mapping.get(province, province)

# Crear una función para aplicar el mapeo de municipios
def map_municipality(row):
    province = row['provincia_norm_mapped']
    municipality = row['municipio_norm']
    
    if pd.isna(province) or pd.isna(municipality) or province is None or municipality is None:
        return municipality
    
    key = (province, municipality)
    return municipality_mapping.get(key, municipality)

# Aplicar los mapeos
housing_df['provincia_norm_mapped'] = housing_df['provincia_norm'].apply(map_province)
housing_df['municipio_norm_mapped'] = housing_df.apply(map_municipality, axis=1)

# Realizar la unión con el dataset municipal
print("\nRealizando unión con el dataset municipal...")
# Seleccionar las variables socioeconómicas relevantes
socioeconomic_vars = [
    'year', 'municipio_norm', 'provincia_norm', 'poblacion_total', 
    'importe_total_PIE', 'idh', 'idh_salud', 'idh_educacion', 
    'idh_renta', 'proporcion_urbana', 'total_empresas'
]

municipal_subset = municipal_df[socioeconomic_vars]

# Realizar la unión
merged_df = pd.merge(
    housing_df,
    municipal_subset,
    how='left',
    left_on=['municipio_norm_mapped', 'provincia_norm_mapped', 'obtention_year'],
    right_on=['municipio_norm', 'provincia_norm', 'year']
)

# Verificar el resultado de la unión
print(f"\nDimensiones del dataset integrado: {merged_df.shape}")
print(f"Porcentaje de filas con datos municipales: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%")

# Para las filas sin correspondencia, intentar unir solo por provincia
if merged_df['poblacion_total'].notna().sum() / len(merged_df) < 0.5:
    print("\nIntentando unión solo por provincia...")
    # Calcular agregados a nivel provincial
    province_aggs = municipal_df.groupby(['provincia_norm', 'year']).agg({
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
        left_on=['provincia_norm_mapped', 'obtention_year'],
        right_on=['provincia_norm', 'year']
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
with open('diagnostico_integracion.txt', 'w') as f:
    f.write("# Informe de Diagnóstico de Integración de Datos\n\n")
    
    f.write("## Estadísticas de Provincias\n")
    f.write(f"- Provincias únicas en housing_df: {len(housing_provinces)}\n")
    f.write(f"- Provincias únicas en municipal_df: {len(municipal_provinces)}\n")
    f.write(f"- Provincias comunes: {len(set(housing_provinces) & set(municipal_provinces))}\n\n")
    
    f.write("## Estadísticas de Municipios\n")
    for province, counts in province_municipality_counts.items():
        f.write(f"### Provincia: {province}\n")
        f.write(f"- Municipios en housing_df: {counts['housing']}\n")
        f.write(f"- Municipios en municipal_df: {counts['municipal']}\n")
        f.write(f"- Municipios comunes: {counts['common']}\n")
        f.write(f"- Porcentaje de coincidencia: {(counts['common'] / counts['housing']) * 100:.2f}%\n\n")
    
    f.write("## Resultados de la Integración\n")
    f.write(f"- Dimensiones del dataset integrado: {merged_df.shape}\n")
    f.write(f"- Filas con datos municipales: {merged_df['poblacion_total'].notna().sum()}\n")
    f.write(f"- Porcentaje de filas con datos municipales: {(merged_df['poblacion_total'].notna().sum() / len(merged_df)) * 100:.2f}%\n")

print("Diagnóstico completado. Informe guardado en 'diagnostico_integracion.txt'")
