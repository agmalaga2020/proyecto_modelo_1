# Proyecto de Análisis y Predicción del Mercado Inmobiliario Español

Este proyecto analiza datos del mercado inmobiliario español, integrando información de diferentes provincias y municipios para crear un modelo predictivo de precios de vivienda.

## Contenido del Proyecto

- Datos de viviendas por provincias (`houses_*.csv`)
- Datos integrados de viviendas a nivel municipal (no incluidos en el repositorio por limitaciones de tamaño)
- Notebooks de análisis y modelado
- Modelo predictivo implementado
- Visualizaciones y resultados del modelo
- Interfaz web simple para consultar predicciones

## Archivos Grandes

Debido a las limitaciones de GitHub, los siguientes archivos no están incluidos en el repositorio:
- `spanish_houses.csv` (93.93 MB)
- `housing_municipal_integrated.csv` (101.94 MB)
- `housing_municipal_integrated_fixed.csv` (140.65 MB)

Estos archivos son necesarios para ejecutar algunos de los notebooks. Por favor, contacta con el propietario del repositorio para obtener estos archivos.

## Estructura del Proyecto

- `/notebooks/`: Contiene los scripts de Python para la limpieza de datos, integración y modelado
- `/resultados_modelo/`: Almacena los resultados del modelo, gráficos y el modelo entrenado
- `/website/`: Una interfaz web simple para interactuar con el modelo

## Modelos Implementados

- Random Forest
- Ridge Regression

## Visualizaciones

El proyecto incluye varias visualizaciones como:
- Distribución de precios
- Importancia de características
- Comparación de modelos
- Análisis de residuos

## Cómo Usar

1. Explorar los notebooks para entender el proceso de análisis y modelado
2. Revisar los resultados en la carpeta `resultados_modelo`
3. Utilizar la interfaz web para realizar predicciones
