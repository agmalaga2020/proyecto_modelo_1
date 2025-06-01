# Proyecto de Análisis y Predicción del Mercado Inmobiliario Español

Este proyecto analiza datos del mercado inmobiliario español, integrando información de diferentes provincias y municipios para crear un modelo predictivo de precios de vivienda.

Repositorio: [https://github.com/agmalaga2020/proyecto_modelo_1](https://github.com/agmalaga2020/proyecto_modelo_1)

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

## Demo en Línea

La interfaz web del proyecto está disponible en GitHub Pages:
**[Ver Demo](https://agmalaga2020.github.io/proyecto_modelo_1/website/)**

La web incluye:
- Resumen ejecutivo del análisis
- Metodología utilizada
- Resultados comparativos de modelos
- Análisis de variables más importantes
- Conclusiones y recomendaciones

## Configuración de GitHub Pages

Para activar GitHub Pages en tu repositorio:

1. Ve a la configuración de tu repositorio en GitHub
2. Busca la sección "Pages" en el menú lateral
3. En "Source", selecciona "Deploy from a branch"
4. Selecciona la rama "main" o "master"
5. Selecciona la carpeta "/ (root)" 
6. Haz clic en "Save"

La web estará disponible en: `https://[tu-usuario].github.io/[nombre-repositorio]/website/`
