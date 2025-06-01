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

```
proyecto_modelo_1/
├── data/
│   ├── raw/                 # Datos originales sin procesar
│   │   ├── houses_*.csv     # Datos de viviendas por provincia
│   │   ├── rentas_*.csv     # Datos de rentas
│   │   └── ...              # Otros datasets originales
│   └── processed/           # Datos procesados y limpios
│       ├── train_data*.csv  # Datos de entrenamiento
│       ├── test_data*.csv   # Datos de prueba
│       └── housing_municipal_integrated*.csv
├── notebooks/               # Jupyter notebooks y scripts de Python
│   ├── correccion_*.py      # Scripts de corrección de datos
│   ├── modelo_*.py          # Scripts de modelado
│   └── diagnostico_*.py     # Scripts de diagnóstico
├── models/
│   └── trained/             # Modelos entrenados (.pkl)
│       ├── RandomForest_*.pkl
│       └── Ridge_*.pkl
├── docs/
│   ├── reports/             # Informes y resultados
│   │   ├── informe_resultados.md
│   │   ├── model_comparison_results.csv
│   │   └── ...
│   └── images/              # Visualizaciones y gráficos
│       ├── feature_importance_*.png
│       ├── model_comparison_*.png
│       └── ...
├── web/                     # Sitio web del proyecto
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── images/
└── temp/                    # Archivos temporales
```

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
**[Ver Demo](https://agmalaga2020.github.io/proyecto_modelo_1/web/)**

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
