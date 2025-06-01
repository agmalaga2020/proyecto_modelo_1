# Informe de Resultados: Modelo Predictivo de Precios de Vivienda

## 1. Resumen Ejecutivo

Este informe presenta los resultados del análisis y modelado predictivo de precios de vivienda, utilizando tanto variables propias de las viviendas como variables socioeconómicas municipales.

- **Mejor modelo general**: RandomForest con Con Socioeconómicas
- **R²**: 0.8091
- **RMSE**: 195641.75
- **MAE**: 97732.15

La inclusión de variables socioeconómicas municipales resultó en una mejora del 15.60% en R² y una reducción del 12.25% en RMSE.

## 2. Metodología

### 2.1 Datos Utilizados

- **Dataset de viviendas**: 130733 registros con información sobre características físicas y precios
- **Dataset municipal**: Variables socioeconómicas para 67756 registros

### 2.2 Preprocesamiento

- Limpieza de valores nulos y outliers
- Creación de variables derivadas
- Normalización de variables numéricas
- Codificación one-hot de variables categóricas

### 2.3 Modelos Evaluados

- Ridge Regression
- Random Forest

### 2.4 Validación

- División 80/20 para entrenamiento/prueba
- Métricas: RMSE, MAE, R²

## 3. Resultados Comparativos

### 3.1 Comparación de Modelos

| Modelo | Dataset | RMSE | MAE | R² |
|--------|---------|------|-----|----|
| Ridge | Solo Vivienda | 277234.72 | 162778.50 | 0.5360 |
| RandomForest | Solo Vivienda | 222941.68 | 119468.73 | 0.6999 |
| Ridge | Con Socioeconómicas | 296985.14 | 175579.27 | 0.5602 |
| RandomForest | Con Socioeconómicas | 195641.75 | 97732.15 | 0.8091 |

### 3.2 Impacto de Variables Socioeconómicas

- **Mejor modelo solo con variables de vivienda**: RandomForest, R² = 0.6999
- **Mejor modelo con variables socioeconómicas**: RandomForest, R² = 0.8091
- **Mejora en R²**: +15.60%
- **Mejora en RMSE**: -12.25%

## 4. Variables Más Importantes

### 4.1 Top 10 Variables

| Variable | Importancia |
|----------|------------|
| bath_num | 0.3827 |
| idh_educacion | 0.1474 |
| m2_real | 0.1447 |
| provincia_mapped_balears | 0.1260 |
| poblacion_total | 0.0719 |
| m2_useful | 0.0272 |
| idh_renta | 0.0229 |
| idh | 0.0119 |
| room_bath_ratio | 0.0082 |
| proporcion_urbana | 0.0073 |

### 4.2 Importancia por Tipo de Variable

- **Variables de Vivienda**: 0.5885 (58.85%)
- **Variables Socioeconómicas**: 0.2687 (26.87%)
- **Variables Categóricas**: 0.1427 (14.27%)

## 5. Conclusiones

1. Las variables socioeconómicas municipales tienen un impacto significativo en la predicción de precios de vivienda
2. El modelo de mejor rendimiento es capaz de explicar un alto porcentaje de la varianza en los precios
3. Las variables más importantes incluyen tanto características físicas de las viviendas como factores contextuales
4. La integración de datos municipales mejora sustancialmente la capacidad predictiva del modelo

## 6. Recomendaciones

1. Utilizar el modelo completo (con variables socioeconómicas) para obtener predicciones más precisas
2. Considerar la recopilación de datos adicionales sobre factores urbanos y de accesibilidad
3. Actualizar periódicamente los datos socioeconómicos para mantener la relevancia del modelo
4. Explorar la posibilidad de modelos específicos por región o tipo de vivienda
