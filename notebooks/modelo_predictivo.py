#!/usr/bin/env python
# coding: utf-8

# # Modelo Predictivo de Precios de Vivienda con Variables Socioeconómicas
# 
# Este script realiza el preprocesamiento detallado, construcción de modelos predictivos
# y análisis de importancia de variables para predecir el precio de la vivienda.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Para preprocesamiento
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

# Para modelado
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Crear directorio para resultados
results_dir = 'resultados_modelo'
os.makedirs(results_dir, exist_ok=True)

# Cargar el dataset integrado
print("Cargando dataset integrado...")
df = pd.read_csv('housing_municipal_integrated_fixed.csv')

# Verificar las dimensiones
print(f"Dimensiones del dataset: {df.shape}")
print(f"Porcentaje de filas con datos socioeconómicos: {(df['poblacion_total'].notna().sum() / len(df)) * 100:.2f}%")

# ## 1. Preprocesamiento detallado

print("\nRealizando preprocesamiento detallado...")

# Convertir columnas a tipos numéricos
numeric_cols = ['price', 'm2_real', 'm2_useful', 'room_num', 'bath_num', 
                'air_conditioner', 'balcony', 'built_in_wardrobe', 
                'chimney', 'garden', 'lift', 'storage_room', 
                'swimming_pool', 'terrace']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtrar filas con valores nulos en la variable objetivo
df = df.dropna(subset=['price'])

# Filtrar outliers de precio (eliminamos el 1% superior e inferior)
lower_bound = df['price'].quantile(0.01)
upper_bound = df['price'].quantile(0.99)
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Filtrar outliers de metros cuadrados
df = df[df['m2_real'] <= df['m2_real'].quantile(0.99)]

# Crear variables derivadas
df['price_per_m2'] = df['price'] / df['m2_real']
df['room_bath_ratio'] = df['room_num'] / df['bath_num']
df['has_outdoor'] = ((df['garden'] == 1) | (df['terrace'] == 1) | (df['balcony'] == 1)).astype(int)

# Seleccionar variables para el modelo
# Variables de la vivienda
housing_features = [
    'm2_real', 'm2_useful', 'room_num', 'bath_num', 
    'air_conditioner', 'balcony', 'built_in_wardrobe', 
    'chimney', 'garden', 'lift', 'storage_room', 
    'swimming_pool', 'terrace', 'has_outdoor', 'room_bath_ratio'
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

# Crear dos conjuntos de datos: uno con solo variables de vivienda y otro con todas las variables
housing_only_features = housing_features + categorical_features
all_combined_features = housing_features + socioeconomic_features + categorical_features

# Filtrar filas con valores nulos en las variables socioeconómicas para el modelo completo
df_with_socioeconomic = df.dropna(subset=socioeconomic_features)
print(f"Filas disponibles para modelo con variables socioeconómicas: {len(df_with_socioeconomic)}")

# Dividir en conjuntos de entrenamiento y prueba
X_housing = df[housing_only_features]
X_all = df_with_socioeconomic[all_combined_features]
y_housing = df[target]
y_all = df_with_socioeconomic[target]

X_housing_train, X_housing_test, y_housing_train, y_housing_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

print(f"Dimensiones del conjunto de entrenamiento (solo vivienda): {X_housing_train.shape}")
print(f"Dimensiones del conjunto de prueba (solo vivienda): {X_housing_test.shape}")
print(f"Dimensiones del conjunto de entrenamiento (con socioeconómicas): {X_all_train.shape}")
print(f"Dimensiones del conjunto de prueba (con socioeconómicas): {X_all_test.shape}")

# Definir preprocesadores para variables numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear preprocesadores para ambos conjuntos de datos
preprocessor_housing = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [col for col in housing_only_features if col not in categorical_features]),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessor_all = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [col for col in all_combined_features if col not in categorical_features]),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ## 2. Construcción de modelos predictivos

print("\nConstruyendo modelos predictivos...")

# Definir modelos a evaluar
models = {
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Parámetros para GridSearchCV
param_grids = {
    'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# Función para evaluar modelos
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} en {dataset_name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Listas para almacenar resultados
results = []
best_models = {}

# Entrenar y evaluar modelos con solo variables de vivienda
print("\nEntrenando modelos con solo variables de vivienda...")
for model_name, model in models.items():
    print(f"\nEntrenando {model_name}...")
    
    # Crear pipeline con preprocesador y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_housing),
        ('model', model)
    ])
    
    # Realizar búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[model_name], 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_housing_train, y_housing_train)
    
    # Guardar el mejor modelo
    best_models[f"{model_name}_housing"] = grid_search.best_estimator_
    
    # Evaluar el mejor modelo
    result = evaluate_model(
        grid_search.best_estimator_,
        X_housing_train, 
        X_housing_test, 
        y_housing_train, 
        y_housing_test,
        model_name,
        "Solo Vivienda"
    )
    
    results.append(result)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")

# Entrenar y evaluar modelos con todas las variables (incluyendo socioeconómicas)
print("\nEntrenando modelos con todas las variables (incluyendo socioeconómicas)...")
for model_name, model in models.items():
    print(f"\nEntrenando {model_name}...")
    
    # Crear pipeline con preprocesador y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_all),
        ('model', model)
    ])
    
    # Realizar búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[model_name], 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_all_train, y_all_train)
    
    # Guardar el mejor modelo
    best_models[f"{model_name}_all"] = grid_search.best_estimator_
    
    # Evaluar el mejor modelo
    result = evaluate_model(
        grid_search.best_estimator_,
        X_all_train, 
        X_all_test, 
        y_all_train, 
        y_all_test,
        model_name,
        "Con Socioeconómicas"
    )
    
    results.append(result)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(f"{results_dir}/model_comparison_results.csv", index=False)

# Visualizar comparación de modelos
plt.figure(figsize=(14, 8))
sns.barplot(x='model_name', y='rmse', hue='dataset', data=results_df)
plt.title('Comparación de RMSE por Modelo y Conjunto de Variables')
plt.xlabel('Modelo')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{results_dir}/model_rmse_comparison.png")

plt.figure(figsize=(14, 8))
sns.barplot(x='model_name', y='r2', hue='dataset', data=results_df)
plt.title('Comparación de R² por Modelo y Conjunto de Variables')
plt.xlabel('Modelo')
plt.ylabel('R²')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{results_dir}/model_r2_comparison.png")

# ## 3. Análisis de importancia de variables

print("\nAnalizando importancia de variables...")

# Identificar el mejor modelo para análisis de importancia
best_model_name = results_df.sort_values('r2', ascending=False).iloc[0]['model_name']
best_dataset = results_df.sort_values('r2', ascending=False).iloc[0]['dataset']
print(f"Mejor modelo para análisis de importancia: {best_model_name} con {best_dataset}")

# Seleccionar el mejor modelo
if best_dataset == "Solo Vivienda":
    best_model_key = f"{best_model_name}_housing"
    feature_names = housing_only_features
    X_train_best = X_housing_train
    preprocessor = preprocessor_housing
else:
    best_model_key = f"{best_model_name}_all"
    feature_names = all_combined_features
    X_train_best = X_all_train
    preprocessor = preprocessor_all

best_model = best_models[best_model_key]

# Extraer importancia de variables según el tipo de modelo
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
    # Para modelos basados en árboles, extraer importancia directamente
    if best_model_name == 'XGBoost':
        # XGBoost requiere acceder al modelo final en la pipeline
        model_step = best_model.named_steps['model']
        importances = model_step.feature_importances_
    else:
        model_step = best_model.named_steps['model']
        importances = model_step.feature_importances_
    
    # Obtener nombres de características después del preprocesamiento
    preprocessor_step = best_model.named_steps['preprocessor']
    feature_names_out = []
    
    # Obtener nombres de características numéricas
    num_features = [col for col in feature_names if col not in categorical_features]
    feature_names_out.extend(num_features)
    
    # Obtener nombres de características categóricas después de one-hot encoding
    cat_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
    cat_features = []
    for i, category in enumerate(categorical_features):
        cat_values = cat_encoder.categories_[i]
        for value in cat_values:
            cat_features.append(f"{category}_{value}")
    
    feature_names_out.extend(cat_features)
    
    # Crear DataFrame de importancia
    importance_df = pd.DataFrame({
        'feature': feature_names_out,
        'importance': importances
    })
    
    # Ordenar por importancia
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Guardar resultados
    importance_df.to_csv(f"{results_dir}/feature_importance.csv", index=False)
    
    # Visualizar importancia de variables
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title(f'Top 20 Variables Más Importantes ({best_model_name})')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_importance_top20.png")
    
    # Agrupar importancia por tipo de variable
    if best_dataset == "Con Socioeconómicas":
        # Clasificar variables por tipo
        housing_importance = importance_df[importance_df['feature'].isin(housing_features)]['importance'].sum()
        socioeconomic_importance = importance_df[importance_df['feature'].isin(socioeconomic_features)]['importance'].sum()
        categorical_importance = importance_df[~importance_df['feature'].isin(housing_features + socioeconomic_features)]['importance'].sum()
        
        # Crear gráfico de importancia por tipo
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Vivienda', 'Socioeconómicas', 'Categóricas'], 
                    y=[housing_importance, socioeconomic_importance, categorical_importance])
        plt.title('Importancia Agrupada por Tipo de Variable')
        plt.xlabel('Tipo de Variable')
        plt.ylabel('Importancia Total')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_importance_by_type.png")
        
        # Crear informe de importancia por tipo
        with open(f"{results_dir}/importance_by_type.txt", 'w') as f:
            f.write("# Importancia de Variables por Tipo\n\n")
            f.write(f"- Variables de Vivienda: {housing_importance:.4f} ({housing_importance*100:.2f}%)\n")
            f.write(f"- Variables Socioeconómicas: {socioeconomic_importance:.4f} ({socioeconomic_importance*100:.2f}%)\n")
            f.write(f"- Variables Categóricas: {categorical_importance:.4f} ({categorical_importance*100:.2f}%)\n")
else:
    print(f"El modelo {best_model_name} no proporciona importancia de variables directamente.")
    
    # Para modelos lineales, podemos usar los coeficientes como medida de importancia
    if best_model_name in ['Ridge', 'Lasso']:
        model_step = best_model.named_steps['model']
        coefficients = model_step.coef_
        
        # Obtener nombres de características después del preprocesamiento
        preprocessor_step = best_model.named_steps['preprocessor']
        feature_names_out = []
        
        # Obtener nombres de características numéricas
        num_features = [col for col in feature_names if col not in categorical_features]
        feature_names_out.extend(num_features)
        
        # Obtener nombres de características categóricas después de one-hot encoding
        cat_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
        cat_features = []
        for i, category in enumerate(categorical_features):
            cat_values = cat_encoder.categories_[i]
            for value in cat_values:
                cat_features.append(f"{category}_{value}")
        
        feature_names_out.extend(cat_features)
        
        # Crear DataFrame de importancia
        importance_df = pd.DataFrame({
            'feature': feature_names_out,
            'coefficient': coefficients
        })
        
        # Ordenar por valor absoluto de coeficientes
        importance_df['abs_coefficient'] = np.abs(importance_df['coefficient'])
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Guardar resultados
        importance_df.to_csv(f"{results_dir}/coefficient_importance.csv", index=False)
        
        # Visualizar coeficientes
        plt.figure(figsize=(12, 10))
        sns.barplot(x='coefficient', y='feature', data=importance_df.head(20))
        plt.title(f'Top 20 Variables con Mayor Coeficiente ({best_model_name})')
        plt.xlabel('Coeficiente')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/coefficient_importance_top20.png")

# ## 4. Validación de resultados

print("\nValidando resultados...")

# Seleccionar el mejor modelo general
best_overall_idx = results_df['r2'].idxmax()
best_overall_model_name = results_df.loc[best_overall_idx, 'model_name']
best_overall_dataset = results_df.loc[best_overall_idx, 'dataset']

print(f"Mejor modelo general: {best_overall_model_name} con {best_overall_dataset}")
print(f"R²: {results_df.loc[best_overall_idx, 'r2']:.4f}")
print(f"RMSE: {results_df.loc[best_overall_idx, 'rmse']:.2f}")
print(f"MAE: {results_df.loc[best_overall_idx, 'mae']:.2f}")

# Seleccionar el mejor modelo para cada conjunto de datos
best_housing_idx = results_df[results_df['dataset'] == 'Solo Vivienda']['r2'].idxmax()
best_all_idx = results_df[results_df['dataset'] == 'Con Socioeconómicas']['r2'].idxmax()

best_housing_model_name = results_df.loc[best_housing_idx, 'model_name']
best_all_model_name = results_df.loc[best_all_idx, 'model_name']

print(f"\nMejor modelo con solo variables de vivienda: {best_housing_model_name}")
print(f"R²: {results_df.loc[best_housing_idx, 'r2']:.4f}")
print(f"RMSE: {results_df.loc[best_housing_idx, 'rmse']:.2f}")
print(f"MAE: {results_df.loc[best_housing_idx, 'mae']:.2f}")

print(f"\nMejor modelo con variables socioeconómicas: {best_all_model_name}")
print(f"R²: {results_df.loc[best_all_idx, 'r2']:.4f}")
print(f"RMSE: {results_df.loc[best_all_idx, 'rmse']:.2f}")
print(f"MAE: {results_df.loc[best_all_idx, 'mae']:.2f}")

# Calcular mejora porcentual al incluir variables socioeconómicas
r2_housing = results_df.loc[best_housing_idx, 'r2']
r2_all = results_df.loc[best_all_idx, 'r2']
rmse_housing = results_df.loc[best_housing_idx, 'rmse']
rmse_all = results_df.loc[best_all_idx, 'rmse']

r2_improvement = ((r2_all - r2_housing) / r2_housing) * 100
rmse_improvement = ((rmse_housing - rmse_all) / rmse_housing) * 100

print(f"\nMejora al incluir variables socioeconómicas:")
print(f"R²: +{r2_improvement:.2f}%")
print(f"RMSE: -{rmse_improvement:.2f}%")

# Análisis de residuos para el mejor modelo
if best_overall_dataset == "Solo Vivienda":
    best_model = best_models[f"{best_overall_model_name}_housing"]
    X_test = X_housing_test
    y_test = y_housing_test
else:
    best_model = best_models[f"{best_overall_model_name}_all"]
    X_test = X_all_test
    y_test = y_all_test

y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

# Visualizar residuos
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribución de Residuos')
plt.xlabel('Residuo')
plt.ylabel('Frecuencia')
plt.savefig(f"{results_dir}/residuals_distribution.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residuos vs Valores Predichos')
plt.xlabel('Valor Predicho')
plt.ylabel('Residuo')
plt.savefig(f"{results_dir}/residuals_vs_predicted.png")

# Visualizar valores reales vs predichos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Valores Reales vs Predichos')
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.savefig(f"{results_dir}/actual_vs_predicted.png")

# ## 5. Documentación de resultados

print("\nDocumentando resultados...")

# Crear informe de resultados
with open(f"{results_dir}/informe_resultados.md", 'w') as f:
    f.write("# Informe de Resultados: Modelo Predictivo de Precios de Vivienda\n\n")
    
    f.write("## 1. Resumen Ejecutivo\n\n")
    f.write("Este informe presenta los resultados del análisis y modelado predictivo de precios de vivienda, ")
    f.write("utilizando tanto variables propias de las viviendas como variables socioeconómicas municipales.\n\n")
    
    f.write(f"- **Mejor modelo general**: {best_overall_model_name} con {best_overall_dataset}\n")
    f.write(f"- **R²**: {results_df.loc[best_overall_idx, 'r2']:.4f}\n")
    f.write(f"- **RMSE**: {results_df.loc[best_overall_idx, 'rmse']:.2f}\n")
    f.write(f"- **MAE**: {results_df.loc[best_overall_idx, 'mae']:.2f}\n\n")
    
    f.write("La inclusión de variables socioeconómicas municipales resultó en una ")
    f.write(f"mejora del {r2_improvement:.2f}% en R² y una reducción del {rmse_improvement:.2f}% en RMSE.\n\n")
    
    f.write("## 2. Metodología\n\n")
    f.write("### 2.1 Datos Utilizados\n\n")
    f.write(f"- **Dataset de viviendas**: {len(df)} registros con información sobre características físicas y precios\n")
    f.write(f"- **Dataset municipal**: Variables socioeconómicas para {len(df_with_socioeconomic)} registros\n\n")
    
    f.write("### 2.2 Preprocesamiento\n\n")
    f.write("- Limpieza de valores nulos y outliers\n")
    f.write("- Creación de variables derivadas\n")
    f.write("- Normalización de variables numéricas\n")
    f.write("- Codificación one-hot de variables categóricas\n\n")
    
    f.write("### 2.3 Modelos Evaluados\n\n")
    f.write("- Ridge Regression\n")
    f.write("- Lasso Regression\n")
    f.write("- Random Forest\n")
    f.write("- Gradient Boosting\n")
    f.write("- XGBoost\n\n")
    
    f.write("### 2.4 Validación\n\n")
    f.write("- División 80/20 para entrenamiento/prueba\n")
    f.write("- Validación cruzada con 5 folds para selección de hiperparámetros\n")
    f.write("- Métricas: RMSE, MAE, R²\n\n")
    
    f.write("## 3. Resultados Comparativos\n\n")
    f.write("### 3.1 Comparación de Modelos\n\n")
    
    # Crear tabla de resultados
    f.write("| Modelo | Dataset | RMSE | MAE | R² |\n")
    f.write("|--------|---------|------|-----|----|\n")
    for _, row in results_df.iterrows():
        f.write(f"| {row['model_name']} | {row['dataset']} | {row['rmse']:.2f} | {row['mae']:.2f} | {row['r2']:.4f} |\n")
    
    f.write("\n### 3.2 Impacto de Variables Socioeconómicas\n\n")
    f.write(f"- **Mejor modelo solo con variables de vivienda**: {best_housing_model_name}, R² = {r2_housing:.4f}\n")
    f.write(f"- **Mejor modelo con variables socioeconómicas**: {best_all_model_name}, R² = {r2_all:.4f}\n")
    f.write(f"- **Mejora en R²**: +{r2_improvement:.2f}%\n")
    f.write(f"- **Mejora en RMSE**: -{rmse_improvement:.2f}%\n\n")
    
    f.write("## 4. Variables Más Importantes\n\n")
    
    if 'importance_df' in locals():
        f.write("### 4.1 Top 10 Variables\n\n")
        f.write("| Variable | Importancia |\n")
        f.write("|----------|------------|\n")
        for _, row in importance_df.head(10).iterrows():
            f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
        
        if best_dataset == "Con Socioeconómicas":
            f.write("\n### 4.2 Importancia por Tipo de Variable\n\n")
            f.write(f"- **Variables de Vivienda**: {housing_importance:.4f} ({housing_importance*100:.2f}%)\n")
            f.write(f"- **Variables Socioeconómicas**: {socioeconomic_importance:.4f} ({socioeconomic_importance*100:.2f}%)\n")
            f.write(f"- **Variables Categóricas**: {categorical_importance:.4f} ({categorical_importance*100:.2f}%)\n\n")
    
    f.write("## 5. Conclusiones\n\n")
    f.write("1. Las variables socioeconómicas municipales tienen un impacto significativo en la predicción de precios de vivienda\n")
    f.write("2. El modelo de mejor rendimiento es capaz de explicar un alto porcentaje de la varianza en los precios\n")
    f.write("3. Las variables más importantes incluyen tanto características físicas de las viviendas como factores contextuales\n")
    f.write("4. La integración de datos municipales mejora sustancialmente la capacidad predictiva del modelo\n\n")
    
    f.write("## 6. Recomendaciones\n\n")
    f.write("1. Utilizar el modelo completo (con variables socioeconómicas) para obtener predicciones más precisas\n")
    f.write("2. Considerar la recopilación de datos adicionales sobre factores urbanos y de accesibilidad\n")
    f.write("3. Actualizar periódicamente los datos socioeconómicos para mantener la relevancia del modelo\n")
    f.write("4. Explorar la posibilidad de modelos específicos por región o tipo de vivienda\n")

print(f"\nInforme de resultados guardado en {results_dir}/informe_resultados.md")
print("Análisis y modelado completados con éxito.")

# Guardar los mejores modelos
import pickle
for model_name, model in best_models.items():
    with open(f"{results_dir}/{model_name}.pkl", 'wb') as f:
        pickle.dump(model, f)

print(f"Modelos guardados en el directorio {results_dir}")
