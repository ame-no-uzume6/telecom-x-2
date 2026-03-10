## Challenge Telecom X – Parte 2: Predicción de Cancelación (Churn)

Este repositorio contiene la implementación del **Challenge Telecom X – Parte 2**, enfocada en la **predicción de cancelación de clientes** para una empresa de telecomunicaciones.  
El objetivo es construir un flujo completo de **ciencia de datos / machine learning** que vaya desde los datos ya tratados hasta la evaluación de varios modelos de clasificación y el análisis de los factores que más influyen en la evasión de clientes.

---

### 1. Estructura relevante del proyecto

- `telecom_x_2.ipynb`  
  Notebook principal de la **Parte 2** del challenge. Contiene todo el flujo:
  - Carga de los datos tratados (archivo CSV generado en la Parte 1).
  - Preparación de datos para modelado (encoding, balanceo, normalización).
  - Análisis de correlación y análisis dirigido de variables.
  - Entrenamiento de múltiples modelos de clasificación.
  - Evaluación con métricas y matrices de confusión.
  - Análisis de importancia de variables y conclusiones estratégicas.

- `telecom_x_limpio.csv`  
  Conjunto de datos ya **limpios y tratados** al finalizar la Parte 1 del desafío (extracción y tratamiento inicial de datos).  
  Este archivo es el **punto de partida** de `telecom_x_2.ipynb` y debe estar en el mismo directorio que el notebook.

- `requirements.txt`  
  Lista de dependencias necesarias para ejecutar el notebook (pandas, scikit-learn, imbalanced-learn, seaborn, etc.).

---

### 2. Configuración del entorno

Se recomienda utilizar un **entorno virtual** (por ejemplo, con `venv`) para aislar las dependencias del proyecto.

Desde la carpeta del proyecto (donde está `telecom_x_2.ipynb`):

```bash
python -m venv .venv
```

Activar el entorno virtual:

- PowerShell (Windows):

```bash
.\.venv\Scripts\Activate.ps1
```

- CMD (Windows):

```bash
.\.venv\Scripts\activate.bat
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Registrar el kernel de Jupyter asociado al entorno (opcional pero recomendado):

```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name ".venv (Telecom X)"
```

Después de esto, en tu entorno de Jupyter / VSCode / Cursor podrás seleccionar el kernel `.venv (Telecom X)` para ejecutar el notebook.

---

### 3. Flujo de trabajo dentro de `telecom_x_2.ipynb`

El notebook está organizado en secciones numeradas que reflejan el pipeline típico de ML:

1. **Preparación de los datos**  
   - Carga de `telecom_x_limpio.csv` en un `DataFrame` `df`.  
   - Inspección inicial de columnas y tipos.

2. **Encoding de variables categóricas**  
   - Separación de variables explicativas (`X`) y objetivo (`y = Evasion`).  
   - Identificación automática de columnas categóricas (`dtype=object`).  
   - Uso de `OneHotEncoder` dentro de `make_column_transformer` (siguiendo el estilo de otros notebooks del curso) para generar `X_encoded`.

3. **Proporción de cancelación (churn)**  
   - Cálculo de conteos y proporciones de `Evasion` usando `value_counts(normalize=True)`.  
   - Verificación de **desbalance** entre clases.

4. **Balanceo de clases con SMOTE**  
   - Uso de `SMOTE` (`imbalanced-learn`) sobre `X_encoded` e `y`.  
   - Obtención de `X_balanceada` y `y_balanceada` con una distribución equilibrada entre clases.

5. **Normalización / estandarización**  
   - Aplicación de `StandardScaler` sobre `X_balanceada` cuando es necesario (modelos sensibles a escala, como Regresión Logística, KNN, SVM).

6. **Correlación y selección de variables**  
   - Construcción de una matriz de correlación.  
   - Cálculo de la correlación de cada variable con `Evasion`.  
   - Visualización de las variables con mayor correlación absoluta con la cancelación.

7. **Análisis dirigido de variables clave**  
   - Visualizaciones específicas:
     - Tipo de contrato × evasión (gráficos de barras).  
     - Cargos Totales × evasión (boxplots).  
     - Cargos Totales vs Cargos Mensuales coloreado por evasión (scatter plots).  
   - Este bloque ayuda a entender las relaciones de negocio más allá de la correlación global.

8. **Separación en entrenamiento y prueba**  
   - Uso de `train_test_split` sobre los datos balanceados (`X_balanceada`, `y_balanceada`) con `test_size=0.3` y `stratify=y_balanceada`.  
   - Se obtienen `X_train`, `X_test`, `y_train`, `y_test`.

9. **Entrenamiento de modelos de clasificación**  
   Se entrenan y comparan varios modelos:
   - **Regresión Logística** (con `X_train_scaled`, `X_test_scaled`).  
   - **KNN** (igualmente con datos escalados).  
   - **Random Forest** (trabaja directamente con `X_train`, `X_test` sin necesidad de normalización).  
   - **SVM lineal** (con datos escalados, para poder interpretar coeficientes).

10. **Evaluación de modelos**  
    - Cálculo de:
      - Accuracy  
      - Precisión  
      - Recall  
      - F1-score  
    - Generación de **matrices de confusión** para cada modelo (`ConfusionMatrixDisplay`).  
    - Análisis más cualitativo de posibles indicios de overfitting / underfitting comparando rendimiento en entrenamiento y prueba.

11. **Importancia de variables**  
    Distintos enfoques según el modelo:
    - **Regresión Logística**: análisis de los coeficientes (y su valor absoluto) para ver la contribución de cada variable.  
    - **Random Forest**: uso de `feature_importances_` para medir cómo contribuye cada variable a la reducción de impureza.  
    - **KNN**: uso de *permutation importance* para identificar qué variables impactan más en la exactitud del modelo.  
    - **SVM lineal**: análisis de los coeficientes de la frontera de decisión (vector de pesos del hiperplano separador).

12. **Conclusiones y recomendaciones de negocio**  
    - Resumen de los modelos que mejor funcionaron (típicamente Random Forest y KNN).  
    - Identificación de los **factores más relevantes para el churn**:
      - Tipo de contrato (mes a mes vs anual/bianual).  
      - Cargos Totales y Cargos Mensuales (especialmente clientes de alto gasto).  
      - Meses de antigüedad (tenure).  
      - Servicios de internet y adicionales.  
      - Método de pago y facturación.  
    - Propuesta de **estrategias de retención** basadas en estos hallazgos.

---

### 4. Cómo extender o modificar el proyecto

Algunas ideas para otros desarrolladores que quieran seguir iterando sobre este trabajo:

- **Incorporar más modelos**:  
  Añadir por ejemplo `GradientBoosting`, `XGBoost` o `LightGBM` (teniendo en cuenta sus dependencias y la gestión del desbalance de clases).

- **Buscar mejores hiperparámetros**:  
  Implementar `GridSearchCV` o `RandomizedSearchCV` para:
  - Profundidad y número de árboles en Random Forest.  
  - Número de vecinos y métricas de distancia en KNN.  
  - Parámetros de regularización en Regresión Logística y SVM.

- **Validación cruzada más robusta**:  
  Sustituir la simple separación entrenamiento/prueba por esquemas de **validación cruzada estratificada** para obtener estimaciones más estables del desempeño.

- **Exportar modelos y pipeline**:  
  Serializar el pipeline completo (preprocesamiento + modelo) con `joblib` o `pickle` para usarlo en una API o en un proceso batch.

---

### 5. Recomendaciones de uso para nuevos desarrolladores

- Ejecuta el notebook **sección por sección**, leyendo los comentarios de markdown, para entender la intención de cada paso.  
- Si modificas la estructura de `telecom_x_limpio.csv` (por ejemplo, añadiendo o renombrando columnas), revisa las partes de:
  - Selección de columnas categóricas,  
  - Definición de `columnas_categoricas`,  
  - Visualizaciones que referencian por nombre columnas específicas.
- Si obtienes errores de importación, verifica que:
  - Estás usando el **kernel correcto** (`.venv (Telecom X)` o el que tenga instaladas las dependencias).  
  - Has ejecutado `pip install -r requirements.txt` en el entorno actual.

Con estas instrucciones deberías poder reproducir, entender y extender el análisis de churn de **Telecom X – Parte 2** sin depender de otros archivos del proyecto.
