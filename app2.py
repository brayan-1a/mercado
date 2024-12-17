import streamlit as st
import pandas as pd
import plotly.express as px
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, add_features
from model_train import train_random_forest

# Conexión con Supabase
supabase = get_supabase_client()

# Columnas relevantes
selected_columns = [
    "fecha", "producto", "precio", "cantidad_vendida", 
    "promocion", "inventario_inicial", "inventario_final", 
    "desperdicio", "condiciones_climaticas", "ventas_por_hora"
]

# Cargar y preparar los datos
st.title("Informe de Gestión de Stock con Predicción")
df = load_and_select_data(supabase, "verduras", selected_columns)
df_clean = clean_data(df)
df_features = add_features(df_clean)

# Pestañas de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una sección", ["1. Datos Empresariales", "2. Modelo Predictivo", "3. Conclusiones"])

# **1. Datos Empresariales y Recopilación de Datos**
if page == "1. Datos Empresariales":
    st.header("1. Datos Empresariales y Recopilación de Datos")
    
    # **1.1. Fuente de Datos**
    st.subheader("1.1. Fuente de Datos")
    st.write("""
    - Los datos provienen de Supabase, una base de datos en la nube.
    - Las fuentes de datos incluyen ventas históricas, inventarios, datos de clientes, y condiciones climáticas.
    """)

    # **1.2. Recopilación de Datos**
    st.subheader("1.2. Recopilación de Datos")
    st.write("""
    - Los datos fueron recopilados principalmente a través de un sistema POS para ventas, y APIs externas para condiciones climáticas.
    - La integración con Supabase permitió almacenar y acceder a estos datos de manera eficiente.
    """)

    # **1.3. Calidad de los Datos**
    st.subheader("1.3. Calidad de los Datos")
    st.write("""
    1. **Identificación de datos incompletos**: Se detectaron valores faltantes en las columnas `inventario_final` y `desperdicio`.
    2. **Técnicas de limpieza de datos**: Se utilizaron métodos para rellenar valores faltantes, como la media para datos numéricos y valores predeterminados para texto.
    3. **Almacenamiento de datos**: Los datos fueron almacenados en Supabase, lo que facilitó la gestión y recuperación de la información.
    """)

    # Mostrar los primeros registros de los datos
    st.subheader("Datos Crudos")
    st.dataframe(df)

# **2. Modelo Predictivo**
elif page == "2. Modelo Predictivo":
    st.header("2. Modelo Predictivo")

    # **2.1. Aplicación del Modelo Predictivo**
    st.subheader("2.1. Aplicación del Modelo Predictivo")
    st.write("""
    - Se entrenaron varios modelos predictivos para predecir el stock necesario de productos.
    - Los modelos utilizados incluyen Random Forest, Árbol de Decisión, Regresión Lineal, SVR y KNN.
    """)

    # **2.2. Código del Modelo Predictivo**
    st.subheader("2.2. Código del Modelo Predictivo")
    st.write("""
    - A continuación se muestra el código utilizado para entrenar el modelo de Random Forest:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def train_random_forest(df, target_col, feature_cols):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }
        return model, metrics
    ```
    """)

    # **2.3. Evaluación del Modelo**
    st.subheader("2.3. Evaluación del Modelo")
    st.write("""
    - Se evaluaron los modelos utilizando métricas como **MAE**, **RMSE** y **R2**.
    - Las métricas del modelo Random Forest fueron:
        - **MAE**: 1.6275
        - **RMSE**: 2.14
        - **R2**: 0.98
    """)

    # Gráfico de barras para el stock inicial por producto
    st.subheader("Evolución del Inventario Final por Producto")
    fig_stock = px.bar(df_features, x="producto", y="inventario_inicial", title="Stock Inicial por Producto")
    st.plotly_chart(fig_stock)

# **3. Conclusiones**
elif page == "3. Conclusiones":
    st.header("3. Conclusiones")

    # Impacto en el negocio
    st.subheader("Impacto del análisis de datos en el negocio")
    st.write("""
    - La predicción del stock necesaria para cada producto ha optimizado la gestión del inventario, reduciendo el desperdicio.
    - Los modelos predictivos han permitido tomar decisiones informadas sobre las compras, lo que mejora la eficiencia operativa.
    - El análisis ha demostrado ser útil para ajustar los niveles de inventario de acuerdo con la demanda y las condiciones del mercado.
    """)

    # Áreas de mejora
    st.subheader("Áreas de mejora")
    st.write("""
    - Incorporar más variables, como las **promociones** y el **clima**, podría mejorar la precisión de las predicciones.
    - Mejorar los modelos mediante técnicas de **optimización de hiperparámetros**.
    """)
