import streamlit as st
import joblib
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, normalize_data, add_features
from model_train import (
    train_random_forest, 
    train_decision_tree, 
    train_linear_regression,
    train_svr,
    train_knn
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Conexión con Supabase
supabase = get_supabase_client()

# Columnas relevantes
selected_columns = [
    "fecha",
    "producto",
    "precio",
    "cantidad_vendida",
    "promocion",
    "inventario_inicial",
    "inventario_final",
    "desperdicio",
    "condiciones_climaticas",
    "ventas_por_hora"
]

# Cargar los datos
st.title("Gestión de Stock con Predicción")
df = load_and_select_data(supabase, "verduras", selected_columns)
df_clean = clean_data(df)
df_features = add_features(df_clean)

# Definir columnas de entrada y objetivo
feature_cols = [
    "precio",
    "cantidad_vendida",
    "inventario_inicial",
    "desperdicio",
    "diferencia_inventario",
    "porcentaje_desperdicio",
    "es_fin_de_semana"
]
target_col = "inventario_final"

# Pestañas de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una pestaña", ["Entrenar Modelo", "Consultar Predicciones"])

# Pestaña 1: Entrenar y Guardar el Modelo
if page == "Entrenar Modelo":
    st.header("Entrenar y Guardar el Modelo")
    
    # Seleccionar el modelo
    modelo_seleccionado = st.radio(
        "Selecciona un modelo para entrenar:", 
        ["Random Forest", "Árbol de Decisión", "Regresión Lineal", "SVR", "KNN"]
    )

    if st.button("Entrenar Modelo"):
        if modelo_seleccionado == "Random Forest":
            model, metrics = train_random_forest(df_features, target_col, feature_cols)
            model_path = "random_forest_model.pkl"
        elif modelo_seleccionado == "Árbol de Decisión":
            model, metrics = train_decision_tree(df_features, target_col, feature_cols)
            model_path = "decision_tree_model.pkl"
        elif modelo_seleccionado == "Regresión Lineal":
            model, metrics = train_linear_regression(df_features, target_col, feature_cols)
            model_path = "linear_regression_model.pkl"
        elif modelo_seleccionado == "SVR":
            model, metrics = train_svr(df_features, target_col, feature_cols)
            model_path = "svr_model.pkl"
        else:  # KNN
            model, metrics = train_knn(df_features, target_col, feature_cols)
            model_path = "knn_model.pkl"

        # Guardar el modelo entrenado
        joblib.dump(model, model_path)
        st.success(f"Modelo {modelo_seleccionado} entrenado y guardado en {model_path}")
        st.write("Métricas del modelo:", metrics)

# Pestaña 2: Consultar Predicciones
elif page == "Consultar Predicciones":
    st.header("Consultar Predicciones")

    # Seleccionar producto
    producto_seleccionado = st.selectbox("Selecciona el producto:", df["producto"].unique())
    producto_data = df[df["producto"] == producto_seleccionado]

    # Entrada de datos
    cantidad_actual = st.number_input("Cantidad actual en stock:", min_value=0, value=int(producto_data["inventario_final"].iloc[0]))
    promocion = st.checkbox("¿Promoción activa?")

    # Selección del modelo para predicción
    modelo_a_usar = st.radio("Selecciona el modelo a usar:", 
                             ["Random Forest", "Árbol de Decisión", "Regresión Lineal", "SVR", "KNN"])
    
    # Cargar el modelo seleccionado
    if modelo_a_usar == "Random Forest":
        model_path = "random_forest_model.pkl"
    elif modelo_a_usar == "Árbol de Decisión":
        model_path = "decision_tree_model.pkl"
    elif modelo_a_usar == "Regresión Lineal":
        model_path = "linear_regression_model.pkl"
    elif modelo_a_usar == "SVR":
        model_path = "svr_model.pkl"
    else:  # KNN
        model_path = "knn_model.pkl"

    try:
        model = joblib.load(model_path)
        st.success(f"Modelo {modelo_a_usar} cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")

    # Hacer la predicción
    if st.button("Predecir"):
        try:
            # Crear vector de características basado en feature_cols
            caracteristicas = [
                producto_data["precio"].iloc[0],  # precio
                producto_data["cantidad_vendida"].iloc[0],  # cantidad_vendida
                producto_data["inventario_inicial"].iloc[0],  # inventario_inicial
                producto_data["desperdicio"].iloc[0],  # desperdicio
                producto_data["diferencia_inventario"].iloc[0],  # diferencia_inventario
                producto_data["porcentaje_desperdicio"].iloc[0],  # porcentaje_desperdicio
                producto_data["es_fin_de_semana"].iloc[0]  # es_fin_de_semana
            ]

            # Realizar la predicción con el modelo cargado
            prediccion = model.predict([caracteristicas])

            # Mostrar el resultado
            st.write(f"Recomendación: Comprar {round(prediccion[0], 2)} unidades de {producto_seleccionado}")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")



