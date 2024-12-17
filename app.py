import streamlit as st
import joblib
import os
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, add_features
from model_train import (
    train_random_forest, train_decision_tree, 
    train_linear_regression, train_svr, train_knn
)

# Conexión con Supabase
supabase = get_supabase_client()
selected_columns = ["fecha", "producto", "precio", "cantidad_vendida", 
                    "promocion", "inventario_inicial", "inventario_final", 
                    "desperdicio", "condiciones_climaticas", "ventas_por_hora"]

# Cargar y preparar datos
st.title("Entrenar y Descargar Modelos de Predicción")
st.header("Gestión de Stock")

# Cargar y preprocesar los datos
df = load_and_select_data(supabase, "verduras", selected_columns)
df_clean = clean_data(df)
df_features = add_features(df_clean)

feature_cols = ["precio", "cantidad_vendida", "inventario_inicial", 
                "desperdicio", "diferencia_inventario", "porcentaje_desperdicio", "es_fin_de_semana"]
target_col = "inventario_final"

# Selección del modelo
st.sidebar.title("Entrenamiento del Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona un modelo para entrenar:", 
    ["Random Forest", "Árbol de Decisión", "Regresión Lineal", "SVR", "KNN"]
)

# Entrenar el modelo seleccionado
if st.sidebar.button("Entrenar Modelo"):
    st.subheader(f"Entrenando Modelo: {modelo_seleccionado}")
    
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

    # Guardar el modelo temporalmente
    joblib.dump(model, model_path)
    
    # Mostrar métricas
    st.write("### Métricas del Modelo:")
    st.json(metrics)
    st.success(f"Modelo {modelo_seleccionado} entrenado y guardado temporalmente.")

    # Descargar el modelo
    with open(model_path, "rb") as file:
        st.download_button(
            label="Descargar Modelo",
            data=file,
            file_name=model_path,
            mime="application/octet-stream"
        )

    # Eliminar el archivo después de la descarga (opcional, para limpieza)
    os.remove(model_path)





