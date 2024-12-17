import streamlit as st
import joblib
import plotly.express as px
from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, add_features
from model_train import (
    train_random_forest, train_decision_tree, 
    train_linear_regression, train_svr, train_knn
)

# Conexión a Supabase
supabase = get_supabase_client()
selected_columns = ["fecha", "producto", "precio", "cantidad_vendida", 
                    "promocion", "inventario_inicial", "inventario_final", 
                    "desperdicio", "condiciones_climaticas", "ventas_por_hora"]

# Cargar y preparar datos
st.title("Gestión de Stock con Predicción")
df = load_and_select_data(supabase, "verduras", selected_columns)
df_clean = clean_data(df)
df_features = add_features(df_clean)

feature_cols = ["precio", "cantidad_vendida", "inventario_inicial", 
                "desperdicio", "diferencia_inventario", "porcentaje_desperdicio", "es_fin_de_semana"]
target_col = "inventario_final"

# Tabs de navegación
tab1, tab2 = st.tabs(["Entrenar Modelo", "Consultar Predicciones"])

# Tab 1: Entrenar Modelo
with tab1:
    st.header("Entrenar y Guardar el Modelo")
    modelo = st.selectbox("Selecciona un modelo", ["Random Forest", "Árbol de Decisión", "Regresión Lineal", "SVR", "KNN"])
    if st.button("Entrenar"):
        if modelo == "Random Forest":
            model, metrics = train_random_forest(df_features, target_col, feature_cols)
        elif modelo == "Árbol de Decisión":
            model, metrics = train_decision_tree(df_features, target_col, feature_cols)
        elif modelo == "Regresión Lineal":
            model, metrics = train_linear_regression(df_features, target_col, feature_cols)
        elif modelo == "SVR":
            model, metrics = train_svr(df_features, target_col, feature_cols)
        else:
            model, metrics = train_knn(df_features, target_col, feature_cols)
        joblib.dump(model, f"{modelo.lower().replace(' ', '_')}.pkl")
        st.write("Métricas del Modelo:", metrics)

# Tab 2: Consultar Predicciones
with tab2:
    st.header("Consultar Predicciones")
    producto = st.selectbox("Selecciona el producto", df["producto"].unique())
    modelo = st.radio("Modelo", ["Random Forest", "Árbol de Decisión"])
    if st.button("Predecir"):
        model = joblib.load(f"{modelo.lower().replace(' ', '_')}.pkl")
        st.write(f"Predicción para {producto}: {model.predict([[10, 20, 30, 40, 0, 0, 1]])[0]}")





