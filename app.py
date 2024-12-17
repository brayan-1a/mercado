import streamlit as st
import joblib
import os
import requests

from config import get_supabase_client
from preprocess import load_and_select_data, clean_data, add_features

# Conexión con Supabase
supabase = get_supabase_client()

# Columnas relevantes
selected_columns = [
    "fecha", "producto", "precio", "cantidad_vendida", 
    "promocion", "inventario_inicial", "inventario_final", 
    "desperdicio", "condiciones_climaticas", "ventas_por_hora"
]

# Cargar y preparar datos
st.title("Gestión de Stock con Predicción")
df = load_and_select_data(supabase, "verduras", selected_columns)
df_clean = clean_data(df)
df_features = add_features(df_clean)

# Definir columnas de entrada y objetivo
feature_cols = [
    "precio", "cantidad_vendida", "inventario_inicial", 
    "desperdicio", "diferencia_inventario", "porcentaje_desperdicio", "es_fin_de_semana"
]
target_col = "inventario_final"

# Pestañas de navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una pestaña", ["Consultar Predicciones"])

# Pestaña 1: Consultar Predicciones
if page == "Consultar Predicciones":
    st.header("Consultar Predicciones")

    # Seleccionar producto
    producto_seleccionado = st.selectbox("Selecciona el producto:", df["producto"].unique())
    producto_data = df[df["producto"] == producto_seleccionado]

    # Entrada de datos
    cantidad_actual = st.number_input("Cantidad actual en stock:", min_value=0, value=int(producto_data["inventario_final"].iloc[0]))
    promocion = st.checkbox("¿Promoción activa?")

    # Ruta del modelo guardado en una ubicación conocida localmente
    model_path = "models/random_forest_model.pkl"

    # Cargar el modelo guardado localmente
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success(f"Modelo Random Forest cargado exitosamente desde {model_path}.")
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
    else:
        st.error(f"No se encontró el archivo {model_path}. Asegúrate de haber guardado correctamente el modelo.")

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







