import streamlit as st
import joblib
import requests
import io
import plotly.express as px
import pandas as pd
from config import MODEL_URL  # Importamos la URL del modelo desde config.py
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

# Cargar y preparar los datos
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
page = st.sidebar.radio("Selecciona una pestaña", ["Análisis de Datos", "Consultar Predicciones"])

# Pestaña 1: Análisis de Datos
if page == "Análisis de Datos":
    st.header("Análisis de Datos")

    # Mostrar los datos seleccionados
    st.subheader("Datos Crudos")
    st.dataframe(df)

    # Mostrar datos limpios y con características
    st.subheader("Datos Limpiados y Características Agregadas")
    st.dataframe(df_features)

    # Mostrar resumen estadístico
    st.subheader("Resumen Estadístico")
    st.write(df_features.describe())

    # Gráficos Interactivos
    st.subheader("Visualización de Datos")

    # Gráfico de barras para el stock inicial por producto
    fig_stock = px.bar(df_features, x="producto", y="inventario_inicial", title="Stock Inicial por Producto")
    st.plotly_chart(fig_stock)

    # Gráfico interactivo de líneas para la evolución del inventario final
    st.subheader("Evolución del Inventario Final por Producto")

    # Agregar un filtro para seleccionar un producto
    producto_seleccionado_grafico = st.selectbox("Selecciona un producto para ver su evolución:", df_features["producto"].unique())

    # Filtrar los datos solo para el producto seleccionado
    df_producto = df_features[df_features["producto"] == producto_seleccionado_grafico]

    # Crear el gráfico de líneas solo para el producto seleccionado
    fig_inventario = px.line(
        df_producto, 
        x="fecha", 
        y="inventario_final", 
        title=f"Evolución del Inventario Final - {producto_seleccionado_grafico}",
        markers=True  # Agregar puntos en las líneas para mayor claridad
    )
    st.plotly_chart(fig_inventario)

    # Gráfico de barras agrupadas: Promedio de Cantidad Vendida y Desperdicio por Producto
    st.subheader("Promedio de Cantidad Vendida y Desperdicio por Producto")

    # Calcular promedios
    df_resumen = df_features.groupby("producto")[["cantidad_vendida", "desperdicio"]].mean().reset_index()

    # Crear gráfico de barras agrupadas
    fig_resumen = px.bar(
        df_resumen, 
        x="producto", 
        y=["cantidad_vendida", "desperdicio"], 
        barmode="group", 
        title="Comparación Promedio: Cantidad Vendida vs Desperdicio",
        labels={"value": "Promedio", "variable": "Métrica"}
    )

    # Mostrar el gráfico
    st.plotly_chart(fig_resumen)

# Pestaña 2: Consultar Predicciones
elif page == "Consultar Predicciones":
    st.header("Consultar Predicciones")

    # Seleccionar producto
    producto_seleccionado = st.selectbox("Selecciona el producto:", df["producto"].unique())
    producto_data = df[df["producto"] == producto_seleccionado]

    # Entrada de datos
    cantidad_actual = st.number_input("Cantidad actual en stock:", min_value=0, value=int(producto_data["inventario_final"].iloc[0]))
    promocion = st.checkbox("¿Promoción activa?")

    # Cargar el modelo desde Supabase Storage usando la URL desde config.py
    try:
        response = requests.get(MODEL_URL)  # Usamos la URL del modelo desde config.py
        response.raise_for_status()  # Lanza un error si la respuesta no es 200
        model_data = io.BytesIO(response.content)
        model = joblib.load(model_data)
        st.success(f"Modelo Random Forest cargado exitosamente desde Supabase.")
    except Exception as e:
        st.error(f"Error al cargar el modelo desde Supabase: {e}")

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

            # Redondear la predicción a entero
            unidades_predichas = round(prediccion[0])  # O también puedes usar `int(prediccion[0])`

            # Convertir unidades a kilos
            def unidades_a_kilos(producto, unidades):
                if producto == "zanahoria":
                    kilos = unidades / 2.5  # Promedio entre 2-3 zanahorias
                elif producto == "cebolla":
                    kilos = unidades / 3.5  # Promedio entre 3-4 cebollas
                elif producto == "lechuga":
                    kilos = unidades / 2.5  # Promedio entre 2-3 lechugas
                elif producto == "pepino":
                    kilos = unidades / 2.5  # Promedio entre 2-3 pepinos
                elif producto == "tomate":
                    kilos = unidades / 5.5  # Promedio entre 5-6 tomates
                else:
                    kilos = unidades  # Por defecto si no hay especificación
                return kilos

            kilos_predichos = unidades_a_kilos(producto_seleccionado, unidades_predichas)

            # Mostrar los tres resultados
            st.write(f"Predicción (en unidades): {prediccion[0]:.2f} unidades de {producto_seleccionado}")
            st.write(f"Predicción (en unidades redondeadas): {unidades_predichas} unidades de {producto_seleccionado}")
            st.write(f"Predicción (en kilos): {kilos_predichos:.2f} kilos de {producto_seleccionado}")

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")













