import streamlit as st
from supabase import create_client

# Configuración de Supabase
url = "https://odlosqyzqrggrhvkdovj.supabase.co"  # Reemplaza con tu URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMDA2ODI4OSwiZXhwIjoyMDQ1NjQ0Mjg5fQ.h-pFrlUWOwEQpKNVZKm04SXflfG6q7KNH0hov9XZxvI"  # Reemplaza con tu API key
supabase = create_client(url, key)

# Mostrar la tabla "ventas"
def mostrar_ventas():
    data = supabase.table("ventas").select("*").execute()
    ventas = data['data']  # Obtén los datos
    
    st.title("Tabla de Ventas")
    st.write(ventas)

if st.button("Cargar Ventas"):
    mostrar_ventas()