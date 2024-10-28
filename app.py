import streamlit as st
import supabase
import pickle
import numpy as np

# Cargar el modelo
with open("modelo.pkl", "rb") as file:
    modelo = pickle.load(file)
