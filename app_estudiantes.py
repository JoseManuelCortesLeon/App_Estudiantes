import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Configuración de la página web
st.set_page_title("Predictor Académico MLP", layout="centered")

st.title("🎓 Sistema Inteligente de Alerta Académica (MLP)")
st.write("Ingrese los hábitos y calificaciones del estudiante para predecir en tiempo real si aprobará o reprobará la asignatura.")

# 1. Cargar datos y entrenar el modelo automáticamente para el servicio web
@st.cache_resource
def entrenar_modelo_web():
    # Cargar dataset
    df = pd.read_csv("dataset_estudiantes_balanceado.csv")
    features = ['Tareas', 'Examenes', 'Asistencia', 'Estudio_Horas', 'Sueno_Horas', 'Extracurriculares']
    
    X = df[features]
    y = df['Aprobado']
    
    # Imputación de nulos por mediana
    X = X.fillna(X.median())
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Arquitectura MLP optimizada
    model = Sequential([
        Input(shape=(X_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=40, batch_size=16, verbose=0)
    
    return model, scaler

# Cargar el modelo entrenado
model, scaler = entrenar_modelo_web()

st.sidebar.header("📝 Parámetros del Estudiante")

# Controles interactivos (Sliders y Inputs) en la barra lateral
tareas = st.sidebar.slider("Calificación de Tareas (0-100)", 0.0, 100.0, 75.0)
examenes = st.sidebar.slider("Calificación de Exámenes (0-100)", 0.0, 100.0, 70.0)
asistencia = st.sidebar.slider("Porcentaje de Asistencia (0-100)", 0.0, 100.0, 85.0)
estudio_horas = st.sidebar.number_input("Horas de Estudio Semanal", 0.0, 40.0, 10.0)
sueno_horas = st.sidebar.number_input("Horas de Sueño Diario", 0.0, 12.0, 7.0)
extracurriculares = st.sidebar.selectbox("Participa en Extracurriculares (1=Sí, 0=No)", [0, 1])

# Botón de predicción
if st.button("🚀 Evaluar Desempeño del Estudiante"):
    # Organizar los datos ingresados
    entrada_usuario = np.array([[tareas, examenes, asistencia, estudio_horas, sueno_horas, extracurriculares]])
    
    # Escalar la entrada con el mismo StandardScaler entrenado
    entrada_escalada = scaler.transform(entrada_usuario)
    
    # Obtener probabilidad del modelo
    probabilidad = model.predict(entrada_escalada)[0][0]
    porcentaje = probabilidad * 100
    
    st.subheader("📊 Resultado del Análisis:")
    st.write(f"**Probabilidad estimada de aprobación:** `{porcentaje:.2f}%`")
    
    # Mostrar alerta visual según el resultado
    if probabilidad > 0.5:
        st.success("🎉 **Resultado:** El estudiante tiene alta probabilidad de **APROBAR**. ¡Excelente desempeño!")
    else:
        st.error("⚠️ **Resultado:** El estudiante tiene alta probabilidad de **REPROBAR**. Se sugiere intervención pedagógica oportuna.")