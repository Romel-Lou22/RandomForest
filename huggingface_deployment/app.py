from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json

app = FastAPI(title="API Detección de Alertas - Cultivo de Mora")

# Cargar modelo
model = joblib.load('modelos/random_forest_mora.joblib')

# Solution map
solution_map = {
    "ph_bajo": "Aplicar cal agrícola o enmiendas alcalinas para subir el pH del suelo.",
    "ph_alto": "Aplicar materia orgánica ácida o enmiendas para bajar ligeramente el pH.",
    "hum_baja": "Aumentar la frecuencia o duración del riego y revisar el sistema de riego.",
    "hum_alta": "Reducir los riegos y mejorar el drenaje para evitar encharcamientos.",
    "temp_baja": "Proteger el cultivo con coberturas o plásticos en las horas más frías.",
    "temp_alta": "Implementar sombra parcial y regar en horas frescas para reducir el estrés térmico.",
    "n_bajo": "Aplicar fertilizante nitrogenado según recomendación técnica y análisis de suelo.",
    "n_alto": "Reducir o suspender momentáneamente la fertilización nitrogenada.",
    "p_bajo": "Aplicar fertilizante fosforado adecuado para el cultivo y las condiciones del suelo.",
    "p_alto": "Reducir o suspender la aplicación de fósforo para evitar sobrefertilización.",
    "k_bajo": "Aplicar fertilizante potásico según las recomendaciones del análisis de suelo.",
    "k_alto": "Reducir o suspender temporalmente la aplicación de potasio."
}

alert_cols = [
    'ph_bajo', 'ph_alto', 'hum_baja', 'hum_alta',
    'temp_baja', 'temp_alta', 'n_bajo', 'n_alto',
    'p_bajo', 'p_alto', 'k_bajo', 'k_alto'
]

class SensorData(BaseModel):
    pH: float = Field(..., ge=0, le=14, description="pH del suelo")
    temperatura_C: float = Field(..., description="Temperatura en °C")
    humedad_suelo_pct: float = Field(..., ge=0, le=100, description="Humedad del suelo en %")
    N_ppm: float = Field(..., ge=0, description="Nitrógeno en ppm")
    P_ppm: float = Field(..., ge=0, description="Fósforo en ppm")
    K_ppm: float = Field(..., ge=0, description="Potasio en ppm")

@app.get("/")
def root():
    return {
        "mensaje": "API de Detección de Alertas para Cultivo de Mora",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predice alertas basadas en valores de sensores",
            "/docs": "Documentación interactiva"
        }
    }

@app.post("/predict")
def predict(data: SensorData):
    try:
        # Preparar input
        X = np.array([[
            data.pH,
            data.temperatura_C,
            data.humedad_suelo_pct,
            data.N_ppm,
            data.P_ppm,
            data.K_ppm
        ]])
        
        # Predecir
        predicciones = model.predict(X)[0]
        
        # Generar alertas
        alertas_detectadas = []
        todas_alertas = {}
        
        for i, alerta in enumerate(alert_cols):
            todas_alertas[alerta] = int(predicciones[i])
            if predicciones[i] == 1:
                alertas_detectadas.append({
                    'tipo': alerta,
                    'recomendacion': solution_map[alerta]
                })
        
        return {
            'alertas_detectadas': alertas_detectadas,
            'total_alertas': len(alertas_detectadas),
            'todas_alertas': todas_alertas,
            'valores_input': {
                'pH': data.pH,
                'temperatura_C': data.temperatura_C,
                'humedad_suelo_%': data.humedad_suelo_pct,
                'N_ppm': data.N_ppm,
                'P_ppm': data.P_ppm,
                'K_ppm': data.K_ppm
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")