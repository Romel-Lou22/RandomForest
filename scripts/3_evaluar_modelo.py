import joblib
import numpy as np
import json

# Cargar modelo
model = joblib.load('modelos/random_forest_mora.joblib')
with open('modelos/metadata.json', 'r') as f:
    metadata = json.load(f)

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

alert_cols = metadata['alertas']
feature_cols = metadata['features']

def predecir_alertas(pH, temperatura_C, humedad_suelo, N_ppm, P_ppm, K_ppm):
    """Predice alertas y genera recomendaciones"""
    X = np.array([[pH, temperatura_C, humedad_suelo, N_ppm, P_ppm, K_ppm]])
    predicciones = model.predict(X)[0]
    
    alertas_detectadas = []
    for i, alerta in enumerate(alert_cols):
        if predicciones[i] == 1:
            alertas_detectadas.append({
                'tipo': alerta,
                'recomendacion': solution_map[alerta]
            })
    
    return alertas_detectadas

# Casos de prueba
print("=" * 60)
print("EVALUACIÓN DEL MODELO - CASOS DE PRUEBA")
print("=" * 60)

casos_prueba = [
    {
        'nombre': 'Caso 1: pH bajo, humedad baja',
        'valores': [4.8, 18, 40, 25, 30, 150]
    },
    {
        'nombre': 'Caso 2: Temperatura alta',
        'valores': [6.0, 30, 65, 25, 30, 150]
    },
    {
        'nombre': 'Caso 3: Nitrógeno bajo, fósforo bajo',
        'valores': [6.0, 18, 65, 12, 10, 150]
    },
    {
        'nombre': 'Caso 4: Condiciones óptimas',
        'valores': [6.0, 18, 65, 25, 30, 150]
    }
]

for caso in casos_prueba:
    print(f"\n{caso['nombre']}")
    print(f"Input: pH={caso['valores'][0]}, Temp={caso['valores'][1]}°C, Hum={caso['valores'][2]}%, "
          f"N={caso['valores'][3]}ppm, P={caso['valores'][4]}ppm, K={caso['valores'][5]}ppm")
    print("-" * 60)
    
    alertas = predecir_alertas(*caso['valores'])
    
    if alertas:
        for alerta in alertas:
            print(f"⚠️  {alerta['tipo'].upper()}")
            print(f"   → {alerta['recomendacion']}")
    else:
        print("✓ Sin alertas - Condiciones óptimas")

# Ejemplo de integración con API
print("\n" + "=" * 60)
print("EJEMPLO DE RESPUESTA PARA API")
print("=" * 60)

ejemplo_input = [5.2, 22, 50, 15, 20, 100]
alertas_api = predecir_alertas(*ejemplo_input)

respuesta_api = {
    'alertas_detectadas': alertas_api,
    'total_alertas': len(alertas_api)
}

print(json.dumps(respuesta_api, indent=2, ensure_ascii=False))

print("\n✓ Evaluación completada. Modelo listo para deployment.")