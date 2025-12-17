import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import json
from datetime import datetime

# Cargar datos
df = pd.read_csv('data/dataset_final_entrenamiento.csv', sep=';')

# Separar features y targets
feature_cols = ['pH', 'temperatura_C', 'humedad_suelo_%', 'N_ppm', 'P_ppm', 'K_ppm']
alert_cols = [
    'ph_bajo', 'ph_alto', 'hum_baja', 'hum_alta',
    'temp_baja', 'temp_alta', 'n_bajo', 'n_alto',
    'p_bajo', 'p_alto', 'k_bajo', 'k_alto'
]

X = df[feature_cols].values
y = df[alert_cols].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[:, 0]
)

print("=" * 60)
print("ENTRENAMIENTO DEL MODELO")
print("=" * 60)
print(f"Train: {len(X_train)} registros")
print(f"Test: {len(X_test)} registros")

# Entrenar Random Forest multi-output
print("\n⏳ Entrenando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

print("✓ Modelo entrenado")

# Evaluar en test
print("\n" + "=" * 60)
print("EVALUACIÓN EN TEST SET")
print("=" * 60)

y_pred = model.predict(X_test)

# Métricas por alerta
metricas = {}
print(f"{'Alerta':<12} {'Precision':<10} {'Recall':<10} {'F1-Score'}")
print("-" * 60)

for i, col in enumerate(alert_cols):
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
    recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
    f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
    
    metricas[col] = {
        'precision': round(float(precision), 3),
        'recall': round(float(recall), 3),
        'f1_score': round(float(f1), 3)
    }
    
    print(f"{col:<12} {precision:<10.3f} {recall:<10.3f} {f1:.3f}")

# F1 promedio
f1_avg = np.mean([metricas[col]['f1_score'] for col in alert_cols])
print("-" * 60)
print(f"{'PROMEDIO':<12} {'':<10} {'':<10} {f1_avg:.3f}")

# Guardar modelo
print("\n⏳ Guardando modelo...")
joblib.dump(model, 'modelos/random_forest_mora.joblib')

# Guardar metadata
metadata = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'n_registros_train': int(len(X_train)),
    'n_registros_test': int(len(X_test)),
    'features': feature_cols,
    'alertas': alert_cols,
    'f1_promedio': round(float(f1_avg), 3),
    'metricas_por_alerta': metricas,
    'hiperparametros': {
        'n_estimators': 100,
        'max_depth': 15,
        'class_weight': 'balanced'
    }
}

with open('modelos/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Modelo guardado: modelos/random_forest_mora.joblib")
print("✓ Metadata guardada: modelos/metadata.json")
print("\n✓ Entrenamiento completado exitosamente")