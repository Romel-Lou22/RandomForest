import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/dataset_final_entrenamiento.csv', sep=';')

print("=" * 60)
print("ANÁLISIS EXPLORATORIO - DATASET MORA")
print("=" * 60)

# Información básica
print(f"\nTotal de registros: {len(df)}")
print(f"Total de columnas: {len(df.columns)}")

# Verificar valores nulos
print(f"\nValores nulos: {df.isnull().sum().sum()}")

# Columnas de features y alertas
feature_cols = ['pH', 'temperatura_C', 'humedad_suelo_%', 'N_ppm', 'P_ppm', 'K_ppm']
alert_cols = [
    'ph_bajo', 'ph_alto', 'hum_baja', 'hum_alta',
    'temp_baja', 'temp_alta', 'n_bajo', 'n_alto',
    'p_bajo', 'p_alto', 'k_bajo', 'k_alto'
]

# Estadísticas de features
print("\n" + "=" * 60)
print("ESTADÍSTICAS DE VARIABLES DE ENTRADA")
print("=" * 60)
print(df[feature_cols].describe().round(2))

# Balance de clases para alertas
print("\n" + "=" * 60)
print("BALANCE DE ALERTAS (CLASES)")
print("=" * 60)
print(f"{'Alerta':<12} {'Positivos':<10} {'Negativos':<10} {'%Positivos':<12} {'Estado'}")
print("-" * 60)

desbalanceadas = []
for col in alert_cols:
    positivos = df[col].sum()
    negativos = len(df) - positivos
    porcentaje = (positivos / len(df)) * 100
    
    if porcentaje < 5:
        estado = "⚠️  MUY BAJO"
        desbalanceadas.append(col)
    elif porcentaje < 15:
        estado = "⚠️  BAJO"
        desbalanceadas.append(col)
    elif porcentaje > 85:
        estado = "⚠️  MUY ALTO"
        desbalanceadas.append(col)
    else:
        estado = "✓  BALANCEADO"
    
    print(f"{col:<12} {positivos:<10} {negativos:<10} {porcentaje:<12.2f} {estado}")

# Resumen de desbalance
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
if desbalanceadas:
    print(f"⚠️  {len(desbalanceadas)} alertas desbalanceadas: {', '.join(desbalanceadas)}")
    print("   → Se aplicará class_weight='balanced' en el entrenamiento")
else:
    print("✓  Todas las alertas están balanceadas")

print("\n✓ Análisis completado. Los datos están listos para entrenar.")