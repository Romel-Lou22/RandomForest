# 🍇 API Detección de Alertas - Cultivo de Mora

Sistema de notificación inteligente para detección de condiciones anómalas en cultivo de mora usando Random Forest.

## 🚀 Uso

### Endpoint Principal: `/predict`

**Método:** POST

**Body (JSON):**
```json
{
  "pH": 5.8,
  "temperatura_C": 22.5,
  "humedad_suelo_pct": 45.0,
  "N_ppm": 120,
  "P_ppm": 35,
  "K_ppm": 180
}
```

**Respuesta:**
```json
{
  "alertas_detectadas": [
    {
      "tipo": "hum_baja",
      "recomendacion": "Aumentar la frecuencia o duración del riego..."
    }
  ],
  "total_alertas": 1,
  "todas_alertas": {
    "ph_bajo": 0,
    "ph_alto": 0,
    "hum_baja": 1,
    ...
  }
}
```

## 📊 Variables de Entrada

- **pH**: pH del suelo (0-14)
- **temperatura_C**: Temperatura ambiente en °C
- **humedad_suelo_pct**: Humedad del suelo en % (0-100)
- **N_ppm**: Nitrógeno en ppm
- **P_ppm**: Fósforo en ppm
- **K_ppm**: Potasio en ppm

## 🎯 Alertas Detectadas

El modelo detecta 12 tipos de alertas:
- pH bajo/alto
- Humedad baja/alta
- Temperatura baja/alta
- NPK bajo/alto

## 🛠️ Tecnologías

- FastAPI
- Random Forest (scikit-learn)
- Python 3.10

## 📱 Integración

Ideal para apps móviles agrícolas, sistemas IoT y dashboards de monitoreo.
