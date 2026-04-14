import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_FILE = "actividad3.json"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. CARGAR JSON
# =========================
with open(INPUT_FILE, "r") as f: data = json.load(f)

hits = np.array(data["hits"])
runs = np.array(data["runs"])
pred = np.array(data["predicciones"])
errores = np.array(data["errores"])

metrics = data["metrics"]

# =========================
# 2. SCATTER + REGRESIÓN
# =========================
plt.figure(figsize=(8,5))
plt.scatter(hits, runs, label="Datos reales", alpha=0.7)
plt.plot(hits, pred, color="red", label="Modelo (predicción)")
plt.title("Hits vs Runs")
plt.xlabel("Hits")
plt.ylabel("Runs")
plt.legend()
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/scatter_regresion.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 3. RESIDUALES (ERRORES)
# =========================
plt.figure(figsize=(8,5))
plt.scatter(hits, errores, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Errores del modelo (residuales)")
plt.xlabel("Hits")
plt.ylabel("Error (pred - real)")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/residuales.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 4. HISTOGRAMA DE ERRORES
# =========================
plt.figure(figsize=(8,5))
plt.hist(errores, bins=15, color="gray", edgecolor="black")
plt.title("Distribución de errores")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/histograma_errores.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 5. REAL vs PREDICHO
# =========================
plt.figure(figsize=(8,5))
plt.plot(runs, label="Real", marker="o")
plt.plot(pred, label="Predicho", marker="x")
plt.title("Comparación Real vs Predicho")
plt.xlabel("Índice")
plt.ylabel("Runs")
plt.legend()
plt.grid()

plt.savefig(f"{OUTPUT_DIR}/comparacion_real_vs_predicho.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 6. MÉTRICAS
# =========================
print("\n===== MÉTRICAS DEL MODELO =====")
print("Pearson:", metrics["pearson"])
print("RMSE:", metrics["rmse"])
print("Error medio:", metrics["error_media"])
print("Desviación estándar:", metrics["error_std"])
print("Varianza:", metrics["error_var"])

# =========================
# 7. RESUMEN FINAL
# =========================
print("\n===== RESUMEN =====")
print(f"Datos cargados: {len(hits)} registros")
print(f"Gráficas guardadas en: {OUTPUT_DIR}/")