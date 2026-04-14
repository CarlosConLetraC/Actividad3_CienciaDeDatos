import os
import json
import numpy as np
import matplotlib.pyplot as plt

# CONFIG
INPUT_FILE = "data/actividad3.json"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.system(f"rm -rf {OUTPUT_DIR}/*")

# HELPERS
def add_metrics(ax, text):
    ax.text(
        0.05, 0.95,
        text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )


def load_data(path):
    with open(path, "r") as f: return json.load(f)


# LOAD DATA
data = load_data(INPUT_FILE)

hits = np.array(data["hits"])
runs = np.array(data["runs"])
pred = np.array(data["predicciones"])
errors = np.array(data["errores"])
metrics = data["metrics"]

# 1. SCATTER + REGRESIÓN
fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(hits, runs, alpha=0.7, label="Real")
ax.plot(hits, pred, color="red", label="Modelo")

ax.set_title("Hits vs Runs")
ax.set_xlabel("Hits")
ax.set_ylabel("Runs")
ax.legend()
ax.grid()

add_metrics(ax,
    f"Pearson: {metrics['pearson']:.4f}\n"
    f"RMSE: {metrics['rmse']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/01_scatter.png", dpi=300, bbox_inches="tight")
plt.close()


# 2. RESIDUALES
fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(hits, errors, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")

ax.set_title("Residuales del modelo")
ax.set_xlabel("Hits")
ax.set_ylabel("Error")
ax.grid()

add_metrics(ax,
    f"Error medio: {metrics['error_media']:.4f}\n"
    f"Std: {metrics['error_std']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/02_residuales.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. HISTOGRAMA DE ERRORES
fig, ax = plt.subplots(figsize=(8,5))

ax.hist(errors, bins=15, color="gray", edgecolor="black")

ax.set_title("Distribución de errores")
ax.set_xlabel("Error")
ax.set_ylabel("Frecuencia")
ax.grid()

add_metrics(ax,
    f"Varianza: {metrics['error_var']:.4f}\n"
    f"RMSE: {metrics['rmse']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/03_histograma.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. REAL vs PREDICHO
fig, ax = plt.subplots(figsize=(8,5))

ax.plot(runs, label="Real")
ax.plot(pred, label="Predicho")

ax.set_title("Real vs Predicho")
ax.set_xlabel("Índice")
ax.set_ylabel("Runs")
ax.legend()
ax.grid()

add_metrics(ax,
    f"Pearson: {metrics['pearson']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/04_comparacion.png", dpi=300, bbox_inches="tight")
plt.close()


# 5. RESUMEN FINAL
print("\n===== MÉTRICAS =====")
print(json.dumps(metrics, indent=4))

print("\n===== RESUMEN =====")
print(f"Datos: {len(hits)} registros")
print(f"Gráficas en: {OUTPUT_DIR}/")
