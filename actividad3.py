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

contact = np.array(data["contact"])
slg = np.array(data["slg"])
iso = np.array(data["iso"])
power_ratio = np.array(data["power_ratio"])
bb_rate = np.array(data["bb_rate"])
clutch = np.array(data["clutch"])
opi = np.array(data["opi"])
rc = np.array(data["rc"])

runs = np.array(data["runs"])
pred = np.array(data["predicciones"])
errors = np.array(data["errores"])

metrics = data["metrics"]


# 1. REAL vs PREDICHO
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(runs, label="Real", linewidth=2)
ax.plot(pred, label="Predicho", alpha=0.8)

ax.set_title("Runs: Real vs Predicho")
ax.set_xlabel("Jugador (índice)")
ax.set_ylabel("Runs")
ax.legend()
ax.grid()

add_metrics(ax,
    f"RMSE: {metrics['rmse_modelo']:.4f}\n"
    f"Pearson: {metrics['r_contact_runs']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/01_real_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()


# 2. RESIDUALES
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(range(len(errors)), errors, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")

ax.set_title("Residuales del modelo")
ax.set_xlabel("Jugador")
ax.set_ylabel("Error")
ax.grid()

add_metrics(ax,
    f"Error medio: {metrics['error_media']:.4f}\n"
    f"Std: {metrics['error_std']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/02_residuales.png", dpi=300, bbox_inches="tight")
plt.close()


# 3. HISTOGRAMA DE ERRORES
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(errors, bins=15, color="gray", edgecolor="black")

ax.set_title("Distribución de errores")
ax.set_xlabel("Error")
ax.set_ylabel("Frecuencia")
ax.grid()

add_metrics(ax,
    f"Varianza: {metrics['error_var']:.4f}\n"
    f"RMSE: {metrics['rmse_modelo']:.4f}"
)

plt.savefig(f"{OUTPUT_DIR}/03_histograma.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. IMPORTANCIA RELATIVA (proxy visual)
fig, ax = plt.subplots(figsize=(8, 5))

features = {
    "contact": contact,
    "slg": slg,
    "iso": iso,
    "power_ratio": power_ratio,
    "bb_rate": bb_rate,
    "clutch": clutch,
    "opi": opi,
    "rc": rc
}

means = {k: np.mean(np.abs(v)) for k, v in features.items()}

ax.bar(means.keys(), means.values(), color="steelblue")

ax.set_title("Magnitud media de features")
ax.set_ylabel("Valor medio absoluto")
ax.grid(axis="y")

plt.xticks(rotation=30)

add_metrics(ax, "Proxy de importancia de features\n(no es coeficiente real)")

plt.savefig(f"{OUTPUT_DIR}/04_features.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. ERROR vs PREDICCIÓN
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(pred, errors, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Error vs Predicción")
ax.set_xlabel("Predicción")
ax.set_ylabel("Error")
ax.grid()
plt.savefig(f"{OUTPUT_DIR}/05_error_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. RESUMEN FINAL
print("\n===== METRICAS =====")
print(json.dumps(metrics, indent=4))
print("\n===== RESUMEN =====")
print(f"Datos: {len(runs)} registros")
print(f"Graficas guardadas en: {OUTPUT_DIR}/")