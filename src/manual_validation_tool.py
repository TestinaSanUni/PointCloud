import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

# --- CONFIGURAZIONE PERCORSI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
output_csv = os.path.join(data_dir, "validazione_manuale.csv")

# Lettura di tutti i file .xyz
files = sorted(glob.glob(os.path.join(data_dir, "tronco_*.xyz")))

if len(files) == 0:
    print(f"ERRORE: Non ho trovato file 'tronco_*.xyz' in: {data_dir}")
    exit()

print(f"Trovati {len(files)} file in {data_dir}.")
print("-" * 30)
print("ORDINE CLICK PER OGNI TRONCO:")
print("1-2: Diametro TESTA | 3-4: Diametro CODA | 5-6: LUNGHEZZA")
print("-" * 30)

# Creazione file CSV se non esiste
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["file", "lunghezza_manuale_m", "d_testa_manuale_cm", "d_coda_manuale_cm"])

for f in files:
    try:
        points = np.loadtxt(f)
    except Exception as e:
        print(f"File {f} saltato: {e}")
        continue

    if points.size == 0: continue

    x, y = points[:, 0], points[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=2, c='black', alpha=0.4)
    ax.set_title(f"File: {os.path.basename(f)}\n1-2: D_Testa | 3-4: D_Coda | 5-6: Lunghezza")
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.3)

    try:
        fig.canvas.manager.window.attributes('-topmost', 1)
        fig.canvas.manager.window.attributes('-topmost', 0)
    except:
        pass

    points_clicked = []

    # Ciclo per i 6 click
    for i in range(6):
        p = plt.ginput(1, timeout=0)
        if p:
            points_clicked.append(p[0])
            ax.plot(p[0][0], p[0][1], 'ro', markersize=6)
            ax.text(p[0][0], p[0][1], f" {i + 1}", color='red', weight='bold')
            plt.draw()

    if len(points_clicked) == 6:
        pts = np.array(points_clicked)
        d_testa = np.linalg.norm(pts[0] - pts[1])
        d_coda = np.linalg.norm(pts[2] - pts[3])
        lunghezza = np.linalg.norm(pts[4] - pts[5])

        # Salvataggio nel CSV
        with open(output_csv, "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([os.path.basename(f), round(lunghezza, 3), round(d_testa * 100, 2), round(d_coda * 100, 2)])

        print(f"Registrato {os.path.basename(f)}: L={lunghezza:.2f}m, D1={d_testa * 100:.1f}cm, D2={d_coda * 100:.1f}cm")

    plt.close(fig)

print(f"\nOperazione conclusa. File salvato in: {output_csv}")