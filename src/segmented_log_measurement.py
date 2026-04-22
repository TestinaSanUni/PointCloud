# (4)
# applica i filtri alle maschere ottenute tramite il modello
# esegue le misurazioni
#

import numpy as np
import os
import glob

# --- CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(base_dir), "data")
files_tronchi = glob.glob(os.path.join(data_dir, "tronco_*.xyz"))


def d_calc(points, asse_lungo, num_sezioni=3, percentile_esc=95):
    # Proiezione dei punti sull'asse principale
    proj_asse = points @ asse_lungo

    # Divisione in sezioni
    min_proj, max_proj = np.min(proj_asse), np.max(proj_asse)
    bordi_sezioni = np.linspace(min_proj, max_proj, num_sezioni + 1)

    diametri_sezioni = []
    for i in range(num_sezioni):
        mask = (proj_asse >= bordi_sezioni[i]) & (proj_asse < bordi_sezioni[i + 1])
        punti_sezione = points[mask]

        if len(punti_sezione) < 10:
            continue

        if abs(asse_lungo[2]) < 0.9:
            v1 = np.cross(asse_lungo, [0, 0, 1])
        else:
            v1 = np.cross(asse_lungo, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(asse_lungo, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Coordinate nel piano
        coord1 = punti_sezione @ v1
        coord2 = punti_sezione @ v2

        # Calcolo del diametro come massimo tra le estensioni
        if len(coord1) > 0:
            estensione1 = np.percentile(coord1, percentile_esc) - np.percentile(coord1, 100 - percentile_esc)
            estensione2 = np.percentile(coord2, percentile_esc) - np.percentile(coord2, 100 - percentile_esc)
            diametro_sezione = max(estensione1, estensione2)
            diametri_sezioni.append(diametro_sezione)

    if diametri_sezioni:
        return np.mean(diametri_sezioni), np.std(diametri_sezioni), diametri_sezioni
    else:
        return 0, 0, []

# --- ELABORAZIONE PRINCIPALE ---
if not files_tronchi:
    print("Nessun file trovato!")
else:
    report_righe = ["ID_Tronco;Lunghezza_m;Diametro_Medio_m;Volume_m3;Num_Punti\n"]
    print(f"{'ID':<12} | {'L (m)':<8} | {'D_medio (m)':<10} | {'Vol (m3)':<10} | {'#Punti':<8}")
    print("-" * 65)

    for file_path in sorted(files_tronchi, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
        id_t = os.path.basename(file_path).replace(".xyz", "")

        try:
            points = np.loadtxt(file_path)
        except:
            continue

        if len(points) < 200:
            continue

        # --- PULIZIA OUTLIER ---
        mean_p = np.mean(points, axis=0)
        dist = np.linalg.norm(points - mean_p, axis=1)
        points = points[dist < np.percentile(dist, 98)]

        # --- PCA ---
        points_centered = points - np.mean(points, axis=0)
        cov = np.cov(points_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, idx]
        asse_lungo = eig_vecs[:, 0]

        # 1. LUNGHEZZA
        projections_long = points @ asse_lungo
        lunghezza = np.max(projections_long) - np.min(projections_long)

        # 2. DIAMETRO
        d_medio, d_std, lista_diametri = d_calc(points, asse_lungo, num_sezioni=15)

        # 3. VOLUME (basato sul diametro medio)
        volume = np.pi * (d_medio / 2) ** 2 * lunghezza if d_medio > 0 else 0

        print(f"{id_t:<12} | {lunghezza:<8.2f} | {d_medio:<10.2f} | {volume:<10.4f} | {len(points):<8}")

        report_righe.append(f"{id_t};{lunghezza:.3f};{d_medio:.3f};{volume:.5f};{len(points)}\n")

    # Salvataggio report CSV
    report_path = os.path.join(data_dir, "report_tesi_tronchi.csv")
    with open(report_path, "w") as f:
        f.writelines(report_righe)

    print("-" * 55)
    print(f"Completato! Salvato in: {report_path}")