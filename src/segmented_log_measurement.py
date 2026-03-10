import numpy as np
import os
import glob

# --- CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(base_dir), "data")
files_tronchi = glob.glob(os.path.join(data_dir, "tronco_*.xyz"))

if not files_tronchi:
    print("Nessun file trovato!")
else:
    report_righe = ["ID_Tronco;Lunghezza_m;Diametro_Maggiore_m;Volume_m3;Num_Punti\n"]
    print(f"{'ID':<12} | {'L (m)':<8} | {'D_Max (m)':<10} | {'Vol (m3)':<10}")
    print("-" * 55)

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

        # 1. LUNGHEZZA
        projections_long = points @ eig_vecs[:, 0]
        lunghezza = np.max(projections_long) - np.min(projections_long)

        # 2. DIAMETRO MAGGIORE
        proj_d1 = points @ eig_vecs[:, 1]
        proj_d2 = points @ eig_vecs[:, 2]

        # Calcoliamo le estensioni usando il percentile 2-98
        w1 = np.percentile(proj_d1, 98) - np.percentile(proj_d1, 2)
        w2 = np.percentile(proj_d2, 98) - np.percentile(proj_d2, 2)
        d_maggiore = max(w1, w2)

        # 3. VOLUME (Basato sul diametro maggiore)
        volume = np.pi * (d_maggiore / 2) ** 2 * lunghezza

        print(f"{id_t:<12} | {lunghezza:<8.2f} | {d_maggiore:<10.2f} | {volume:<10.4f}")
        report_righe.append(f"{id_t};{lunghezza:.3f};{d_maggiore:.3f};{volume:.5f};{len(points)}\n")

    # Salvataggio report CSV
    report_path = os.path.join(data_dir, "report_tesi_tronchi.csv")
    with open(report_path, "w") as f:
        f.writelines(report_righe)

    print("-" * 55)
    print(f"Completato! Salvato in: {report_path}")