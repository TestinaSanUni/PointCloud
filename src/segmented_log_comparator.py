import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestNeighbors
import time

#TODO: Automaticamente identificare e calcolare IoU dei tronchi di un certo file

# --- FUNZIONI DI UTILITÀ ---
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("XYZ files", "*.xyz")])
    root.destroy()
    return path

def rigid_normalization(points):
    pts = points[:, :2].copy()
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    primary_axis = evecs[:, np.argmax(evals)]
    angle = np.arctan2(primary_axis[1], primary_axis[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    return pts @ R.T

def downsample(points, n=10000):
    if len(points) <= n: return points
    idx = np.random.choice(len(points), n, replace=False)
    return points[idx]

# --- ICP OTTIMIZZATO ---
def icp_optimized(A, B, nbrs_model, max_iterations=16, tolerance=1e-6):
    src = A.copy()
    dst = B.copy()

    prev_error = 0
    R_final = np.eye(2)
    t_final = np.zeros(2)

    for i in range(max_iterations):
        distances, indices = nbrs_model.kneighbors(src)
        closest = dst[indices[:, 0]]

        centroid_src = np.mean(src, axis=0)
        centroid_dst = np.mean(closest, axis=0)

        AA = src - centroid_src
        BB = closest - centroid_dst
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_dst - R @ centroid_src
        src = (R @ src.T).T + t

        R_final = R @ R_final
        t_final = R @ t_final + t

        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance: break
        prev_error = mean_error

    return src, R_final, t_final

def calculate_iou_metric(pts_a, pts_m, res=0.01):
    all_p = np.vstack([pts_a, pts_m])
    x_min, y_min = all_p.min(axis=0) - 0.05
    x_max, y_max = all_p.max(axis=0) + 0.05
    w, h = int((x_max - x_min) / res) + 1, int((y_max - y_min) / res) + 1
    m1, m2 = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
    for p, mask in [(pts_a, m1), (pts_m, m2)]:
        c = np.clip(((p[:, 0] - x_min) / res).astype(int), 0, w - 1)
        r = np.clip(((p[:, 1] - y_min) / res).astype(int), 0, h - 1)
        mask[r, c] = True
    union = np.logical_or(m1, m2).sum()
    return (np.logical_and(m1, m2).sum() / union if union > 0 else 0), m1, m2

# --- ESECUZIONE ---
f_auto = select_file("Seleziona AUTOMATICO")
f_man = select_file("Seleziona MANUALE")
if not f_auto or not f_man: exit()

start_time = time.time()

print("Caricamento...")
data_a_raw = np.loadtxt(f_auto)[:, :2]
data_m_raw = np.loadtxt(f_man)[:, :2]
data_a_norm = rigid_normalization(data_a_raw)
data_m_norm = rigid_normalization(data_m_raw)
nbrs_manual = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(data_m_norm)
best_iou_global = -1
final_aligned_a = None
best_masks = None

print("Ricerca dell'orientamento ottimale...")
for mx in [1, -1]:
    for my in [1, -1]:
        # Simmetria
        current_flip = data_a_norm.copy()
        current_flip[:, 0] *= mx
        current_flip[:, 1] *= my

        # Test delle 3 posizioni di partenza per evitare massimi locali
        for align_mode in ['left', 'center', 'right']:
            current_test = current_flip.copy()

            if align_mode == 'left':
                shift = data_m_norm[:, 0].min() - current_test[:, 0].min()
            elif align_mode == 'right':
                shift = data_m_norm[:, 0].max() - current_test[:, 0].max()
            else:
                shift = ((data_m_norm[:, 0].min() + data_m_norm[:, 0].max()) / 2) - \
                        ((current_test[:, 0].min() + current_test[:, 0].max()) / 2)

            current_test[:, 0] += shift

            # Downsample per velocizzare l'ICP
            a_small = downsample(current_test, n=5000)

            _, R, t = icp_optimized(a_small, data_m_norm, nbrs_manual)
            aligned_full = (R @ current_test.T).T + t
            iou_tmp, m_a, m_m = calculate_iou_metric(aligned_full, data_m_norm)
            print(f" > [Flip: {mx},{my} | Pos: {align_mode:7}] -> IoU: {iou_tmp:.2%}")

            if iou_tmp > best_iou_global:
                best_iou_global = iou_tmp
                final_aligned_a = aligned_full
                best_masks = (m_a, m_m)

end_time = time.time()
print(f"\nAllineamento completato in {end_time - start_time:.2f} secondi.")
print(f"Miglior IoU trovato: {best_iou_global:.4f}")

# --- VISUALIZZAZIONE FINALE ---
mask_a, mask_m = best_masks
cmap = ListedColormap(['#f0f0f0', '#e74c3c', '#3498db', '#2ecc71'])
viz = np.zeros(mask_a.shape)
viz[mask_m] += 1
viz[mask_a] += 2

plt.figure(figsize=(10, 8))
plt.imshow(viz, origin='lower', cmap=cmap)
plt.title(f"VALIDAZIONE FINALE\nIoU: {best_iou_global:.2%}")
plt.axis('off')
plt.show()