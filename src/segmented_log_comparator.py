import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestNeighbors

#TODO: Automatizzazione

# --- FUNZIONI DI UTILITÀ ---
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("XYZ files", "*.xyz")])
    root.destroy()
    return path

# --- NORMALIZZAZIONE RIGIDA (PCA + Orientamento) ---
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

# --- ALLINEAMENTO BOUNDING BOX ---
def align_bbox_center(A, B):
    center_A = (A[:, 0].min() + A[:, 0].max()) / 2
    center_B = (B[:, 0].min() + B[:, 0].max()) / 2
    shift = center_B - center_A
    A_aligned = A.copy()
    A_aligned[:, 0] += shift
    return A_aligned

# --- DOWNSAMPLING PER ICP ---
def downsample(points, n=15000):
    if len(points) <= n: return points
    idx = np.random.choice(len(points), n, replace=False)
    return points[idx]

# --- ICP PER MICRO-ALLINEAMENTO ---
def icp(A, B, max_iterations=40, tolerance=1e-6):
    src = A.copy()
    dst = B.copy()
    prev_error = 0
    R_final = np.eye(2)
    t_final = np.zeros(2)

    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1).fit(dst)
        distances, indices = nbrs.kneighbors(src)
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

# --- CALCOLO METRICA IoU ---
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

print("Caricamento e Normalizzazione PCA...")
data_a_raw = np.loadtxt(f_auto)[:, :2]
data_m_raw = np.loadtxt(f_man)[:, :2]
data_a_norm = rigid_normalization(data_a_raw)
data_m_norm = rigid_normalization(data_m_raw)

print("Ricerca dell'orientamento ottimale...")
best_iou_global = -1
final_aligned_a = None
best_masks = None

# Test delle 4 possibili simmetrie (Normal, FlipX, FlipY, FlipXY)
for mx in [1, -1]:
    for my in [1, -1]:
        current_test = data_a_norm.copy()
        current_test[:, 0] *= mx
        current_test[:, 1] *= my
        current_test = align_bbox_center(current_test, data_m_norm)
        a_small = downsample(current_test)
        m_small = downsample(data_m_norm)
        _, R, t = icp(a_small, m_small)
        aligned_full = (R @ current_test.T).T + t
        iou_tmp, m_a, m_m = calculate_iou_metric(aligned_full, data_m_norm)
        print(f" > Test [MirrorX: {mx:2}, MirrorY: {my:2}] -> IoU: {iou_tmp:.2%}")
        if iou_tmp > best_iou_global:
            best_iou_global = iou_tmp
            final_aligned_a = aligned_full
            best_masks = (m_a, m_m)

# --- VISUALIZZAZIONE FINALE ---
mask_a, mask_m = best_masks
cmap = ListedColormap(['#f0f0f0', '#e74c3c', '#3498db', '#2ecc71'])
viz = np.zeros(mask_a.shape)
viz[mask_m] += 1
viz[mask_a] += 2

plt.figure(figsize=(10, 8))
plt.imshow(viz, origin='lower', cmap=cmap)
plt.title(f"VALIDAZIONE METRICA FINALE\nIoU Ottimizzato: {best_iou_global:.2%}")
plt.axis('off')
print(f"\nAllineamento completato. Miglior IoU: {best_iou_global:.4f}")
plt.show()