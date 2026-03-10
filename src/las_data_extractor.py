import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import laspy
import cv2
import os
import sys
import argparse

# --- GESTIONE ARGOMENTI ---
parser = argparse.ArgumentParser()
parser.add_argument("las_path", help="Percorso del file .las")
parser.add_argument("--mode", default="SAM_CLASSIC", help="Modalità: SAM_CLASSIC o LANG_SAM")
args = parser.parse_args()

las_path = args.las_path
mode = args.mode

# --- CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"Caricamento nuvola in modalità: {mode}...")

try:
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
except Exception as e:
    print(f"ERRORE durante la lettura del file LAS: {e}")
    sys.exit(1)

# Calcolo offset per centrare la nuvola
offset = np.mean(points, axis=0)
points -= offset

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# --- LOGICA DI COLORAZIONE ---
if mode == "SAM_CLASSIC":
    print("Applicazione mappa MAGMA (High Contrast)...")
    z_values = points[:, 2]
    z_min = np.percentile(z_values, 1)
    z_max = np.percentile(z_values, 99)
    z_norm = np.clip((z_values - z_min) / (z_max - z_min + 1e-7), 0, 1)
    colors = plt.get_cmap('magma')(z_norm)[:, :3]

else:  # LANG_SAM
    z_values = points[:, 2]
    z_ground = np.percentile(z_values, 2)

    # Maschera per il pavimento (con background subtraction)
    floor_mask = z_values < (z_ground + 0.28)

    try:
        if hasattr(las, 'red'):
            print("Utilizzo colori originali RGB dal file LAS...")
            r = np.array(las.red) / 65535.0
            g = np.array(las.green) / 65535.0
            b = np.array(las.blue) / 65535.0
            colors = np.vstack((r, g, b)).transpose()
        else:
            raise AttributeError("Nessun dato RGB trovato")
    except:
        print("Colori RGB non trovati. Utilizzo scala di grigi basata sulla quota...")
        z_norm = np.clip((z_values - z_ground) / (np.ptp(z_values) + 1e-7), 0, 1)
        colors = np.column_stack((z_norm, z_norm, z_norm))

    # APPLICAZIONE BACKGROUND SUBTRACTION
    # Forziamo a colore nero tutti i punti identificati come 'floor'
    colors[floor_mask] = [0, 0, 0]
    colors[~floor_mask] = np.clip(colors[~floor_mask] * 1.2, 0, 1)

pcd.colors = o3d.utility.Vector3dVector(colors)

# --- SETUP VISUALIZZAZIONE ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Cattura Nadirale - {mode}", width=1920, height=1080, visible=True)
vis.add_geometry(pcd)

# --- CONFIGURAZIONE RENDERING ---
render_option = vis.get_render_option()
render_option.point_size = 2.5
render_option.background_color = np.array([0, 0, 0])

# --- POSIZIONAMENTO CAMERA ---
ctr = vis.get_view_control()
bbox = pcd.get_axis_aligned_bounding_box()
center = bbox.get_center()

ctr.set_front([0, 0, 1])
ctr.set_lookat(center)
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.7)

vis.poll_events()
vis.update_renderer()

print("Generazione vista...")

# --- CATTURA IMMAGINE, PROFONDITÀ E PARAMETRI CAMERA ---
image_float = vis.capture_screen_float_buffer(True)
depth_float = vis.capture_depth_float_buffer(True)
params = ctr.convert_to_pinhole_camera_parameters()

# --- CONVERSIONE IMMAGINE E PROFONDITÀ IN NUMPY ARRAY ---
image_np = (np.asarray(image_float) * 255).astype(np.uint8)
depth_np = np.asarray(depth_float)

# Salvataggio dati
np.save(os.path.join(data_dir, "color_data.npy"), image_np)
np.save(os.path.join(data_dir, "depth_data.npy"), depth_np)
np.savez(os.path.join(data_dir, "camera_params.npz"),
         intrinsic=params.intrinsic.intrinsic_matrix,
         extrinsic=params.extrinsic)
np.save(os.path.join(data_dir, "offset.npy"), offset)

# Salvataggio preview PNG
cv2.imwrite(os.path.join(data_dir, "color_preview.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

print(f"Successo! Vista salvata in {data_dir}.")
vis.destroy_window()