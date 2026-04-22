# (3)
# applica il modello SAM o Lang SAM
#

import numpy as np
import cv2
import torch
import open3d as o3d
import os
import glob
import sys
import argparse
import csv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GESTIONE ARGOMENTI ---
parser = argparse.ArgumentParser()
parser.add_argument("las_path", help="Percorso del file .las")
parser.add_argument("--mode", default="SAM_CLASSIC", help="SAM_CLASSIC o LANG_SAM")
args = parser.parse_args()

# --- CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
model_dir = os.path.abspath(os.path.join(base_dir, "..", "model"))
checkpoint_path = os.path.join(model_dir, "sam_vit_h_4b8939.pth")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Pulizia file precedenti
for f in glob.glob(os.path.join(data_dir, "tronco_*")):
    try:
        os.remove(f)
    except:
        pass

# --- CARICAMENTO DATI COMUNI ---
print(f"Caricamento dati per modalità: {args.mode}...")

# Caricamento della nuvola di punti originale
print("Caricamento nuvola di punti originale...")
try:
    import laspy

    las = laspy.read(args.las_path)
    points_original = np.vstack((las.x, las.y, las.z)).transpose()
except Exception as e:
    print(f"ERRORE durante la lettura del file LAS: {e}")
    sys.exit(1)

# Caricamento dell'offset applicato durante la generazione della vista
offset = np.load(os.path.join(data_dir, "offset.npy"))
print(f"Offset caricato: {offset}")

# Caricamento dei dati della camera
image_np = np.load(os.path.join(data_dir, "color_data.npy"))
params = np.load(os.path.join(data_dir, "camera_params.npz"))
intrinsic = params['intrinsic']
extrinsic = params['extrinsic']
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]

h, w, _ = image_np.shape
print(f"Immagine caricata: {w}x{h}")

# --- PREPARAZIONE NUVOLA PER PROIEZIONE ---
print("Preparazione nuvola per proiezione...")
points_centered = points_original - offset
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_centered)

# Costruzione della matrice di proiezione della camera
proj_matrix = intrinsic @ extrinsic[:3, :]

# Pre-calcolo delle coordinate dei punti nello spazio immagine
print("Proiezione punti nello spazio immagine...")
points_homo = np.hstack([points_centered, np.ones((len(points_centered), 1))])
points_camera = (extrinsic @ points_homo.T).T
points_camera = points_camera[:, :3]

# Proiezione dei punti sull'immagine (Z > 0)
valid_camera = points_camera[:, 2] > 0
points_camera_valid = points_camera[valid_camera]

x_proj = (points_camera_valid[:, 0] * fx / points_camera_valid[:, 2]) + cx
y_proj = (points_camera_valid[:, 1] * fy / points_camera_valid[:, 2]) + cy

# Creazione della mappa pixel - punti
print("Costruzione mappa pixel -> punti 3D...")
pixel_to_points = {}
x_int = np.round(x_proj).astype(int)
y_int = np.round(y_proj).astype(int)
in_image = (x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h)
x_int = x_int[in_image]
y_int = y_int[in_image]
indici_originali = np.where(valid_camera)[0][in_image]
for idx_pixel, (x, y, idx_point) in enumerate(zip(x_int, y_int, indici_originali)):
    key = (x, y)
    if key not in pixel_to_points:
        pixel_to_points[key] = []
    pixel_to_points[key].append(idx_point)

print(f"Mappa creata: {len(pixel_to_points)} pixel coperti da {len(indici_originali)} punti")


# --- MODELLO 1: SAM CLASSIC ---
def run_sam_classic(image):
    print("Inizializzazione SAM Classic (Automatic Mask Generator)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device=device)

        # FIXME: valori di partenza 48, 0.92, 0.92, 1800
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.90,
            min_mask_region_area=1800
        )

        return mask_generator.generate(image)

    except RuntimeError as e:
        if "indices" in str(e) or "CUDA" in str(e):
            print(f"\033[31mATTENZIONE: Errore GPU sul file corrente. Switch su CPU in corso...\033[0m")
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            sam.to(device="cpu")
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=48,
                pred_iou_thresh=0.90,
                stability_score_thresh=0.90,
                min_mask_region_area=1800
            )
            return mask_generator.generate(image)
        else:
            raise e


# --- MODELLO 2: LANG-SAM ---
def run_lang_sam(image):
    print("Inizializzazione Lang-SAM (Text-to-Segmentation)...")
    from lang_sam import LangSAM
    from PIL import Image

    model = LangSAM()
    image_pil = Image.fromarray(image).convert("RGB")

    text_prompt = "single wood log . single wood stick . single stick . single log"
    results = model.predict([image_pil], [text_prompt], box_threshold=0.20, text_threshold=0.20) # FIXME: valori originali box_threshold=0.25, text_threshold=0.20

    formatted_masks = []
    masks_data = results[0]['masks']
    boxes_data = results[0]['boxes']
    scores_data = results[0]['scores']
    num_masks = masks_data.shape[0] if hasattr(masks_data, 'shape') else len(masks_data)

    for i in range(num_masks):
        m = masks_data[i]
        if hasattr(m, 'cpu'): m = m.cpu().numpy()

        b = boxes_data[i]
        if hasattr(b, 'cpu'): b = b.cpu().numpy()

        s = scores_data[i]
        if hasattr(s, 'item'): s = s.item()

        formatted_masks.append({
            'segmentation': m.astype(bool),
            'area': np.sum(m),
            'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]],
            'score': s
        })
    return formatted_masks


# --- ESECUZIONE SELEZIONATA ---
if args.mode == "SAM_CLASSIC":
    masks = run_sam_classic(image_np)
else:
    masks = run_lang_sam(image_np)

# --- LOGICA DI FILTRAGGIO, ESTRAZIONE E PROIEZIONE ---
print(f"Elaborazione maschere e riproiezione 3D...")
# FIXME: valori iniziali 0.10 e 0.0010
max_area_consentita = (h * w) * 0.05
min_area_consentita = (h * w) * 0.002

overlay_all = image_np.copy()
count_tronchi = 0
report_score = []

# ordina le maschere in base alla confidenza
if args.mode == "SAM_CLASSIC":
    masks = sorted(masks, key=(lambda x: x['predicted_iou']), reverse=True)
else:
    masks = sorted(masks, key=(lambda x: x['score']), reverse=True)

canvas_occupato = np.zeros((h, w), dtype=bool)

for i, mask_data in enumerate(masks):
    mask = mask_data['segmentation']
    area = mask_data['area']

    # Filtro area
    if area > max_area_consentita or area < min_area_consentita:
        continue

    # controlla la sovrapposizione tra le maschere -> tiene quella a confidenza maggiore
    sovrapposizione = np.logical_and(canvas_occupato, mask).sum() / area
    if sovrapposizione > 0.25:
        continue

    # Recupero score in base al modello usato
    if args.mode == "SAM_CLASSIC":
        score = mask_data.get('predicted_iou', 0.0)
    else:
        score = mask_data.get('score', 0.0)

    v, u = np.where(mask > 0)

    # Individuazione degli indici dei punti 3D che cadono in questi pixel
    punti_indices = []
    punti_profondita = []

    for u_pix, v_pix in zip(u, v):
        key = (u_pix, v_pix)
        if key in pixel_to_points:
            for idx_punto in pixel_to_points[key]:
                punti_indices.append(idx_punto)
                punti_profondita.append(points_camera[idx_punto, 2])

    punti_indices = np.array(punti_indices)
    punti_profondita = np.array(punti_profondita)

    # Filtro di unicità punti
    if len(np.unique(punti_indices)) < 10:
        continue

    # Calcolo delle metriche di profondità
    if len(punti_profondita) > 0:
        z_range = np.max(punti_profondita) - np.min(punti_profondita)
        z_std = np.std(punti_profondita)
        z_media = np.mean(punti_profondita)
    else:
        continue

    # Filtro di spessore (adattivo basato sulla distanza)
    soglia_range = max(0.05, z_media * 0.02) # FIXME: valore originale max(0.05, z_media * 0.02)
    if args.mode == "SAM_CLASSIC":
        if z_range < soglia_range or z_std < 0.02:
            continue

    # Filtro di colore
    pixel_valori = image_np[mask]
    if np.mean(pixel_valori) < 80 and args.mode == "SAM_CLASSIC":
        continue

    # Maschera valida
    punti_unici = np.unique(punti_indices)
    points_3d_centered = points_centered[punti_unici]
    points_3d_original = points_3d_centered + offset

    count_tronchi += 1

    canvas_occupato = np.logical_or(canvas_occupato, mask)

    # Aggiunta dati al report
    report_score.append({'id': count_tronchi, 'score': round(float(score), 4)})

    print(f"DEBUG: tronco {count_tronchi} | area: {area} | punti: {len(points_3d_original)} | z_media: {z_media:.2f}m")

    # Salvataggio tronco
    np.savetxt(os.path.join(data_dir, f"tronco_{count_tronchi}.xyz"), points_3d_original)

    # Preview ritagliata
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x_b, y_b, w_b, h_b = cv2.boundingRect(contours[0])
        pad = 20
        y1, y2 = max(0, y_b - pad), min(h, y_b + h_b + pad)
        x1, x2 = max(0, x_b - pad), min(w, x_b + w_b + pad)
        img_crop = image_np[y1:y2, x1:x2].copy()
        cv2.imwrite(os.path.join(data_dir, f"tronco_{count_tronchi}_view.png"),
                    cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))

        color = np.random.randint(0, 255, (3,)).tolist()
        cv2.drawContours(overlay_all, contours, -1, color, 3)
        cv2.putText(overlay_all, f"ID:{count_tronchi}", (int(u.mean()), int(v.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- SCRITTURA FILE CSV ---
csv_path = os.path.join(data_dir, "tronchi_score.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score'], delimiter=';')
    writer.writeheader()
    for item in report_score:
        writer.writerow(item)

cv2.imwrite(os.path.join(data_dir, "mask_check_all.png"), cv2.cvtColor(overlay_all, cv2.COLOR_RGB2BGR))
print(f"Terminato. Metodo: {args.mode}. Estratti {count_tronchi} tronchi.")
print(f"Report score salvato in: {csv_path}")