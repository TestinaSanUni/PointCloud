import numpy as np
import cv2
import torch
import os
import glob
import sys
import argparse
import csv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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

if not os.path.exists(data_dir): os.makedirs(data_dir)

# Pulizia file precedenti
for f in glob.glob(os.path.join(data_dir, "tronco_*")):
    try:
        os.remove(f)
    except:
        pass

# --- CARICAMENTO DATI COMUNI ---
print(f"Caricamento dati per modalità: {args.mode}...")
image_np = np.load(os.path.join(data_dir, "color_data.npy"))
depth_array = np.load(os.path.join(data_dir, "depth_data.npy"))
# kernel = np.ones((3, 3), np.uint8)
# depth_array = cv2.dilate(depth_array, kernel, iterations=1)
params = np.load(os.path.join(data_dir, "camera_params.npz"))
intrinsic = params['intrinsic']
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]


# --- MODELLO 1: SAM CLASSIC ---
def run_sam_classic(image):
    print("Inizializzazione SAM Classic (Automatic Mask Generator)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=48,
        pred_iou_thresh=0.92,
        stability_score_thresh=0.92,
        min_mask_region_area=1800
    )
    return mask_generator.generate(image)


# --- MODELLO 2: LANG-SAM ---
def run_lang_sam(image):
    print("Inizializzazione Lang-SAM (Text-to-Segmentation)...")
    from lang_sam import LangSAM
    from PIL import Image

    model = LangSAM()
    image_pil = Image.fromarray(image).convert("RGB")

    text_prompt = "single wood log . single wood stick . single stick . single log"
    results = model.predict([image_pil], [text_prompt], box_threshold=0.25, text_threshold=0.20)

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

# --- LOGICA DI FILTRAGGIO E ESTRAZIONE ---
print(f"Elaborazione maschere e riproiezione 3D...")
h, w, _ = image_np.shape
max_area_consentita = (h * w) * 0.10
min_area_consentita = (h * w) * 0.0010

overlay_all = image_np.copy()
count_tronchi = 0
report_score = []
masks = sorted(masks, key=(lambda x: x['area']))

for i, mask_data in enumerate(masks):
    mask = mask_data['segmentation']
    area = mask_data['area']
    bbox = mask_data['bbox']

    # Recupero score in base al modello usato
    if args.mode == "SAM_CLASSIC":
        score = mask_data.get('predicted_iou', 0.0)
    else:
        score = mask_data.get('score', 0.0)

    # Filtri geometrici (Area e Profondità)
    if area > max_area_consentita or area < min_area_consentita:
        continue

    v, u = np.where(mask > 0)
    z = depth_array[v, u]
    valid = z > 0.001

    # Filtro di validità dei punti 3D
    if np.sum(valid) < 800:
        continue

    z_valid = z[valid]
    z_range = np.max(z_valid) - np.min(z_valid)
    z_std = np.std(z_valid)

    # Filtro di spessore
    if args.mode == "SAM_CLASSIC":
        if z_range < 0.10 or z_std < 0.04:
            continue

    # Filtro Colore (Luminosità)
    pixel_valori = image_np[mask]
    if np.mean(pixel_valori) < 80 and args.mode == "SAM_CLASSIC":
        continue

    count_tronchi += 1

    # Aggiunta dati al report
    report_score.append({'id': count_tronchi, 'score': round(float(score), 4)})

    print(f"DEBUG: tronco {count_tronchi} | area: {area} | score: {score:.4f}")

    # Back-projection XYZ
    x_3d = (u[valid] - cx) * z_valid / fx
    y_3d = (v[valid] - cy) * z_valid / fy
    points_3d = np.stack([x_3d, y_3d, z_valid], axis=1)
    np.savetxt(os.path.join(data_dir, f"tronco_{count_tronchi}.xyz"), points_3d)

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
        # Ho aggiunto lo score anche sulla preview visiva per comodità
        cv2.putText(overlay_all, f"ID:{count_tronchi} (S:{score:.2f})", (int(u.mean()), int(v.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- SCRITTURA FILE CSV ---
csv_path = os.path.join(data_dir, "tronchi_score.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score'])
    writer.writeheader()
    writer.writerows(report_score)

cv2.imwrite(os.path.join(data_dir, "mask_check_all.png"), cv2.cvtColor(overlay_all, cv2.COLOR_RGB2BGR))
print(f"Terminato. Metodo: {args.mode}. Estratti {count_tronchi} tronchi.")
print(f"Report score salvato in: {csv_path}")