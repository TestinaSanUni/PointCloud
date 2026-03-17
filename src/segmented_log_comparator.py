import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import time
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import ListedColormap
#from sklearn.neighbors import NearestNeighbors
import argparse

# --- GESTIONE ARGOMENTI ---
parser = argparse.ArgumentParser()
parser.add_argument("cartella_automatici", nargs='?', help="Percorso cartella con file automatici e tronchi_score.csv")
parser.add_argument("cartella_manuali", nargs='?', help="Percorso cartella con file manuali (ground-truth)")
parser.add_argument("--output_dir", default=None, help="Cartella output per plot (default: data_dir progetto)")
parser.add_argument("--top_percent", type=float, default=0.65, help="Percentuale superiore per identificazione (default: 0.65)")
parser.add_argument("--iou_threshold", type=float, default=0.70, help="Soglia minima IoU per identificazione (default: 0.70)")
args = parser.parse_args()

root = tk.Tk()
root.withdraw()

if not args.cartella_automatici:
    print("Seleziona la cartella con i file AUTOMATICI (contenente tronchi_score.csv)...")
    args.cartella_automatici = filedialog.askdirectory(title="Seleziona cartella AUTOMATICI")
    if not args.cartella_automatici:
        print("Nessuna cartella selezionata.")
        exit(1)

if not args.cartella_manuali:
    print("Seleziona la cartella con i file MANUALI (ground-truth)...")
    args.cartella_manuali = filedialog.askdirectory(title="Seleziona cartella MANUALI")
    if not args.cartella_manuali:
        print("Nessuna cartella selezionata.")
        exit(1)

root.destroy()

# --- CONFIGURAZIONE PERCORSI ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(base_dir), "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"\nCartella automatici: {args.cartella_automatici}")
print(f"Cartella manuali: {args.cartella_manuali}")
print(f"Cartella output: {data_dir}")
print(f"Modalità: {args.top_percent * 100:.0f}% superiore per identificazione")
print(f"Soglia IoU identificazione: {args.iou_threshold * 100:.0f}%")


# --- FUNZIONI DI UTILITÀ ---
def extract_top_points(points, prc=0.65):
    if len(points) == 0:
        return points

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z_range = z_max - z_min
    soglia_z = z_max - (z_range * prc)
    mask = points[:, 2] >= soglia_z
    return points[mask]


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


def find_best_match_direct(automatico, lista_manuali, top_percent=0.65):
    auto_superiore = extract_top_points(automatico, top_percent)
    auto_xy = auto_superiore[:, :2]

    if len(auto_superiore) < 50:
        return None, -1, None, None

    best_iou = -1
    best_idx = -1
    best_masks = (None, None)

    for idx_man, manuale in enumerate(lista_manuali):
        man_superiore = extract_top_points(manuale, top_percent)
        man_xy = man_superiore[:, :2]

        if len(man_superiore) < 50:
            continue

        iou_tmp, m_a, m_m = calculate_iou_metric(auto_xy, man_xy)

        if iou_tmp > best_iou:
            best_iou = iou_tmp
            best_idx = idx_man
            best_masks = (m_a, m_m)

            if best_iou > 0.96:
                return best_idx, best_iou, m_a, m_m

    if best_iou > 0:
        return best_idx, best_iou, best_masks[0], best_masks[1]
    else:
        return None, -1, None, None


def iou_full_calc_direct(automatico, manuale):
    auto_xy = automatico[:, :2]
    man_xy = manuale[:, :2]

    iou, m_a, m_m = calculate_iou_metric(auto_xy, man_xy)
    return iou, m_a, m_m


def plot_validation(tronco_id, score, iou, mask_a, mask_m, output_dir, type, top_percent):
    cmap = ListedColormap(['#f0f0f0', '#e74c3c', '#3498db', '#2ecc71'])
    viz = np.zeros(mask_a.shape)
    viz[mask_m] += 1
    viz[mask_a] += 2

    plt.figure(figsize=(10, 8))
    plt.imshow(viz, origin='lower', cmap=cmap)

    if type == 'top':
        titolo = f"VALIDAZIONE TRONCO {tronco_id} (TOP {top_percent * 100:.0f}%)\nScore: {score:.4f} | IoU: {iou:.2%}"
        output_path = os.path.join(output_dir, f"validazione_tronco_{tronco_id}_iou_top_{iou:.3f}.png")
    else:
        titolo = f"VALIDAZIONE TRONCO {tronco_id} (FULL)\nScore: {score:.4f} | IoU: {iou:.2%}"
        output_path = os.path.join(output_dir, f"validazione_tronco_{tronco_id}_iou_full_{iou:.3f}.png")

    plt.title(titolo)
    plt.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


# --- CARICAMENTO DATI ---
csv_path = os.path.join(args.cartella_automatici, "tronchi_score.csv")
if not os.path.exists(csv_path):
    print(f"ERRORE: File {csv_path} non trovato!")
    exit(1)

automatici_con_score = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        id_tronco = int(row['id'])
        score = float(row['score'])
        file_path = os.path.join(args.cartella_automatici, f"tronco_{id_tronco}.xyz")
        if os.path.exists(file_path):
            automatici_con_score.append({
                'id': id_tronco,
                'score': score,
                'file': file_path
            })

automatici_con_score.sort(key=lambda x: x['score'], reverse=True)
print(f"Trovati {len(automatici_con_score)} tronchi automatici.")

file_manuali = glob.glob(os.path.join(args.cartella_manuali, "*.xyz"))
if not file_manuali:
    print(f"ERRORE: Nessun file .xyz trovato in {args.cartella_manuali}")
    exit(1)

print("Caricamento file manuali in memoria...")
manuali_punti = []
for file_man in file_manuali:
    try:
        punti = np.loadtxt(file_man)
        manuali_punti.append(punti)
    except Exception as e:
        print(f"ERRORE caricamento {file_man}: {e}")
        manuali_punti.append(None)

print(f"Trovati {len(file_manuali)} tronchi manuali (ground-truth).")

# --- ASSOCIAZIONE ---
print("\nInizio associazione automatico-manuale...")
start_time = time.time()

manuali_disponibili = list(range(len(file_manuali)))
associazioni = []

for auto in automatici_con_score:
    print(f"\nElaborazione tronco automatico {auto['id']} (score: {auto['score']:.4f})...")

    if len(manuali_disponibili) == 0:
        print("  -> Nessun manuale rimasto disponibile.")
        break

    try:
        punti_auto_completo = np.loadtxt(auto['file'])
    except Exception as e:
        print(f"  -> ERRORE caricamento file automatico: {e}")
        continue

    manuali_da_testare = [(idx, manuali_punti[idx]) for idx in manuali_disponibili if manuali_punti[idx] is not None]

    if not manuali_da_testare:
        print("  -> Nessun manuale valido disponibile.")
        continue

    # FASE 1: Identificazione con parte superiore
    print(f"  Identificazione diretta in corso su {len(manuali_da_testare)} manuali...")

    best_idx_rel, iou_ident, m_a_top, m_m_top = find_best_match_direct(
        punti_auto_completo,
        [punti for _, punti in manuali_da_testare],
        top_percent=args.top_percent
    )

    if best_idx_rel is None:
        print(f"  -> Nessun match valido trovato.")
        continue

    best_idx_assoluto = manuali_da_testare[best_idx_rel][0]
    nome_manuale = os.path.basename(file_manuali[best_idx_assoluto])
    punti_man_completo = manuali_punti[best_idx_assoluto]

    # FASE 2: Calcolo IoU completo
    iou_full, m_a_full, m_m_full = iou_full_calc_direct(
        punti_auto_completo,
        punti_man_completo
    )

    if iou_ident >= args.iou_threshold:
        manuali_disponibili.remove(best_idx_assoluto)

        # Salvataggio plot top
        plot_top = plot_validation(
            auto['id'],
            auto['score'],
            iou_ident,
            m_a_top, m_m_top,
            data_dir,
            'top',
            args.top_percent
        )

        # Salvataggio plot full
        plot_full = plot_validation(
            auto['id'],
            auto['score'],
            iou_full,
            m_a_full, m_m_full,
            data_dir,
            'full',
            args.top_percent
        )

        associazioni.append({
            'id_auto': auto['id'],
            'score_auto': auto['score'],
            'file_manuale': nome_manuale,
            'iou_ident': iou_ident,
            'iou_full': iou_full
        })

        print(f"  >>> ASSOCIATO: Auto {auto['id']} -> {nome_manuale}")
        print(f"      IoU(ident): {iou_ident:.2%} >= {args.iou_threshold * 100:.0f}%")
        print(f"      IoU(full): {iou_full:.2%}")
    else:
        print(f"  >>> NON ASSOCIATO: IoU identificazione {iou_ident:.2%} < {args.iou_threshold * 100:.0f}%")

# --- REPORT FINALE ---
end_time = time.time()
print(f"\n{'=' * 60}")
print(f"ASSOCIAZIONE COMPLETATA in {end_time - start_time:.2f} secondi")
print(f"{'=' * 60}")
print(f"Tronchi automatici totali: {len(automatici_con_score)}")
print(f"Tronchi associati: {len(associazioni)}")
print(f"Tronchi manuali non associati: {len(manuali_disponibili)}")
print(f"Tronchi automatici non associati: {len(automatici_con_score) - len(associazioni)}")

report_path = os.path.join(data_dir, "associazioni_tronchi.csv")
with open(report_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['id_auto', 'score_auto', 'file_manuale', 'iou_ident', 'iou_full'])
    for ass in associazioni:
        writer.writerow([
            ass['id_auto'],
            f"{ass['score_auto']:.4f}",
            ass['file_manuale'],
            f"{ass['iou_ident']:.4f}",
            f"{ass['iou_full']:.4f}"
        ])

print(f"\nReport salvato in: {report_path}")

if associazioni:
    iou_medi_ident = np.mean([a['iou_ident'] for a in associazioni])
    iou_medi_full = np.mean([a['iou_full'] for a in associazioni])

    print(f"\nSTATISTICHE:")
    print(f"  IoU identificazione medio: {iou_medi_ident:.2%}")
    print(f"  IoU FULL medio: {iou_medi_full:.2%}")
    print(f"  Miglior IoU FULL: {max([a['iou_full'] for a in associazioni]):.2%}")
    print(f"  Peggior IoU FULL: {min([a['iou_full'] for a in associazioni]):.2%}")