import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import ListedColormap

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("XYZ files", "*.xyz")])
    root.destroy()
    return path

print("Seleziona il file AUTOMATICO...")
f_auto = select_file("Seleziona AUTOMATICO")
print("Seleziona il file MANUALE...")
f_man = select_file("Seleziona MANUALE")

if not f_auto or not f_man:
    print("Selezione annullata.")
    exit()

# Caricamento e centratura iniziale
data_a = np.loadtxt(f_auto)[:, :2]
data_m = np.loadtxt(f_man)[:, :2]
data_a -= np.mean(data_a, axis=0)
data_m -= np.mean(data_m, axis=0)

# Parametri iniziali
off_x, off_y = 0.0, 0.0
mirror = 1
res = 0.01
cmap = ListedColormap(['#f0f0f0', '#e74c3c', '#3498db', '#2ecc71'])

print("\n--- COMANDI TASTIERA ---")
print("Frecce: Sposta il tronco in piccoli step")
print("WASD: Sposta il tronco in grandi step")
print("M: Specchia")
print("Invio: Chiudi")
print("------------------------")

def render():
    global off_x, off_y, mirror
    temp_a = data_a.copy()
    temp_a[:, 0] *= mirror
    temp_a[:, 0] += off_x
    temp_a[:, 1] += off_y

    all_pts = np.vstack([temp_a, data_m])
    x_min, y_min = all_pts.min(axis=0) - 0.05
    x_max, y_max = all_pts.max(axis=0) + 0.05
    w, h = int((x_max - x_min) / res) + 1, int((y_max - y_min) / res) + 1

    m_a, m_m = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
    for p, m in [(temp_a, m_a), (data_m, m_m)]:
        c = np.clip(((p[:, 0] - x_min) / res).astype(int), 0, w - 1)
        r = np.clip(((p[:, 1] - y_min) / res).astype(int), 0, h - 1)
        m[r, c] = True

    iou = np.logical_and(m_a, m_m).sum() / np.logical_or(m_a, m_m).sum()
    viz = np.zeros((h, w))
    viz[m_m] += 1
    viz[m_a] += 2

    ax.clear()
    ax.imshow(viz, origin='lower', cmap=cmap)
    ax.set_title(f"IoU: {iou:.2%}\nFrecce per muovere, 'M' per specchiare, 'Enter' per uscire")
    ax.axis('off')
    plt.draw()

fig, ax = plt.subplots(figsize=(10, 8))

# Gestore eventi tastiera
def on_key(event):
    global off_x, off_y, mirror
    step = 0.005
    if event.key == 'up':
        off_y += step
    elif event.key == 'down':
        off_y -= step
    elif event.key == 'left':
        off_x -= step
    elif event.key == 'right':
        off_x += step
    elif event.key == 'w':
        off_y += step * 100
    elif event.key == 'x':
        off_y -= step * 100
    elif event.key == 'a':
        off_x -= step * 100
    elif event.key == 'd':
        off_x += step * 100
    elif event.key == 'm':
        mirror *= -1
    elif event.key == 'enter':
        plt.close()
    render()

fig.canvas.mpl_connect('key_press_event', on_key)
render()
plt.show()