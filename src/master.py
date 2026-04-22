# (1)
# gestisce la pipeline
#

import os
import sys
import subprocess
from tkinter import filedialog, Tk

def run_pipeline(las_path, mode):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    scripts = [
        "las_data_extractor.py",
        "sam_segmentation.py",
        "segmented_log_measurement.py"
    ]

    print(f"\nAvvio Pipeline in modalità: [{mode}]")
    print(f"File: {os.path.basename(las_path)}")

    for script in scripts:
        script_path = os.path.join(current_dir, script)

        if not os.path.exists(script_path):
            print(f"ERRORE: Script non trovato: {script}")
            return

        print(f"\n--- ESECUZIONE: {script} ---")

        cmd = [sys.executable, script_path, las_path, "--mode", mode]

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"\nErrore in {script}. Pipeline interrotta.")
            return

    print(f"\nPipeline {mode} completata con successo!")


def get_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_scelto = filedialog.askopenfilename(title="Seleziona file .las", filetypes=[("File LAS", "*.las")])
    root.destroy()
    return os.path.abspath(file_scelto) if file_scelto else None


if __name__ == "__main__":
    current_mode = "SAM_CLASSIC"

    while True:
        print("\n" + "=" * 45)
        print(f"SEGMENTAZIONE TRONCHI")
        print(f"MODELLO ATTIVO: [{current_mode}]")
        print("=" * 45)
        print("1. Esegui Pipeline (Extractor + SAM + Misure)")
        print("2. Cambia Modello (SAM Classic / Lang-SAM)")
        print("3. Visualizza Nuvola 3D (las_viewer)")
        print("0. Esci")
        print("=" * 45)

        scelta = input("Seleziona opzione: ").strip()

        if scelta == "1":
            file_scelto = get_file()
            if file_scelto:
                run_pipeline(file_scelto, current_mode)

        elif scelta == "2":
            current_mode = "LANG_SAM" if current_mode == "SAM_CLASSIC" else "SAM_CLASSIC"
            print(f"\nModalità cambiata in: {current_mode}")

        elif scelta == "3":
            file_scelto = get_file()
            if file_scelto:
                subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "las_viewer.py"), file_scelto])

        elif scelta == "0":
            break