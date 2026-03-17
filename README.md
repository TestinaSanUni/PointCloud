*Progetto sviluppato come parte della tesi triennale di ricerca in Ingegneria Informatica presso l'Università di Firenze, con titolo "Segmentazione di nuvole di punti attraverso proiezione di segmentazione di immagini".*

-- Tommaso Ticci

---

# SAM & Lang-SAM LAS Segmentation Pipeline

Pipeline per la segmentazione automatica, visualizzazione e misure automatiche di tronchi d'albero da nuvole di punti las utilizzando modelli **Segment Anything Model (SAM)** e **Vision-Language (Lang-SAM)**.

## Architettura del Software

Il progetto è composto dai seguenti moduli funzionali:

**File principali della pipeline**
* **`master.py`**: Punto di ingresso principale per l'esecuzione della pipeline.
* **`las_data_extractor.py`**: Generatore di vista 2D dalle nuvole di punti LAS per l'input a SAM.
* **`sam_segmentation.py`**: Script che esegue segmentazione assistita da SAM e Lang-SAM e riproiezione delle maschere segmentate nello spazio 3D.
* **`segmented_log_measurement.py`**: Estrazione delle misure dai tronchi segmentati.

**File secondari di controllo manuale**
* **`manual_validation_tool.py`**: Interfaccia interattiva per effettuare misure manuali su file XYZ.
* **`segmented_log_comparator.py`**: Algoritmo di calcolo IoU (Intersection over Union) per la validazione delle maschere tramite identificazione di corrispondenze tra trochi manuali e segmentati.
* **`las_viewer.py`**: Utility per la visualizzazione 3D delle nuvole di punti LAS.

## Modelli Pre-addestrati
**! IMPORTANTE !**

Per il corretto funzionamento della segmentazione, è necessario scaricare i pesi del checkpoint **ViT-H** di SAM:

*  **File**: `sam_vit_h_4b8939.pth` (2.56 GB)
* **Download**: [Official SAM Checkpoint (Meta AI)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* **Collocazione**: Posizionare il file nella directory `models/`.

## Requisiti

Le dipendenze necessarie sono elencate nel file `requirements.txt`. Per l'installazione:
`pip install -r requirements.txt`

**Specifiche versioni:**
* `laspy~=2.6.1`
* `open3d~=0.19.0`
* `numpy~=2.2.6`
* `opencv-python~=4.12.0.88`
* `torch>=2.4.1`
* `torchvision>=0.19.1`
* `segment-anything~=1.0`
* `matplotlib~=3.10.8`

