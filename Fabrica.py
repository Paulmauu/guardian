# ==============================================================================
# 🏭 GUARDIAN AI FACTORY
# ==============================================================================

# [PASO 0] INSTALACIÓN
!pip install torchmetrics onnx onnxruntime onnxscript pandas scikit-learn --quiet

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from google.colab import drive
from pathlib import Path

# [CONFIGURACIÓN GLOBAL]
print("🚀 INICIANDO FÁBRICA DE MODELOS v2.3...")
drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/guardian_data_v21"
ARTIFACTS_DIR = "/content/drive/MyDrive/guardian_artifacts_onnx"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

MAX_LEN = 1024
BATCH_SIZE = 64
EPOCHS = 5  # 5 épocas para asegurar un buen aprendizaje
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"   ✅ Hardware: {DEVICE}")
print(f"   📦 Artefactos irán a: {ARTIFACTS_DIR}")

# ==============================================================================
# [CAPA 1] DESCARGA DE DATOS (ENLACES ACTUALIZADOS)
# ==============================================================================
def download_data():
    print("\n⬇️ [CAPA 1] ADQUIRIENDO DATOS CRUDOS...")

    # === WAF (CSIC 2010 - enlaces estables) ===
    waf_files = {
        "waf_legit.txt": "https://raw.githubusercontent.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/master/normalTrafficTraining.txt",
        "waf_malic.txt": "https://raw.githubusercontent.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/master/anomalousTrafficTest.txt"
    }
    for name, url in waf_files.items():
        path = f"{BASE_DIR}/{name}"
        if not os.path.exists(path) or os.path.getsize(path) < 10000:
            print(f"      ⬇️ Descargando {name}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"      ✅ {name} OK")
            except Exception as e:
                print(f"      ⚠️ Error descargando {name}: {e}")

    # === DLP (Enron) ===
    if not os.path.exists(f"{BASE_DIR}/dlp_enron.zip") or os.path.getsize(f"{BASE_DIR}/dlp_enron.zip") < 10000:
        print("      ⬇️ Descargando Enron spam...")
        try:
            urllib.request.urlretrieve("https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip", f"{BASE_DIR}/dlp_enron.zip")
            print("      ✅ DLP dataset OK")
        except Exception as e:
            print(f"      ⚠️ Error descargando DLP: {e}")

    # === IDS / HONEYPOT ===
    if not os.path.exists(f"{BASE_DIR}/ids_traffic.csv") or os.path.getsize(f"{BASE_DIR}/ids_traffic.csv") < 10000:
        print("      ⬇️ Descargando NSL-KDD...")
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", f"{BASE_DIR}/ids_traffic.csv")
            print("      ✅ IDS dataset OK")
        except Exception as e:
            print(f"      ⚠️ Error descargando IDS: {e}")

    # === LOGS ===
    if not os.path.exists(f"{BASE_DIR}/sys_logs.log") or os.path.getsize(f"{BASE_DIR}/sys_logs.log") < 1000:
        print("      ⬇️ Descargando logs BGL...")
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log", f"{BASE_DIR}/sys_logs.log")
            print("      ✅ LOGS dataset OK")
        except Exception as e:
            print(f"      ⚠️ Error descargando LOGS: {e}")

    # === AV (DATASET RAW DE CARACTERÍSTICAS PE) ===
    av_path = f"{BASE_DIR}/av_malware.csv"
    if not os.path.exists(av_path) or os.path.getsize(av_path) < 10000:
        print("      ⬇️ Descargando dataset AV...")
        av_url = "https://raw.githubusercontent.com/mburakergenc/Malware-Detection-using-Machine-Learning/refs/heads/master/data.csv"
        try:
            urllib.request.urlretrieve(av_url, av_path)
            print("      ✅ AV dataset OK")
        except Exception as e:
            print(f"      ⚠️ Error descargando AV: {e}")

download_data()

# ==============================================================================
# [CAPA 3.2] DATASET Y LECTORES ESPECÍFICOS
# ==============================================================================
class GuardianDataset(Dataset):
    def __init__(self, context, limit=5000):
        self.data = []
        benign, attack = [], []

        print(f"   ⚙️ Procesando contexto '{context.upper()}'...")

        if context == "waf":
            self._load_waf_txt(f"{BASE_DIR}/waf_legit.txt", benign, limit)
            self._load_waf_txt(f"{BASE_DIR}/waf_malic.txt", attack, limit)
        elif context == "dlp":
            self._load_enron(f"{BASE_DIR}/dlp_enron.zip", benign, attack, limit)
        elif context == "ids" or context == "honeypot":
            self._load_ids(f"{BASE_DIR}/ids_traffic.csv", benign, attack, limit)
        elif context == "log":
            self._load_logs(f"{BASE_DIR}/sys_logs.log", benign, attack, limit)
        elif context == "av":
            self._load_av(f"{BASE_DIR}/av_malware.csv", benign, attack, limit)

        min_len = min(len(benign), len(attack), limit)
        if min_len == 0:
            print(f"      ⚠️ ADVERTENCIA: No se procesaron muestras para {context.upper()}")
            return

        # Balanceo estricto
        for i in range(min_len):
            self.data.append((torch.tensor(benign[i]).long(), torch.tensor(0).long()))
            self.data.append((torch.tensor(attack[i]).long(), torch.tensor(1).long()))

        print(f"      ✅ Muestras listas: {len(self.data)}")

    def _to_bytes(self, input_data):
        seq = np.zeros(MAX_LEN, dtype=np.uint8)
        if isinstance(input_data, str):
            b = list(input_data.encode('utf-8')[:MAX_LEN])
            seq[:len(b)] = b
        else:
            vals = (input_data % 255).astype(np.uint8)
            l = min(len(vals), MAX_LEN)
            seq[:l] = vals[:l]
        return seq

    def _load_waf_txt(self, path, buffer, limit):
        if os.path.exists(path):
            try:
                with open(path, 'r', errors='ignore') as f:
                    for line in f:
                        if len(buffer) >= limit: break
                        buffer.append(self._to_bytes(line))
            except: pass

    def _load_enron(self, path, ben, att, limit):
        try:
            with zipfile.ZipFile(path) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, encoding='latin-1', nrows=limit*4)
                    for _, row in df.iterrows():
                        seq = self._to_bytes(str(row.get('Message', '')))
                        if row.get('Spam/Ham') == 'ham': ben.append(seq)
                        else: att.append(seq)
        except: pass

    # --- LECTOR AV CORREGIDO (USANDO COMAS POR DEFECTO) ---
    def _load_av(self, path, ben, att, limit):
        if os.path.exists(path):
            try:
                # Usamos el separador por defecto (comas) en lugar de sep='|'
                df = pd.read_csv(path, low_memory=False)

                # Columnas que estorban (texto y la propia etiqueta)
                cols_to_drop = ['Name', 'md5', 'legitimate']
                features_cols = [c for c in df.columns if c not in cols_to_drop]

                count = 0
                for _, row in df.iterrows():
                    if count >= limit * 4: break

                    # Extraer características, forzar a numérico y llenar vacíos con 0
                    features = pd.to_numeric(row[features_cols], errors='coerce').fillna(0).values
                    seq = self._to_bytes(features)

                    # 1 = Benigno, 0 = Malware
                    if int(row['legitimate']) == 1:
                        ben.append(seq)
                    else:
                        att.append(seq)
                    count += 1
            except Exception as e:
                print(f"      ⚠️ Error procesando CSV de AV: {e}")

    def _load_ids(self, path, ben, att, limit):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, header=None, nrows=limit*4)
                for _, row in df.iterrows():
                    seq = self._to_bytes(pd.to_numeric(row.iloc[4:-2], errors='coerce').fillna(0).values)
                    if "normal" in str(row.iloc[-2]).lower(): ben.append(seq)
                    else: att.append(seq)
            except: pass

    def _load_logs(self, path, ben, att, limit):
        if os.path.exists(path):
            try:
                with open(path, 'r', errors='ignore') as f:
                    for line in f:
                        if len(ben) >= limit and len(att) >= limit: break
                        seq = self._to_bytes(line)
                        if line.startswith('-'): ben.append(seq)
                        else: att.append(seq)
            except: pass

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==============================================================================
# [CAPA 4] NEUROBRAIN (GRU Bidireccional)
# ==============================================================================
class NeuroBrain(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(256, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.head = nn.Linear(d_model * 2, 2)

    def forward(self, x):
        e = self.embed(x)
        o, _ = self.gru(e)
        context_vector = o.mean(dim=1)
        return self.head(context_vector)

class OnnxWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits = self.m(x)
        return F.softmax(logits, dim=1)

# ==============================================================================
# [CAPA 5] ENTRENAMIENTO + EXPORT
# ==============================================================================
print("\n🔥 [CAPA 4] ENTRENANDO MOTORES H-NET...")

CONTEXTS = ["waf", "av", "ids", "log", "dlp", "honeypot"]

for ctx in CONTEXTS:
    print(f"\n🧠 Entrenando: brain_{ctx}.onnx")
    ds = GuardianDataset(ctx)
    if len(ds) == 0:
        print(f"⚠️ Saltando {ctx} (Faltan datos).")
        continue

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = NeuroBrain().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=0.002)

    model.train()
    for ep in range(EPOCHS):
        loss_avg = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optim.step()
            loss_avg += loss.item()
        print(f"   Ep {ep+1}: Loss {loss_avg/len(loader):.4f}")

    # EXPORTACIÓN ONNX v18
    save_path = f"{ARTIFACTS_DIR}/brain_{ctx}.onnx"
    model.to("cpu").eval()
    dummy_input = torch.randint(0, 255, (1, MAX_LEN), dtype=torch.long)

    torch.onnx.export(
        OnnxWrapper(model),
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["probability"],
        dynamic_axes={'input': {0: 'batch_size'}, 'probability': {0: 'batch_size'}},
        opset_version=18,
        do_constant_folding=True
    )
    print(f"   ✅ ARTEFACTO GENERADO: {save_path}")

    # Limpiar memoria de la tarjeta gráfica
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n🏁 [LISTO] Todos los 6 cerebros están en: {ARTIFACTS_DIR}")
print("👉 Operación de fábrica concluida. ¡A integrarlos en Rust!")
