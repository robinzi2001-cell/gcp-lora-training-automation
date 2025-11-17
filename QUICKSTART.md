# GCP LoRA Training - Quick Start Guide

## 📦 Installation

### 1. GitHub-Repo klonen ODER ZIP herunterladen
```bash
# Option A: Git Clone
git clone https://github.com/robinzi2001-cell/gcp-lora-training-automation.git
cd gcp-lora-training-automation

# Option B: Manuell hochladen
# Die 7 Dateien sind bereits in /tmp/gcp-lora-automation/ erstellt
# Du kannst sie manuell ins GitHub-Repo hochladen über die Web-UI
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Google Cloud Setup
```bash
# gcloud installieren (falls nicht vorhanden)
curl https://sdk.cloud.google.com | bash

# Authentifizieren
gcloud auth login

# Projekt setzen
gcloud config set project lora567
```

## 🚀 Workflow

### Schritt 1: VM erstellen (mit 80GB GPU)
```bash
python 1_provision_vm.py
```
**Wichtig:** Benötigt GPU-Quotas in GCP!
**Kosten:** ~$3-4/Stunde für A100

### Schritt 2: Bilder vorbereiten
```bash
# Lege deine Rohbilder in ./raw_data/
mkdir raw_data
cp /pfad/zu/deinen/bildern/*.jpg raw_data/

# Starte Vorverarbeitung
python 2_preprocess_images.py --keyword ciri567
```

Das Skript:
- Fragt dein Keyword ab (z.B. "ciri567")
- Benennt Bilder um: keyword_0001.jpg, keyword_0002.jpg...
- Generiert automatische Captions mit BLIP
- Analysiert Bildqualität
- Erstellt Metadaten

### Schritt 3: Training starten
```bash
# Verbinde mit VM
gcloud compute ssh lora-training-vm --zone us-central1-a

# Upload Daten zur VM
gcloud compute scp --recurse ./processed_data lora-training-vm:/home/lora_training/ --zone us-central1-a

# Auf VM: Training starten
python 3_train_lora.py
```

Training läuft automatisch:
- Installiert Kohya-ss Scripts
- Lädt SDXL Base Model (~6GB, einmalig)
- Trainiert LoRA mit optimalen Settings
- Speichert Modelle als .safetensors

## 🎯 MCP-Server (für direkte Claude-Steuerung)

### Setup
1. Füge zu deiner MCP-Config hinzu (`~/.config/claude/mcp.json`):
```json
{
  "mcpServers": {
    "gcp-lora": {
      "command": "python",
      "args": ["/pfad/zu/mcp_server.py"],
      "env": {
        "GCP_PROJECT": "lora567",
        "GCP_ZONE": "us-central1-a"
      }
    }
  }
}
```

2. Verfügbare Tools in Claude:
- `provision_vm` - VM erstellen
- `upload_images` - Bilder hochladen
- `preprocess_images` - Bilder vorbereiten
- `start_training` - Training starten
- `get_training_status` - Status abfragen
- `download_lora` - Modell herunterladen
- `stop_vm` - VM stoppen (Kosten sparen!)

## 💰 Kosten-Management

**Sehr wichtig:** VM nach Training STOPPEN!

```bash
# VM stoppen (behält Daten)
gcloud compute instances stop lora-training-vm --zone us-central1-a

# VM wieder starten
gcloud compute instances start lora-training-vm --zone us-central1-a

# VM komplett löschen
gcloud compute instances delete lora-training-vm --zone us-central1-a
```

### Kosten-Übersicht:
- **A100 (40GB)**: ~$3-4/h
- **L4 (24GB)**: ~$0.70/h  
- **T4 (16GB)**: ~$0.35/h

## 🎨 Output nutzen

Nach Training findest du LoRA-Modelle in `/home/lora_training/output/`

### Download:
```bash
gcloud compute scp lora-training-vm:/home/lora_training/output/*.safetensors ./ --zone us-central1-a
```

### Verwendung in ComfyUI/A1111:
1. Kopiere `.safetensors` nach `models/lora/`
2. Im Prompt: `<lora:dein_keyword_lora:0.8>`
3. Nutze dein Keyword: `ciri567, detailed portrait, ...`

## 🐛 Troubleshooting

### GPU-Quota-Fehler
→ Beantrage Quota: https://console.cloud.google.com/iam-admin/quotas
→ Oder nutze T4 statt A100 (günstiger, verfügbar)

### Out-of-Memory
→ Reduziere `batch_size` auf 1 in `3_train_lora.py`
→ Aktiviere `gradient_checkpointing`

### NVIDIA-Treiber fehlen
```bash
# Auf VM:
sudo apt-get install --reinstall cuda-drivers
nvidia-smi  # Sollte GPU zeigen
```

## 📁 Datei-Struktur

```
gcp-lora-automation/
├── 1_provision_vm.py       # VM mit GPU erstellen
├── 2_preprocess_images.py  # Bilder vorbereiten + Captions
├── 3_train_lora.py          # SDXL LoRA Training
├── mcp_server.py            # MCP-Integration für Claude
├── requirements.txt         # Python Dependencies
├── README.md                # Vollständige Doku
└── .gitignore
```

## ⚠️ Wichtige Hinweise

1. **Kosten:** GPU-VMs sind TEUER! Immer nach Training stoppen.
2. **Quotas:** GPU-Quotas vorher beantragen (kann 24h dauern).
3. **Bilder:** Mindestens 20-30 hochwertige, diverse Bilder für gute Results.
4. **Rechte:** Nur Bilder verwenden, für die du Rechte hast.
5. **Backup:** Wichtige Modelle regelmäßig herunterladen.

## 🎓 Was die Skripte tun

### 1_provision_vm.py
✓ Erstellt VM mit A100 GPU
✓ Installiert NVIDIA-Treiber + CUDA
✓ Installiert Docker + NVIDIA Container Toolkit
✓ Installiert Python + ML-Libraries
✓ Richtet Firewall-Regeln ein

### 2_preprocess_images.py  
✓ Fragt Custom Keyword ab
✓ Benennt Bilder einheitlich um
✓ Generiert automatische Captions (BLIP)
✓ Analysiert Bildqualität
✓ Erstellt Metadaten + Caption-Dateien

### 3_train_lora.py
✓ Installiert Kohya-ss Training Scripts
✓ Lädt SDXL Base Model
✓ Konfiguriert optimales Training
✓ Trainiert LoRA-Modell
✓ Speichert Checkpoints + finales Modell

---

**Viel Erfolg beim Training! 🎯**

Bei Fragen: GitHub Issues erstellen
