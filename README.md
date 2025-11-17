# GCP LoRA Training Automation 🚀

Vollautomatisches SDXL LoRA-Training auf Google Cloud Platform mit GPU-Unterstützung.

## 🎯 Features

- **Automatische VM-Provisionierung** mit 80GB A100 GPU (oder Alternativen)
- **Intelligente Bildvorverarbeitung** mit Auto-Captioning (BLIP) und Tagging
- **Optimiertes SDXL LoRA Training** mit Kohya-ss Scripts
- **GPU-effiziente Nutzung** - nur aktiv bei Bedarf
- **MCP-Server kompatibel** - direkte Steuerung über Claude
- **Individualisierbare Keywords** für personalisierte Modelle

## 📋 Voraussetzungen

### Lokal
- Python 3.8+
- Google Cloud SDK (`gcloud`)
- Google Cloud Projekt mit aktivierter Compute Engine API
- GPU-Quota in GCP (NVIDIA_A100_GPUS oder T4/L4)

### GCP-Setup
```bash
# gcloud installieren
curl https://sdk.cloud.google.com | bash

# Authentifizieren
gcloud auth login

# Projekt erstellen/setzen
gcloud config set project lora567
```

## 🚀 Schnellstart

### 1. Repository klonen
```bash
git clone https://github.com/robinzi2001-cell/gcp-lora-training-automation.git
cd gcp-lora-training-automation
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Workflow

#### Schritt 1: VM bereitstellen
```bash
python 1_provision_vm.py
```
Dies erstellt:
- VM mit A100 GPU (oder Alternative)
- NVIDIA-Treiber + CUDA Toolkit
- Docker + NVIDIA Container Toolkit
- Python-Umgebung mit allen Dependencies

**Kosten-Hinweis:** A100-VMs kosten ca. $3-4/Stunde!

#### Schritt 2: Bilder vorbereiten
```bash
# Lege deine Rohbilder in ./raw_data/
mkdir raw_data
cp /pfad/zu/deinen/bildern/* raw_data/

# Starte Vorverarbeitung
python 2_preprocess_images.py --keyword ciri567
```

Das Skript:
- Fragt dein individuelles Keyword ab (z.B. "ciri567")
- Benennt Bilder um: `keyword_0001.jpg`, `keyword_0002.jpg`, ...
- Generiert automatische Captions mit BLIP
- Analysiert Bildqualität
- Erstellt Metadaten (JSON + TXT)

Output in `./processed_data/`:
- `images/` - Verarbeitete Bilder
- `training_metadata.json` - Vollständige Metadaten
- `captions.txt` - Captions für Training
- `*.txt` - Individual Captions pro Bild

#### Schritt 3: Training starten
```bash
# Verbinde mit VM
gcloud compute ssh lora-training-vm --zone us-central1-a

# Upload verarbeitete Daten
gcloud compute scp --recurse ./processed_data lora-training-vm:/home/lora_training/ --zone us-central1-a

# Auf VM: Training starten
python 3_train_lora.py
```

Das Training-Skript:
- Installiert Kohya-ss Training Scripts
- Lädt SDXL Base Model
- Konfiguriert Training mit optimalen Settings
- Startet Training mit TensorBoard-Logging
- Speichert LoRA-Modelle im SafeTensors-Format

## ⚙️ Konfiguration

### Training-Parameter anpassen

Erstelle `training_config.json`:
```json
{
  "training": {
    "resolution": 1024,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "max_train_steps": 2000
  },
  "lora": {
    "rank": 32,
    "alpha": 32
  }
}
```

Dann:
```bash
python 3_train_lora.py --config training_config.json
```

### GPU-Alternativen

Falls A100 nicht verfügbar, bearbeite `1_provision_vm.py`:

```python
# Für T4 (16GB) - günstiger
self.machine_type = "n1-standard-8"
self.accelerator = "type=nvidia-tesla-t4,count=1"

# Für L4 (24GB) - balanced
self.machine_type = "g2-standard-8"
self.accelerator = "type=nvidia-l4,count=1"
```

## 🔧 MCP-Server Integration

Für direkte Steuerung über Claude:

1. **MCP-Server konfigurieren** (auf deinem lokalen System):
```json
{
  "mcpServers": {
    "gcp-lora": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "GCP_PROJECT": "lora567",
        "GCP_ZONE": "us-central1-a"
      }
    }
  }
}
```

2. **Verfügbare MCP-Tools**:
- `provision_vm` - VM erstellen
- `upload_images` - Bilder hochladen
- `start_training` - Training starten
- `get_training_status` - Status abfragen
- `download_lora` - Fertiges Modell herunterladen
- `stop_vm` - VM stoppen (Kosten sparen!)

## 📊 Monitoring

### TensorBoard (während Training)
```bash
# Auf VM
tensorboard --logdir /home/lora_training/logs --bind_all

# Lokal (Port-Forward)
gcloud compute ssh lora-training-vm --zone us-central1-a -- -L 6006:localhost:6006
# Öffne: http://localhost:6006
```

### VM-Status
```bash
# Status prüfen
gcloud compute instances describe lora-training-vm --zone us-central1-a

# SSH-Zugriff
gcloud compute ssh lora-training-vm --zone us-central1-a
```

## 💰 Kosten-Management

### VM stoppen (wichtig!)
```bash
# Stoppen (behält Daten, stoppt Kosten)
gcloud compute instances stop lora-training-vm --zone us-central1-a

# Wieder starten
gcloud compute instances start lora-training-vm --zone us-central1-a

# Komplett löschen
gcloud compute instances delete lora-training-vm --zone us-central1-a
```

### Kosten-Übersicht
- **A100 (40GB)**: ~$3-4/h
- **L4 (24GB)**: ~$0.70/h
- **T4 (16GB)**: ~$0.35/h
- **Disk (200GB)**: ~$0.04/h

**Tipp:** Nutze Preemptible VMs für 60-90% Ersparnis:
```bash
--preemptible --maintenance-policy=TERMINATE
```

## 🎨 Output nutzen

Nach Training findest du im `/home/lora_training/output/` Verzeichnis:
- `keyword_lora.safetensors` - Finales LoRA-Modell
- `keyword_lora-000XXX.safetensors` - Zwischenschritte

### Download
```bash
# Einzelne Datei
gcloud compute scp lora-training-vm:/home/lora_training/output/ciri567_lora.safetensors ./ --zone us-central1-a

# Ganzes Verzeichnis
gcloud compute scp --recurse lora-training-vm:/home/lora_training/output ./ --zone us-central1-a
```

### Verwendung in ComfyUI/Automatic1111
1. Kopiere `.safetensors` nach `models/lora/`
2. Im Prompt: `<lora:ciri567_lora:0.8>` (0.8 = 80% Stärke)
3. Verwende dein Keyword: `ciri567, detailed portrait, ...`

## 🐛 Troubleshooting

### GPU nicht erkannt
```bash
# Auf VM prüfen
nvidia-smi

# Falls Fehler, Treiber neu installieren
sudo apt-get install --reinstall cuda-drivers
```

### Quota-Fehler
- Beantrage GPU-Quota: https://console.cloud.google.com/iam-admin/quotas
- Wechsle zu anderer Region (z.B. us-west1-b)
- Nutze kleinere GPU (T4 statt A100)

### Out-of-Memory
- Reduziere `batch_size` auf 1
- Aktiviere `gradient_checkpointing`
- Nutze `use_8bit_adam: true`
- Reduziere `resolution` auf 768 oder 512

### Training-Fehler
```bash
# Logs prüfen
tail -f /home/lora_training/logs/tensorboard.log

# Kohya neu installieren
rm -rf ~/kohya_ss
python 3_train_lora.py
```

## 📚 Weitere Ressourcen

- [Kohya-ss Documentation](https://github.com/kohya-ss/sd-scripts)
- [SDXL Fine-tuning Guide](https://huggingface.co/docs/diffusers/training/lora)
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [LoRA Training Tips](https://rentry.org/lora_train)

## 🤝 Contributing

Pull Requests willkommen! Besonders für:
- Weitere MCP-Server Tools
- Alternative Training-Backends
- Cost-Optimization Features
- Multi-GPU Support

## 📄 Lizenz

MIT License - siehe LICENSE Datei

## ⚠️ Wichtige Hinweise

1. **Kosten:** GPU-VMs sind teuer! Immer nach Training stoppen/löschen
2. **Quotas:** GPU-Quotas vorher beantragen (kann 24h dauern)
3. **Backup:** Wichtige Modelle regelmäßig herunterladen
4. **Bilder:** Verwende nur Bilder mit entsprechenden Rechten
5. **Qualität:** Mindestens 20-30 diverse, hochwertige Bilder für gute Results

---

**Erstellt für effizientes SDXL LoRA-Training auf GCP** 🎯

Bei Fragen: Issues auf GitHub erstellen
