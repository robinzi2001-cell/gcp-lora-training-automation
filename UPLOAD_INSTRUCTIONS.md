# 🚀 Einfache Anleitung: Dateien zu GitHub hochladen

## Problem
Der GitHub MCP-Server hat keine Schreibrechte für dein Repository. Das ist eine Sicherheitsbeschränkung.

## ✅ Lösung: 3 einfache Methoden

---

## Methode 1: GitHub Web-UI (Am Einfachsten) 🌐

### Schritt 1: Container-ID finden
```bash
docker ps
# Kopiere die Container-ID von "mcp/claude-desktop" oder ähnlich
```

### Schritt 2: Dateien aus Container kopieren
```bash
# Ersetze <container-id> mit deiner tatsächlichen Container-ID
CONTAINER_ID=<container-id>

# Erstelle lokales Verzeichnis
mkdir -p ~/gcp-lora-files

# Kopiere alle Dateien
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/1_provision_vm.py ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/2_preprocess_images.py ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/3_train_lora.py ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/mcp_server.py ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/README.md ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/QUICKSTART.md ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/requirements.txt ~/gcp-lora-files/
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/.gitignore ~/gcp-lora-files/
```

### Schritt 3: Auf GitHub hochladen
1. Gehe zu: https://github.com/robinzi2001-cell/gcp-lora-training-automation
2. Klicke "Add file" → "Upload files"
3. Ziehe alle Dateien aus `~/gcp-lora-files/` in das Upload-Feld
4. Commit message: "Add complete GCP LoRA training automation"
5. Klicke "Commit changes"

✅ **Fertig!**

---

## Methode 2: Git CLI (Für Git-User) 💻

### Schritt 1: Container-Dateien kopieren (wie oben)
```bash
docker ps  # Finde Container-ID
CONTAINER_ID=<container-id>
mkdir -p ~/gcp-lora-files
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/. ~/gcp-lora-files/
```

### Schritt 2: Repository klonen und pushen
```bash
cd ~/gcp-lora-files
git init
git add .
git commit -m "Add complete GCP LoRA training automation"

# Füge Remote hinzu
git remote add origin https://github.com/robinzi2001-cell/gcp-lora-training-automation.git
git branch -M main

# Push (benötigt GitHub Personal Access Token)
git push -u origin main
```

**Bei Authentifizierung:**
- Username: `robinzi2001-cell`
- Password: Dein GitHub Personal Access Token (erstelle eins unter: https://github.com/settings/tokens)

---

## Methode 3: GitHub CLI (Wenn installiert) 🔧

```bash
# Container-Dateien kopieren (wie oben)
docker ps
CONTAINER_ID=<container-id>
mkdir -p ~/gcp-lora-files
docker cp $CONTAINER_ID:/tmp/gcp-lora-automation/. ~/gcp-lora-files/

cd ~/gcp-lora-files

# Mit GitHub CLI authentifizieren
gh auth login

# Repository klonen
gh repo clone robinzi2001-cell/gcp-lora-training-automation
cd gcp-lora-training-automation

# Dateien kopieren und pushen
cp -r ~/gcp-lora-files/* .
git add .
git commit -m "Add complete GCP LoRA training automation"
git push
```

---

## 📋 Checkliste: Diese 8 Dateien sollten hochgeladen werden

- [ ] `1_provision_vm.py` (8.9 KB)
- [ ] `2_preprocess_images.py` (12 KB)
- [ ] `3_train_lora.py` (17 KB)
- [ ] `mcp_server.py` (16 KB)
- [ ] `README.md` (7.2 KB)
- [ ] `QUICKSTART.md` (5.1 KB)
- [ ] `requirements.txt` (674 bytes)
- [ ] `.gitignore` (513 bytes)

---

## ❓ Warum funktioniert der direkte Push nicht?

Der GitHub MCP-Server, den ich verwende, hat nur **Leserechte** (read-only). Das ist eine Sicherheitsmaßnahme. Für Schreibrechte benötigt man einen Personal Access Token mit `repo` Scope, der explizit für Schreibzugriff konfiguriert ist.

---

## 🆘 Probleme?

### "docker: command not found"
→ Docker ist nicht installiert oder nicht im PATH

### "No such container"
→ Falsche Container-ID. Prüfe mit `docker ps -a`

### "Permission denied"
→ Auf Linux: Nutze `sudo` vor `docker cp`

### "Authentication failed" bei git push
→ Erstelle Personal Access Token: https://github.com/settings/tokens/new
→ Scope: "repo" (full control)
→ Verwende Token als Passwort

---

## ✅ Nach erfolgreichem Upload

Dein Repository unter:
https://github.com/robinzi2001-cell/gcp-lora-training-automation

Sollte jetzt diese Struktur haben:
```
gcp-lora-training-automation/
├── 1_provision_vm.py
├── 2_preprocess_images.py
├── 3_train_lora.py
├── mcp_server.py
├── README.md
├── QUICKSTART.md
├── requirements.txt
└── .gitignore
```

Dann kannst du starten mit:
```bash
git clone https://github.com/robinzi2001-cell/gcp-lora-training-automation.git
cd gcp-lora-training-automation
pip install -r requirements.txt
python 1_provision_vm.py
```

---

**Viel Erfolg! 🎉**
