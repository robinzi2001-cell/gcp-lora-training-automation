#!/usr/bin/env python3
"""
GCP VM Provisioning Script for LoRA Training
Erstellt eine VM mit 80GB GPU (A100 oder L4) für KI-Modell-Training
"""

import subprocess
import json
import time
import sys
from typing import Dict, Optional

class GCPVMProvisioner:
    def __init__(self, project_id: str = "lora567"):
        self.project_id = project_id
        self.instance_name = "lora-training-vm"
        self.zone = "us-central1-a"  # Verfügbarkeit für A100
        self.machine_type = "a2-highgpu-1g"  # 1x A100 40GB GPU
        
    def check_prerequisites(self) -> bool:
        """Prüft ob gcloud CLI installiert und authentifiziert ist"""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ gcloud CLI aktiv. Aktuelles Projekt: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ gcloud CLI nicht gefunden oder nicht authentifiziert")
            print("Installiere: https://cloud.google.com/sdk/docs/install")
            print("Authentifiziere: gcloud auth login")
            return False
    
    def set_project(self):
        """Setzt das GCP Projekt"""
        try:
            subprocess.run(
                ["gcloud", "config", "set", "project", self.project_id],
                check=True
            )
            print(f"✓ Projekt {self.project_id} gesetzt")
        except subprocess.CalledProcessError as e:
            print(f"✗ Fehler beim Setzen des Projekts: {e}")
            sys.exit(1)
    
    def check_quotas(self):
        """Prüft GPU-Quotas"""
        print("\n⚠️  WICHTIG: Stelle sicher, dass du GPU-Quotas in deinem GCP-Projekt hast:")
        print("   - NVIDIA_A100_GPUS: mindestens 1")
        print("   - Prüfe: https://console.cloud.google.com/iam-admin/quotas")
        print("   - Beantrage Quota-Erhöhung falls nötig\n")
    
    def create_vm(self) -> bool:
        """Erstellt die VM mit GPU"""
        print(f"\n🚀 Erstelle VM '{self.instance_name}' in Zone {self.zone}...")
        
        # Startup-Script für automatische Installation
        startup_script = """#!/bin/bash
# Automatische Installation von NVIDIA-Treibern und CUDA
apt-get update
apt-get install -y wget software-properties-common

# NVIDIA Driver Installation
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-drivers
apt-get -y install cuda-toolkit-12-4

# Docker Installation
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Python und essenzielle Tools
apt-get install -y python3-pip git
pip3 install --upgrade pip
pip3 install google-cloud-storage torch torchvision torchaudio pillow diffusers transformers accelerate

# Arbeitsverzeichnis erstellen
mkdir -p /home/lora_training/{raw_data,processed_data,models,logs}
chmod -R 777 /home/lora_training

# Status-Datei
echo "VM Ready: $(date)" > /home/lora_training/vm_status.txt
"""
        
        cmd = [
            "gcloud", "compute", "instances", "create", self.instance_name,
            "--project", self.project_id,
            "--zone", self.zone,
            "--machine-type", self.machine_type,
            "--accelerator", "type=nvidia-tesla-a100,count=1",
            "--image-family", "ubuntu-2204-lts",
            "--image-project", "ubuntu-os-cloud",
            "--boot-disk-size", "200GB",
            "--boot-disk-type", "pd-balanced",
            "--maintenance-policy", "TERMINATE",
            "--metadata", f"startup-script={startup_script}",
            "--scopes", "cloud-platform",
            "--tags", "lora-training,http-server,https-server"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ VM '{self.instance_name}' erfolgreich erstellt!")
            print("\n⏳ Warte 5 Minuten auf Startup-Script und Treiber-Installation...")
            time.sleep(300)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Fehler beim Erstellen der VM: {e}")
            print("\n💡 Alternativen falls A100 nicht verfügbar:")
            print("   - Ändere machine_type zu 'n1-standard-8'")
            print("   - Ändere accelerator zu 'type=nvidia-tesla-t4,count=1' (16GB)")
            print("   - Oder 'type=nvidia-l4,count=1' (24GB)")
            return False
    
    def get_vm_info(self) -> Optional[Dict]:
        """Holt VM-Informationen"""
        try:
            result = subprocess.run(
                ["gcloud", "compute", "instances", "describe", 
                 self.instance_name, "--zone", self.zone, 
                 "--format", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError:
            return None
    
    def setup_firewall(self):
        """Erstellt Firewall-Regeln für SSH und Jupyter"""
        print("\n🔒 Richte Firewall-Regeln ein...")
        
        rules = [
            {
                "name": "allow-ssh-lora",
                "allow": "tcp:22",
                "description": "SSH für LoRA Training VM"
            },
            {
                "name": "allow-jupyter-lora",
                "allow": "tcp:8888",
                "description": "Jupyter Notebook für LoRA Training"
            }
        ]
        
        for rule in rules:
            try:
                subprocess.run(
                    ["gcloud", "compute", "firewall-rules", "create", 
                     rule["name"],
                     "--allow", rule["allow"],
                     "--target-tags", "lora-training",
                     "--description", rule["description"]],
                    check=True,
                    stderr=subprocess.DEVNULL
                )
                print(f"✓ Firewall-Regel '{rule['name']}' erstellt")
            except subprocess.CalledProcessError:
                print(f"  Regel '{rule['name']}' existiert bereits")
    
    def get_ssh_command(self) -> str:
        """Generiert SSH-Verbindungsbefehl"""
        return f"gcloud compute ssh {self.instance_name} --zone {self.zone} --project {self.project_id}"
    
    def provision(self):
        """Hauptfunktion für Provisionierung"""
        print("="*60)
        print("🖥️  GCP VM Provisioner für LoRA Training")
        print("="*60)
        
        if not self.check_prerequisites():
            sys.exit(1)
        
        self.set_project()
        self.check_quotas()
        
        # Prüfe ob VM bereits existiert
        vm_info = self.get_vm_info()
        if vm_info:
            print(f"\n⚠️  VM '{self.instance_name}' existiert bereits!")
            print(f"Status: {vm_info.get('status', 'UNKNOWN')}")
            response = input("Neu erstellen? (löscht existierende VM) [y/N]: ")
            if response.lower() == 'y':
                print("🗑️  Lösche existierende VM...")
                subprocess.run(
                    ["gcloud", "compute", "instances", "delete", 
                     self.instance_name, "--zone", self.zone, "--quiet"],
                    check=True
                )
                time.sleep(5)
            else:
                print("Abgebrochen.")
                return
        
        if self.create_vm():
            self.setup_firewall()
            
            print("\n" + "="*60)
            print("✅ VM-Provisionierung abgeschlossen!")
            print("="*60)
            print(f"\n📝 VM-Name: {self.instance_name}")
            print(f"📍 Zone: {self.zone}")
            print(f"🎮 GPU: NVIDIA A100 40GB (oder alternative)")
            print(f"\n🔗 SSH-Verbindung:")
            print(f"   {self.get_ssh_command()}")
            print(f"\n📊 VM-Status prüfen:")
            print(f"   gcloud compute instances describe {self.instance_name} --zone {self.zone}")
            print(f"\n💰 Kosten-Warnung: A100-VMs sind teuer (~$3-4/Stunde)")
            print(f"   Stoppen: gcloud compute instances stop {self.instance_name} --zone {self.zone}")
            print(f"   Löschen: gcloud compute instances delete {self.instance_name} --zone {self.zone}")
            print("\n" + "="*60)


def main():
    provisioner = GCPVMProvisioner(project_id="lora567")
    provisioner.provision()


if __name__ == "__main__":
    main()
