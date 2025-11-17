#!/usr/bin/env python3
"""
SDXL LoRA Training Pipeline
- Vollautomatisches Setup auf GCP VM
- GPU-effiziente Nutzung (nur bei Bedarf aktiv)
- Kohya-ss/sd-scripts Integration
- Individualisierbare Training-Parameter
- MCP-Server kompatibel
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class LoRATrainingPipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = self.load_or_create_config()
        
        self.work_dir = Path("/home/lora_training")
        self.raw_data_dir = self.work_dir / "raw_data"
        self.processed_data_dir = self.work_dir / "processed_data"
        self.models_dir = self.work_dir / "models"
        self.logs_dir = self.work_dir / "logs"
        self.output_dir = self.work_dir / "output"
        
        # Kohya-ss Installation
        self.kohya_dir = Path.home() / "kohya_ss"
        
    def load_or_create_config(self) -> Dict:
        """Lädt oder erstellt Training-Konfiguration"""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default Konfiguration
        return {
            "model": {
                "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae": "stabilityai/sdxl-vae",
                "precision": "fp16"
            },
            "training": {
                "resolution": 1024,
                "batch_size": 1,
                "learning_rate": 1e-4,
                "max_train_steps": 1000,
                "save_every_n_steps": 100,
                "gradient_accumulation_steps": 4,
                "mixed_precision": "fp16",
                "gradient_checkpointing": True,
                "use_8bit_adam": True,
                "xformers": True
            },
            "lora": {
                "rank": 32,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]
            },
            "optimization": {
                "optimizer": "AdamW8bit",
                "lr_scheduler": "cosine_with_restarts",
                "lr_warmup_steps": 100,
                "noise_offset": 0.05
            },
            "advanced": {
                "caption_dropout_rate": 0.05,
                "shuffle_caption": True,
                "keep_tokens": 1,
                "max_token_length": 225,
                "clip_skip": 2,
                "enable_bucket": True,
                "bucket_no_upscale": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 2048
            }
        }
    
    def save_config(self, path: Optional[Path] = None):
        """Speichert aktuelle Konfiguration"""
        save_path = path or self.work_dir / "training_config.json"
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Konfiguration gespeichert: {save_path}")
    
    def check_gpu(self) -> bool:
        """Prüft GPU-Verfügbarkeit"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_info = result.stdout.strip()
            print(f"✓ GPU erkannt: {gpu_info}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Keine GPU gefunden oder NVIDIA-Treiber fehlen")
            return False
    
    def setup_environment(self):
        """Richtet Python-Umgebung und Dependencies ein"""
        print("\n🔧 Richte Trainingsumgebung ein...")
        
        # Erstelle Verzeichnisse
        for directory in [self.raw_data_dir, self.processed_data_dir, 
                          self.models_dir, self.logs_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("  ✓ Verzeichnisstruktur erstellt")
        
        # Installiere/Update Dependencies
        packages = [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "xformers",
            "bitsandbytes",
            "safetensors",
            "huggingface-hub",
            "tensorboard",
            "wandb",
            "omegaconf",
            "einops",
            "pytorch-lightning",
            "prodigyopt"
        ]
        
        print("\n  📦 Installiere Python-Pakete...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade"] + packages,
                check=True,
                capture_output=True
            )
            print("  ✓ Pakete installiert")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Einige Pakete konnten nicht installiert werden: {e}")
    
    def install_kohya_ss(self):
        """Installiert Kohya-ss Training Scripts"""
        print("\n📥 Installiere Kohya-ss Training Scripts...")
        
        if self.kohya_dir.exists():
            print("  ℹ️  Kohya-ss bereits installiert, aktualisiere...")
            try:
                subprocess.run(
                    ["git", "-C", str(self.kohya_dir), "pull"],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                print("  ⚠️  Git-Pull fehlgeschlagen, verwende existierende Installation")
        else:
            print("  → Clone Repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/kohya-ss/sd-scripts.git", 
                 str(self.kohya_dir)],
                check=True
            )
        
        # Installiere Kohya Requirements
        requirements_file = self.kohya_dir / "requirements.txt"
        if requirements_file.exists():
            print("  → Installiere Kohya Requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True
            )
        
        print("  ✓ Kohya-ss installiert")
    
    def prepare_dataset(self):
        """Bereitet Dataset für Training vor"""
        print("\n📊 Bereite Dataset vor...")
        
        # Prüfe auf processed_data
        image_dir = self.processed_data_dir / "images"
        if not image_dir.exists() or not list(image_dir.glob("*.jpg")):
            print("  ⚠️  Keine verarbeiteten Bilder gefunden!")
            print("  → Führe zuerst 2_preprocess_images.py aus")
            return False
        
        num_images = len(list(image_dir.glob("*.jpg")))
        print(f"  ✓ {num_images} Trainingsbilder gefunden")
        
        # Prüfe Captions
        captions_file = self.processed_data_dir / "captions.txt"
        if not captions_file.exists():
            print("  ⚠️  Keine Captions gefunden!")
            return False
        
        print("  ✓ Captions vorhanden")
        
        # Lade Metadaten
        metadata_file = self.processed_data_dir / "training_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                keyword = metadata.get('keyword', 'unknown')
                print(f"  ✓ Keyword: '{keyword}'")
        
        return True
    
    def download_base_model(self):
        """Lädt SDXL Base Model herunter (falls nicht vorhanden)"""
        print("\n💾 Prüfe SDXL Base Model...")
        
        model_id = self.config["model"]["base_model"]
        cache_dir = self.models_dir / "base_model"
        
        try:
            from huggingface_hub import snapshot_download
            
            print(f"  → Lade {model_id}...")
            print("  ℹ️  Erstmaliger Download kann 10-15 Min. dauern (6GB+)")
            
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                ignore_patterns=["*.safetensors"] if self.config["model"]["precision"] == "fp16" else []
            )
            
            print("  ✓ Base Model bereit")
            return str(cache_dir)
            
        except Exception as e:
            print(f"  ✗ Fehler beim Model-Download: {e}")
            return None
    
    def create_training_config(self) -> Path:
        """Erstellt Kohya-kompatible Training-Config (TOML)"""
        print("\n⚙️  Erstelle Training-Konfiguration...")
        
        config_file = self.work_dir / "training_config.toml"
        
        # Lade Metadaten für Keyword
        metadata_file = self.processed_data_dir / "training_metadata.json"
        keyword = "custom"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                keyword = metadata.get('keyword', 'custom')
        
        # TOML-Konfiguration
        config_content = f"""
# SDXL LoRA Training Config - Generiert am {datetime.now().isoformat()}

[model]
pretrained_model_name_or_path = "{self.config['model']['base_model']}"
vae = "{self.config['model']['vae']}"
v2 = false
v_parameterization = false

[dataset]
resolution = {self.config['training']['resolution']}
batch_size = {self.config['training']['batch_size']}
enable_bucket = {str(self.config['advanced']['enable_bucket']).lower()}
bucket_no_upscale = {str(self.config['advanced']['bucket_no_upscale']).lower()}
min_bucket_reso = {self.config['advanced']['min_bucket_reso']}
max_bucket_reso = {self.config['advanced']['max_bucket_reso']}

[[dataset.subsets]]
image_dir = "{self.processed_data_dir / 'images'}"
num_repeats = 10
caption_extension = ".txt"
shuffle_caption = {str(self.config['advanced']['shuffle_caption']).lower()}
keep_tokens = {self.config['advanced']['keep_tokens']}

[training]
max_train_steps = {self.config['training']['max_train_steps']}
learning_rate = {self.config['training']['learning_rate']}
lr_scheduler = "{self.config['optimization']['lr_scheduler']}"
lr_warmup_steps = {self.config['optimization']['lr_warmup_steps']}
optimizer_type = "{self.config['optimization']['optimizer']}"
gradient_accumulation_steps = {self.config['training']['gradient_accumulation_steps']}
mixed_precision = "{self.config['training']['mixed_precision']}"
gradient_checkpointing = {str(self.config['training']['gradient_checkpointing']).lower()}
xformers = {str(self.config['training']['xformers']).lower()}

[lora]
network_module = "networks.lora"
network_dim = {self.config['lora']['rank']}
network_alpha = {self.config['lora']['alpha']}
network_dropout = {self.config['lora']['dropout']}

[output]
output_dir = "{self.output_dir}"
output_name = "{keyword}_lora"
save_model_as = "safetensors"
save_every_n_steps = {self.config['training']['save_every_n_steps']}

[advanced]
noise_offset = {self.config['optimization']['noise_offset']}
caption_dropout_rate = {self.config['advanced']['caption_dropout_rate']}
clip_skip = {self.config['advanced']['clip_skip']}
max_token_length = {self.config['advanced']['max_token_length']}

[logging]
logging_dir = "{self.logs_dir}"
log_with = "tensorboard"
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"  ✓ Config erstellt: {config_file}")
        return config_file
    
    def start_training(self, config_file: Path):
        """Startet das LoRA Training"""
        print("\n" + "="*60)
        print("🚀 Starte LoRA Training...")
        print("="*60)
        
        # Training Command
        train_script = self.kohya_dir / "sdxl_train_network.py"
        
        if not train_script.exists():
            train_script = self.kohya_dir / "train_network.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--config_file", str(config_file)
        ]
        
        print(f"\n📝 Training-Befehl:")
        print(" ".join(cmd))
        print("\n⏳ Training läuft... (kann mehrere Stunden dauern)")
        print("💡 TensorBoard: tensorboard --logdir", str(self.logs_dir))
        print("🛑 Abbrechen: Ctrl+C\n")
        
        try:
            # Starte Training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Zeige Output in Echtzeit
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "="*60)
                print("✅ Training erfolgreich abgeschlossen!")
                print("="*60)
                print(f"📁 Output: {self.output_dir}")
                return True
            else:
                print("\n✗ Training mit Fehler beendet")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Training vom Benutzer abgebrochen")
            process.terminate()
            return False
        except Exception as e:
            print(f"\n✗ Training-Fehler: {e}")
            return False
    
    def run_full_pipeline(self):
        """Führt die komplette Training-Pipeline aus"""
        print("\n" + "="*60)
        print("🎯 SDXL LoRA Training Pipeline")
        print("="*60)
        
        # 1. GPU Check
        if not self.check_gpu():
            print("\n⚠️  Kein GPU erkannt! Training auf CPU ist extrem langsam.")
            response = input("Trotzdem fortfahren? [y/N]: ")
            if response.lower() != 'y':
                return
        
        # 2. Environment Setup
        self.setup_environment()
        
        # 3. Kohya Installation
        self.install_kohya_ss()
        
        # 4. Dataset Check
        if not self.prepare_dataset():
            print("\n✗ Dataset-Vorbereitung fehlgeschlagen")
            return
        
        # 5. Base Model Download
        self.download_base_model()
        
        # 6. Training Config
        config_file = self.create_training_config()
        
        # 7. Config Review
        print("\n📋 Training-Konfiguration:")
        print(f"  • Auflösung: {self.config['training']['resolution']}px")
        print(f"  • Batch Size: {self.config['training']['batch_size']}")
        print(f"  • Steps: {self.config['training']['max_train_steps']}")
        print(f"  • Learning Rate: {self.config['training']['learning_rate']}")
        print(f"  • LoRA Rank: {self.config['lora']['rank']}")
        
        response = input("\n▶️  Training starten? [Y/n]: ")
        if response.lower() == 'n':
            print("Abgebrochen. Config gespeichert für späteren Start.")
            self.save_config()
            return
        
        # 8. Start Training
        success = self.start_training(config_file)
        
        if success:
            # 9. Post-Training
            print("\n📦 Finalisiere Output...")
            self.save_config(self.output_dir / "training_config.json")
            
            # Liste Output-Dateien
            output_files = list(self.output_dir.glob("*.safetensors"))
            if output_files:
                print(f"\n✓ {len(output_files)} LoRA-Modelle erstellt:")
                for f in output_files:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  → {f.name} ({size_mb:.1f} MB)")
            
            print("\n💡 Nutzung:")
            print("  1. Download LoRA-Datei (.safetensors)")
            print("  2. In ComfyUI/A1111 unter models/lora/ ablegen")
            print("  3. Im Prompt verwenden: '<lora:FILENAME:0.8>'")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SDXL LoRA Training Pipeline')
    parser.add_argument('--config', type=str, help='Pfad zu Config-Datei (JSON)')
    parser.add_argument('--skip-setup', action='store_true', help='Überspringe Environment-Setup')
    
    args = parser.parse_args()
    
    pipeline = LoRATrainingPipeline(config_path=args.config)
    
    if args.skip_setup:
        print("ℹ️  Überspringe Setup, starte direkt Training...")
        if pipeline.prepare_dataset():
            config_file = pipeline.create_training_config()
            pipeline.start_training(config_file)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
