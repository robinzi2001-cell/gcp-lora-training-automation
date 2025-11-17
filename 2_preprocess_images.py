#!/usr/bin/env python3
"""
Bildvorverarbeitung für LoRA Training
- Umbenennung mit Custom Keyword
- Automatisches Tagging und Captioning
- BLIP/CLIPSeg Integration für Bildbeschreibungen
- Vorbereitung für SDXL Training
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

try:
    from PIL import Image
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("⚠️  Installiere erforderliche Bibliotheken...")
    print("   pip install pillow torch transformers")
    sys.exit(1)


class ImagePreprocessor:
    def __init__(self, keyword: str = None, input_dir: str = "raw_data", output_dir: str = "processed_data"):
        self.keyword = keyword or self.prompt_keyword()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = self.output_dir / "training_metadata.json"
        self.captions_file = self.output_dir / "captions.txt"
        
        # Erstelle Ausgabeverzeichnisse
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Lade Modelle für automatische Bildanalyse
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        
        # Statistiken
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "total_size_mb": 0
        }
    
    def prompt_keyword(self) -> str:
        """Fragt das Custom Keyword vom Benutzer ab"""
        print("\n" + "="*60)
        print("🎨 Bildvorverarbeitung für LoRA Training")
        print("="*60)
        keyword = input("\n🔤 Gib dein individuelles Keyword ein (z.B. 'ciri567'): ").strip()
        
        if not keyword:
            print("⚠️  Kein Keyword angegeben, verwende 'ciri567' als Standard")
            return "ciri567"
        
        # Validierung
        if not keyword.isalnum():
            print("⚠️  Keyword sollte nur Buchstaben und Zahlen enthalten")
            return self.prompt_keyword()
        
        print(f"✓ Keyword gesetzt: '{keyword}'")
        return keyword.lower()
    
    def load_models(self):
        """Lädt BLIP und CLIP Modelle für automatische Bildanalyse"""
        print("\n📥 Lade KI-Modelle für Bildanalyse (einmalig, kann 1-2 Min. dauern)...")
        
        try:
            # BLIP für Bildunterschriften
            print("  → BLIP für Captions...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # CLIP für semantische Analyse
            print("  → CLIP für Semantik...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # GPU falls verfügbar
            if torch.cuda.is_available():
                self.blip_model = self.blip_model.to("cuda")
                self.clip_model = self.clip_model.to("cuda")
                print("  ✓ Modelle auf GPU geladen")
            else:
                print("  ✓ Modelle auf CPU geladen (langsamer)")
                
        except Exception as e:
            print(f"⚠️  Fehler beim Laden der Modelle: {e}")
            print("  Fahre ohne automatische Captions fort...")
    
    def generate_caption(self, image_path: Path) -> str:
        """Generiert automatische Bildunterschrift mit BLIP"""
        if not self.blip_model:
            return ""
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"    ⚠️  Caption-Fehler: {e}")
            return ""
    
    def analyze_image_quality(self, image_path: Path) -> Dict:
        """Analysiert Bildqualität und gibt Empfehlungen"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            file_size = image_path.stat().st_size / (1024 * 1024)  # MB
            
            # Qualitätsbewertung
            min_dimension = min(width, height)
            quality_score = "good"
            recommendations = []
            
            if min_dimension < 512:
                quality_score = "poor"
                recommendations.append("Auflösung zu niedrig für SDXL (min. 512px)")
            elif min_dimension < 768:
                quality_score = "acceptable"
                recommendations.append("Besser wäre 1024px+ für SDXL")
            elif min_dimension >= 1024:
                quality_score = "excellent"
            
            if file_size > 10:
                recommendations.append("Datei sehr groß, erwäge Kompression")
            
            return {
                "width": width,
                "height": height,
                "size_mb": round(file_size, 2),
                "aspect_ratio": round(width / height, 2),
                "quality_score": quality_score,
                "recommendations": recommendations
            }
        except Exception as e:
            return {"error": str(e)}
    
    def process_image(self, image_path: Path, index: int) -> Optional[Dict]:
        """Verarbeitet ein einzelnes Bild"""
        try:
            # Neuer Dateiname mit Keyword
            new_filename = f"{self.keyword}_{index:04d}.jpg"
            output_path = self.output_dir / "images" / new_filename
            
            # Öffne und konvertiere Bild
            img = Image.open(image_path)
            
            # Konvertiere zu RGB falls nötig
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Analysiere Qualität
            quality_info = self.analyze_image_quality(image_path)
            
            # Generiere Caption
            caption = self.generate_caption(image_path)
            full_caption = f"{self.keyword}, {caption}" if caption else self.keyword
            
            # Speichere optimiertes Bild
            img.save(output_path, 'JPEG', quality=95, optimize=True)
            
            # Statistiken aktualisieren
            self.stats["processed"] += 1
            self.stats["total_size_mb"] += quality_info.get("size_mb", 0)
            
            # Metadaten
            metadata = {
                "original_file": str(image_path.name),
                "new_file": new_filename,
                "caption": full_caption,
                "keyword": self.keyword,
                "quality": quality_info,
                "processed_at": datetime.now().isoformat(),
                "hash": hashlib.md5(output_path.read_bytes()).hexdigest()
            }
            
            print(f"  ✓ {image_path.name} → {new_filename}")
            if quality_info.get("recommendations"):
                for rec in quality_info["recommendations"]:
                    print(f"    ℹ️  {rec}")
            
            return metadata
            
        except Exception as e:
            print(f"  ✗ Fehler bei {image_path.name}: {e}")
            self.stats["errors"] += 1
            return None
    
    def find_images(self) -> List[Path]:
        """Findet alle Bilddateien im Input-Verzeichnis"""
        supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        
        for ext in supported_formats:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def save_metadata(self, metadata_list: List[Dict]):
        """Speichert Metadaten in JSON und Captions in TXT"""
        # JSON Metadaten
        metadata_dict = {
            "keyword": self.keyword,
            "processed_at": datetime.now().isoformat(),
            "statistics": self.stats,
            "images": metadata_list
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadaten gespeichert: {self.metadata_file}")
        
        # Captions für Training (jede Zeile: filename.jpg,caption)
        with open(self.captions_file, 'w', encoding='utf-8') as f:
            for item in metadata_list:
                if item:
                    filename = item['new_file'].replace('.jpg', '')
                    caption = item['caption']
                    f.write(f"{filename},{caption}\n")
        
        print(f"✓ Captions gespeichert: {self.captions_file}")
        
        # Zusätzlich: Einzelne .txt-Dateien pro Bild (für einige Trainer erforderlich)
        for item in metadata_list:
            if item:
                txt_file = self.output_dir / "images" / item['new_file'].replace('.jpg', '.txt')
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(item['caption'])
    
    def process_all(self):
        """Hauptverarbeitungsschleife"""
        print("\n" + "="*60)
        print(f"🖼️  Starte Bildverarbeitung für Keyword: '{self.keyword}'")
        print("="*60)
        
        # Finde Bilder
        images = self.find_images()
        if not images:
            print(f"\n✗ Keine Bilder gefunden in {self.input_dir}")
            print("  Unterstützte Formate: .jpg, .jpeg, .png, .webp, .bmp")
            return
        
        print(f"\n📁 Gefunden: {len(images)} Bilder")
        
        # Lade Modelle
        load_models = input("\n🤖 Automatische Captions generieren? (empfohlen) [Y/n]: ")
        if load_models.lower() != 'n':
            self.load_models()
        
        print("\n⚙️  Verarbeite Bilder...")
        print("-" * 60)
        
        # Verarbeite jedes Bild
        metadata_list = []
        for idx, img_path in enumerate(images, 1):
            metadata = self.process_image(img_path, idx)
            if metadata:
                metadata_list.append(metadata)
        
        # Speichere Metadaten
        if metadata_list:
            self.save_metadata(metadata_list)
        
        # Zusammenfassung
        print("\n" + "="*60)
        print("✅ Verarbeitung abgeschlossen!")
        print("="*60)
        print(f"✓ Verarbeitet: {self.stats['processed']}")
        print(f"⊘ Übersprungen: {self.stats['skipped']}")
        print(f"✗ Fehler: {self.stats['errors']}")
        print(f"💾 Gesamtgröße: {self.stats['total_size_mb']:.2f} MB")
        print(f"\n📂 Ausgabeverzeichnis: {self.output_dir.absolute()}")
        print(f"  → Bilder: {self.output_dir / 'images'}")
        print(f"  → Metadaten: {self.metadata_file}")
        print(f"  → Captions: {self.captions_file}")
        print("\n💡 Nächster Schritt: VM vorbereiten mit 3_prepare_training.py")
        print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bildvorverarbeitung für LoRA Training')
    parser.add_argument('--keyword', type=str, help='Custom Keyword (z.B. ciri567)')
    parser.add_argument('--input', type=str, default='raw_data', help='Input-Verzeichnis')
    parser.add_argument('--output', type=str, default='processed_data', help='Output-Verzeichnis')
    
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor(
        keyword=args.keyword,
        input_dir=args.input,
        output_dir=args.output
    )
    preprocessor.process_all()


if __name__ == "__main__":
    main()
