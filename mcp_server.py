#!/usr/bin/env python3
"""
MCP Server für GCP LoRA Training
Ermöglicht direkte Steuerung über Claude Desktop
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

# MCP SDK (wenn verfügbar)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP SDK nicht installiert. Installiere mit: pip install mcp")


class GCPLoRaMCPServer:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT", "lora567")
        self.zone = os.getenv("GCP_ZONE", "us-central1-a")
        self.instance_name = "lora-training-vm"
        self.work_dir = Path("/home/lora_training")
        
    def _run_gcloud_command(self, cmd: List[str]) -> Dict:
        """Führt gcloud-Befehl aus und gibt Result zurück"""
        try:
            result = subprocess.run(
                ["gcloud"] + cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return {
                "success": True,
                "output": result.stdout,
                "error": None
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "output": e.stdout,
                "error": e.stderr
            }
    
    def _run_ssh_command(self, cmd: str) -> Dict:
        """Führt Befehl auf VM via SSH aus"""
        ssh_cmd = [
            "compute", "ssh", self.instance_name,
            "--zone", self.zone,
            "--command", cmd
        ]
        return self._run_gcloud_command(ssh_cmd)
    
    async def provision_vm(self, gpu_type: str = "a100") -> Dict:
        """
        Erstellt VM mit GPU für Training
        
        Args:
            gpu_type: "a100", "l4", oder "t4"
        """
        print(f"🚀 Erstelle VM mit {gpu_type.upper()} GPU...")
        
        result = subprocess.run(
            ["python", "1_provision_vm.py"],
            capture_output=True,
            text=True
        )
        
        return {
            "success": result.returncode == 0,
            "message": "VM erfolgreich erstellt" if result.returncode == 0 else "VM-Erstellung fehlgeschlagen",
            "output": result.stdout,
            "vm_name": self.instance_name,
            "zone": self.zone
        }
    
    async def get_vm_status(self) -> Dict:
        """Gibt aktuellen VM-Status zurück"""
        result = self._run_gcloud_command([
            "compute", "instances", "describe",
            self.instance_name,
            "--zone", self.zone,
            "--format", "json"
        ])
        
        if result["success"]:
            try:
                vm_data = json.loads(result["output"])
                return {
                    "success": True,
                    "status": vm_data.get("status", "UNKNOWN"),
                    "name": vm_data.get("name"),
                    "machine_type": vm_data.get("machineType", "").split("/")[-1],
                    "zone": self.zone,
                    "internal_ip": vm_data.get("networkInterfaces", [{}])[0].get("networkIP"),
                    "external_ip": vm_data.get("networkInterfaces", [{}])[0].get("accessConfigs", [{}])[0].get("natIP")
                }
            except json.JSONDecodeError:
                return {"success": False, "error": "JSON Parse Error"}
        else:
            return {"success": False, "error": result["error"]}
    
    async def upload_images(self, local_path: str) -> Dict:
        """
        Lädt Bilder auf VM hoch
        
        Args:
            local_path: Lokaler Pfad zu Bildern oder Verzeichnis
        """
        print(f"📤 Lade Bilder hoch von {local_path}...")
        
        # Prüfe ob Pfad existiert
        path = Path(local_path)
        if not path.exists():
            return {"success": False, "error": f"Pfad nicht gefunden: {local_path}"}
        
        # Upload via gcloud scp
        result = self._run_gcloud_command([
            "compute", "scp",
            "--recurse" if path.is_dir() else "",
            str(path),
            f"{self.instance_name}:{self.work_dir}/raw_data/",
            "--zone", self.zone
        ])
        
        if result["success"]:
            # Zähle hochgeladene Dateien
            count_result = self._run_ssh_command(
                f"ls -1 {self.work_dir}/raw_data/ | wc -l"
            )
            file_count = count_result["output"].strip() if count_result["success"] else "unknown"
            
            return {
                "success": True,
                "message": f"Upload erfolgreich",
                "files_uploaded": file_count
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    
    async def preprocess_images(self, keyword: str) -> Dict:
        """
        Führt Bildvorverarbeitung auf VM aus
        
        Args:
            keyword: Custom Keyword für Training
        """
        print(f"⚙️  Starte Bildvorverarbeitung für Keyword: {keyword}")
        
        # Führe Preprocessing-Skript auf VM aus
        cmd = f"cd {self.work_dir} && python 2_preprocess_images.py --keyword {keyword}"
        result = self._run_ssh_command(cmd)
        
        if result["success"]:
            # Hole Statistiken
            stats_result = self._run_ssh_command(
                f"cat {self.work_dir}/processed_data/training_metadata.json"
            )
            
            stats = {}
            if stats_result["success"]:
                try:
                    stats = json.loads(stats_result["output"])
                except json.JSONDecodeError:
                    pass
            
            return {
                "success": True,
                "message": "Vorverarbeitung abgeschlossen",
                "keyword": keyword,
                "statistics": stats.get("statistics", {})
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    
    async def start_training(self, config: Optional[Dict] = None) -> Dict:
        """
        Startet LoRA Training auf VM
        
        Args:
            config: Optional custom training config
        """
        print("🎯 Starte LoRA Training...")
        
        cmd = f"cd {self.work_dir} && nohup python 3_train_lora.py"
        
        if config:
            # Speichere Config auf VM
            config_json = json.dumps(config)
            upload_config = self._run_ssh_command(
                f"echo '{config_json}' > {self.work_dir}/custom_config.json"
            )
            if upload_config["success"]:
                cmd += " --config custom_config.json"
        
        cmd += " > training.log 2>&1 &"
        
        result = self._run_ssh_command(cmd)
        
        return {
            "success": result["success"],
            "message": "Training gestartet" if result["success"] else "Training-Start fehlgeschlagen",
            "log_command": f"gcloud compute ssh {self.instance_name} --zone {self.zone} --command 'tail -f {self.work_dir}/training.log'"
        }
    
    async def get_training_status(self) -> Dict:
        """Gibt Training-Status und Progress zurück"""
        # Prüfe ob Training läuft
        process_check = self._run_ssh_command(
            "pgrep -f '3_train_lora.py'"
        )
        
        is_running = bool(process_check["output"].strip())
        
        # Hole letzte Log-Zeilen
        log_result = self._run_ssh_command(
            f"tail -n 20 {self.work_dir}/training.log"
        )
        
        # Prüfe Output-Verzeichnis
        output_check = self._run_ssh_command(
            f"ls -lh {self.work_dir}/output/*.safetensors 2>/dev/null | wc -l"
        )
        
        models_count = int(output_check["output"].strip()) if output_check["success"] else 0
        
        return {
            "success": True,
            "is_running": is_running,
            "status": "running" if is_running else "stopped",
            "latest_logs": log_result["output"] if log_result["success"] else "",
            "models_created": models_count
        }
    
    async def download_lora(self, output_path: str = "./") -> Dict:
        """
        Lädt fertiges LoRA-Modell herunter
        
        Args:
            output_path: Lokaler Ziel-Pfad
        """
        print(f"📥 Lade LoRA-Modell herunter nach {output_path}...")
        
        # Liste verfügbare Modelle
        list_result = self._run_ssh_command(
            f"ls -1 {self.work_dir}/output/*.safetensors"
        )
        
        if not list_result["success"] or not list_result["output"].strip():
            return {
                "success": False,
                "error": "Keine LoRA-Modelle gefunden"
            }
        
        models = list_result["output"].strip().split("\n")
        
        # Download alle Modelle
        downloaded = []
        for model in models:
            model = model.strip()
            result = self._run_gcloud_command([
                "compute", "scp",
                f"{self.instance_name}:{model}",
                output_path,
                "--zone", self.zone
            ])
            
            if result["success"]:
                downloaded.append(Path(model).name)
        
        return {
            "success": len(downloaded) > 0,
            "message": f"{len(downloaded)} Modell(e) heruntergeladen",
            "files": downloaded,
            "output_path": output_path
        }
    
    async def stop_vm(self) -> Dict:
        """Stoppt VM (behält Daten, spart Kosten)"""
        print("🛑 Stoppe VM...")
        
        result = self._run_gcloud_command([
            "compute", "instances", "stop",
            self.instance_name,
            "--zone", self.zone
        ])
        
        return {
            "success": result["success"],
            "message": "VM gestoppt" if result["success"] else "Fehler beim Stoppen",
            "note": "Daten bleiben erhalten. Start mit: gcloud compute instances start"
        }
    
    async def delete_vm(self) -> Dict:
        """Löscht VM komplett (IRREVERSIBEL!)"""
        print("🗑️  Lösche VM...")
        
        result = self._run_gcloud_command([
            "compute", "instances", "delete",
            self.instance_name,
            "--zone", self.zone,
            "--quiet"
        ])
        
        return {
            "success": result["success"],
            "message": "VM gelöscht" if result["success"] else "Fehler beim Löschen",
            "warning": "Alle Daten wurden unwiderruflich gelöscht!"
        }


# MCP Server Setup (wenn SDK verfügbar)
if MCP_AVAILABLE:
    server = Server("gcp-lora-training")
    mcp_handler = GCPLoRaMCPServer()
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Liste verfügbarer Tools"""
        return [
            Tool(
                name="provision_vm",
                description="Erstellt GCP VM mit GPU für LoRA Training",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "gpu_type": {
                            "type": "string",
                            "enum": ["a100", "l4", "t4"],
                            "default": "a100",
                            "description": "GPU-Typ: a100 (80GB), l4 (24GB), oder t4 (16GB)"
                        }
                    }
                }
            ),
            Tool(
                name="get_vm_status",
                description="Gibt aktuellen Status der VM zurück"
            ),
            Tool(
                name="upload_images",
                description="Lädt Bilder auf VM hoch",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "local_path": {
                            "type": "string",
                            "description": "Lokaler Pfad zu Bildern"
                        }
                    },
                    "required": ["local_path"]
                }
            ),
            Tool(
                name="preprocess_images",
                description="Verarbeitet Bilder mit Custom Keyword",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Individuelles Keyword für LoRA"
                        }
                    },
                    "required": ["keyword"]
                }
            ),
            Tool(
                name="start_training",
                description="Startet LoRA Training auf VM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "description": "Optional: Custom Training Config"
                        }
                    }
                }
            ),
            Tool(
                name="get_training_status",
                description="Gibt Training-Status und Progress zurück"
            ),
            Tool(
                name="download_lora",
                description="Lädt fertiges LoRA-Modell herunter",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "default": "./",
                            "description": "Lokaler Ziel-Pfad"
                        }
                    }
                }
            ),
            Tool(
                name="stop_vm",
                description="Stoppt VM (behält Daten, spart Kosten)"
            ),
            Tool(
                name="delete_vm",
                description="Löscht VM komplett (IRREVERSIBEL!)"
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Tool-Handler"""
        handler_map = {
            "provision_vm": mcp_handler.provision_vm,
            "get_vm_status": mcp_handler.get_vm_status,
            "upload_images": mcp_handler.upload_images,
            "preprocess_images": mcp_handler.preprocess_images,
            "start_training": mcp_handler.start_training,
            "get_training_status": mcp_handler.get_training_status,
            "download_lora": mcp_handler.download_lora,
            "stop_vm": mcp_handler.stop_vm,
            "delete_vm": mcp_handler.delete_vm
        }
        
        handler = handler_map.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        try:
            result = await handler(**arguments)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": str(e)}, indent=2)
            )]
    
    async def main():
        """MCP Server Hauptfunktion"""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())
else:
    # Fallback: Standalone-Nutzung ohne MCP
    print("="*60)
    print("⚠️  MCP SDK nicht installiert")
    print("Installiere mit: pip install mcp")
    print("\nOder nutze die Skripte direkt:")
    print("  1. python 1_provision_vm.py")
    print("  2. python 2_preprocess_images.py")
    print("  3. python 3_train_lora.py")
    print("="*60)
