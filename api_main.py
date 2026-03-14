#!/usr/bin/env python3
"""
API FastAPI pour le SoccerNet Tracker
Expose l'algorithme de détection d'actions via une interface REST
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import zipfile
from datetime import datetime

# Import de votre algorithme principal
from briqx_video_clipper5 import SoccerNetRealtimeTracker  # Seule classe qui existe

# Configuration
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("api_results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="SoccerNet Action Detection API",
    description="API pour détecter les actions des joueurs dans des vidéos de football",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage des tâches en cours
tasks = {}

class VideoProcessor:
    """Classe pour gérer le traitement vidéo en arrière-plan"""
    
    def __init__(self, video_path: Path, task_id: str):
        self.video_path = video_path
        self.task_id = task_id
        self.output_dir = RESULTS_DIR / task_id
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "processing.log"
        
    def run(self):
        """Exécute le traitement vidéo"""
        try:
            logger.info(f"Démarrage du traitement pour tâche {self.task_id}")
            
            # Configuration du logging pour cette tâche
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            # === UTILISATION DE VOTRE VRAI ALGORITHME ===
            # Créer une instance du tracker
            tracker = SoccerNetRealtimeTracker()
            
            # Surcharger le chemin de la vidéo
            tracker.video_path = self.video_path
            
            # Surcharger le dossier de sortie
            tracker.output_base = self.output_dir / "SOCCERNET_REALTIME"
            tracker.players_base = tracker.output_base / "PLAYERS"
            tracker.actions_base = tracker.output_base / "ACTIONS"
            tracker.reports_base = tracker.output_base / "REPORTS"
            tracker.frames_base = tracker.output_base / "ANALYSIS_FRAMES"
            
            # Créer les dossiers
            tracker._create_directories()
            
            # Lancer le traitement
            tracker.run()
            
            # Récupérer les résultats
            results = {
                "task_id": self.task_id,
                "status": "completed",
                "video": str(self.video_path.name),
                "actions_detected": tracker.stats['total_actions_detected'],
                "players_detected": len(tracker.player_tracker.get_all_players()),
                "clips_created": tracker.stats['clips_created'],
                "processing_time": tracker.stats.get('processing_time', 0),
                "output_folder": str(tracker.output_base)
            }
            
            # Sauvegarder les résultats
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Créer un fichier ZIP avec les résultats
            self._create_results_zip()
            
            tasks[self.task_id]["status"] = "completed"
            tasks[self.task_id]["results"] = results
            
            logger.info(f"Traitement terminé pour tâche {self.task_id}")
            
        except Exception as e:
            logger.error(f"Erreur dans le traitement: {e}")
            tasks[self.task_id]["status"] = "failed"
            tasks[self.task_id]["error"] = str(e)
    
    def _create_results_zip(self):
        """Crée une archive ZIP des résultats"""
        zip_path = self.output_dir / "results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Ajouter tous les fichiers du dossier de résultats
            results_folder = self.output_dir / "SOCCERNET_REALTIME"
            if results_folder.exists():
                for root, dirs, files in os.walk(results_folder):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.output_dir)
                        zipf.write(file_path, arcname)
            
            # Ajouter le fichier de résultats JSON
            results_json = self.output_dir / "results.json"
            if results_json.exists():
                zipf.write(results_json, "results.json")

@app.get("/")
async def root():
    return {
        "message": "SoccerNet Action Detection API",
        "version": "1.0.0",
        "endpoints": ["/docs", "/health", "/detect/upload", "/task/{task_id}", "/download/{task_id}"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/detect/upload")
async def detect_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Télécharge une vidéo et lance la détection d'actions
    """
    try:
        # Vérifier le type de fichier
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(400, "Format de fichier non supporté")
        
        # Générer un ID de tâche unique
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename.replace(".", "_")
        
        # Sauvegarder le fichier
        video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialiser la tâche
        tasks[task_id] = {
            "status": "processing",
            "video": file.filename,
            "video_path": str(video_path),
            "created_at": datetime.now().isoformat()
        }
        
        # Lancer le traitement en arrière-plan
        processor = VideoProcessor(video_path, task_id)
        background_tasks.add_task(processor.run)
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Traitement démarré",
            "check_status_url": f"/task/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Récupère le statut d'une tâche"""
    if task_id not in tasks:
        raise HTTPException(404, "Tâche non trouvée")
    
    task = tasks[task_id]
    
    # Si la tâche est terminée, charger les résultats
    if task["status"] == "completed":
        results_file = RESULTS_DIR / task_id / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            task["results"] = results
    
    return task

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    """Télécharge les résultats d'une tâche"""
    if task_id not in tasks:
        raise HTTPException(404, "Tâche non trouvée")
    
    zip_path = RESULTS_DIR / task_id / "results.zip"
    
    if not zip_path.exists():
        raise HTTPException(404, "Résultats non disponibles")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"results_{task_id}.zip"
    )

if __name__ == "__main__":
    # Installer python-multipart d'abord !
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )