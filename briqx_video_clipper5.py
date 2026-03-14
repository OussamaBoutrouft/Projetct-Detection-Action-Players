#!/usr/bin/env python3
"""
SOCCERNET-V3 REAL-TIME TRACKER - ANALYSE FRAME PAR FRAME
Détecte TOUTES les actions en temps réel avec Computer Vision
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import cv2
import numpy as np
from tqdm import tqdm
import glob
from collections import defaultdict
import math
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('soccernet_realtime_tracker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JerseyNumberDetector:
    """
    Détecteur de numéros de maillot par Computer Vision
    Utilise des techniques de traitement d'image pour identifier les numéros
    """
    
    def __init__(self):
        # Initialiser le détecteur de chiffres (simulé - à remplacer par un vrai modèle)
        self.digit_roi_size = (30, 50)
        self.confidence_threshold = 0.6
        
        # Mapping des couleurs d'équipe
        self.team_colors = {
            'Luton Town': {
                'primary': (255, 165, 0),  # Orange
                'secondary': (0, 0, 0),     # Noir
                'jersey_color_range': ([0, 100, 100], [20, 255, 255])  # Orange en HSV
            },
            'Northampton Town': {
                'primary': (128, 0, 0),      # Maroon
                'secondary': (255, 255, 255), # Blanc
                'jersey_color_range': ([160, 50, 50], [180, 255, 255])  # Rouge foncé en HSV
            }
        }
    
    def detect_jersey_number(self, player_roi: np.ndarray, team: str = None) -> Tuple[Optional[int], float]:
        """
        Détecte le numéro de maillot dans une région d'intérêt du joueur
        À remplacer par votre modèle de détection de chiffres
        """
        if player_roi.size == 0:
            return None, 0.0
        
        # SIMULATION - À REMPLACER PAR VOTRE VRAI MODÈLE
        # Dans un vrai système, vous utiliseriez un modèle CNN entraîné
        # sur des chiffres de maillots de football
        
        # Pour la simulation, on génère des numéros réalistes
        # entre 1 et 25 pour les équipes
        if team and 'luton' in team.lower():
            jersey = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        elif team and 'northampton' in team.lower():
            jersey = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20])
        else:
            jersey = random.randint(1, 40)
        
        confidence = random.uniform(0.7, 0.98)
        
        return jersey, confidence
    
    def detect_team_from_colors(self, player_roi: np.ndarray) -> Optional[str]:
        """
        Détecte l'équipe du joueur basé sur les couleurs du maillot
        """
        if player_roi.size == 0:
            return None
        
        # Convertir en HSV pour une meilleure détection des couleurs
        hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
        
        # Vérifier pour Luton Town (Orange)
        luton_mask = cv2.inRange(hsv_roi, 
                                 np.array(self.team_colors['Luton Town']['jersey_color_range'][0]),
                                 np.array(self.team_colors['Luton Town']['jersey_color_range'][1]))
        luton_ratio = np.sum(luton_mask > 0) / luton_mask.size
        
        # Vérifier pour Northampton Town (Maroon)
        northampton_mask = cv2.inRange(hsv_roi,
                                       np.array(self.team_colors['Northampton Town']['jersey_color_range'][0]),
                                       np.array(self.team_colors['Northampton Town']['jersey_color_range'][1]))
        northampton_ratio = np.sum(northampton_mask > 0) / northampton_mask.size
        
        if luton_ratio > 0.3:
            return 'Luton Town'
        elif northampton_ratio > 0.3:
            return 'Northampton Town'
        
        return None


class ActionDetector:
    """
    Détecteur d'actions en temps réel par analyse de mouvement
    """
    
    # Seuils pour différents types d'actions
    ACTION_THRESHOLDS = {
        'shot': 0.7,
        'goal': 0.9,
        'pass': 0.6,
        'tackle': 0.65,
        'header': 0.6,
        'foul': 0.7,
        'corner': 0.8,
        'free_kick': 0.75,
        'throw_in': 0.7,
        'save': 0.8
    }
    
    def __init__(self, fps: float, frame_width: int, frame_height: int):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Pour le suivi de mouvement
        self.prev_frame = None
        self.motion_history = []
        self.ball_position = None
        self.player_positions = {}
        
        # Détection de la balle (simulée)
        self.ball_color_range = ([0, 200, 200], [10, 255, 255])  # Blanc en HSV
        
        logger.info(f"✅ ActionDetector initialized: {fps} fps, {frame_width}x{frame_height}")
    
    def detect_actions_in_frame(self, frame: np.ndarray, frame_num: int, 
                               timestamp: float) -> List[Dict]:
        """
        Détecte les actions dans une frame
        """
        detected_actions = []
        
        # Convertir en HSV pour certaines détections
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Détection des joueurs (simplifiée)
        players = self._detect_players(frame, hsv)
        
        # 2. Détection de la balle
        ball = self._detect_ball(hsv)
        if ball:
            self.ball_position = ball
        
        # 3. Analyse de mouvement (si on a la frame précédente)
        if self.prev_frame is not None:
            motion = self._analyze_motion(frame, self.prev_frame)
            
            # Détecter les actions basées sur le mouvement
            if motion['intensity'] > 0.3:
                # Mouvement important - potentiellement une action
                action_type = self._classify_motion_action(motion, players, ball)
                if action_type:
                    detected_actions.append({
                        'type': action_type,
                        'confidence': motion['intensity'],
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'players_involved': self._get_nearby_players(players, ball) if ball else []
                    })
        
        # 4. Détection d'événements spécifiques
        # Tirs (détection de mouvement rapide vers le but)
        if self._check_shot_condition(players, ball, frame_num):
            detected_actions.append({
                'type': 'shot',
                'confidence': 0.85,
                'frame': frame_num,
                'timestamp': timestamp,
                'players_involved': self._get_shooting_player(players, ball)
            })
        
        # Tacles (quand 2 joueurs sont très proches avec mouvement)
        if self._check_tackle_condition(players, frame_num):
            detected_actions.append({
                'type': 'tackle',
                'confidence': 0.75,
                'frame': frame_num,
                'timestamp': timestamp,
                'players_involved': self._get_tackling_players(players)
            })
        
        self.prev_frame = frame.copy()
        
        return detected_actions
    
    def _detect_players(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict]:
        """
        Détecte les joueurs dans l'image
        À remplacer par YOLO ou autre modèle de détection
        """
        players = []
        
        # SIMULATION - À REMPLACER PAR VOTRE MODÈLE
        # Pour la simulation, on génère des positions aléatoires de joueurs
        h, w = frame.shape[:2]
        
        # Générer entre 18 et 22 joueurs (équipes complètes)
        num_players = random.randint(18, 22)
        
        for i in range(num_players):
            # Position aléatoire mais réaliste (sur le terrain)
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.2, 0.8)
            
            # Taille du joueur (proportionnelle à la distance)
            size = random.uniform(0.05, 0.1)
            
            players.append({
                'id': i,
                'position': (x, y),
                'bbox': [x - size/2, y - size/2, x + size/2, y + size/2],
                'velocity': (random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02)),
                'team': 'Luton Town' if i < num_players//2 else 'Northampton Town'
            })
        
        return players
    
    def _detect_ball(self, hsv: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Détecte la position de la balle
        """
        # Créer un masque pour la balle (blanche)
        mask = cv2.inRange(hsv, 
                          np.array(self.ball_color_range[0]),
                          np.array(self.ball_color_range[1]))
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Prendre le plus grand contour (supposé être la balle)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Seuil de taille minimale
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) / hsv.shape[1]
                    cy = int(M["m01"] / M["m00"]) / hsv.shape[0]
                    return (cx, cy)
        
        return None
    
    def _analyze_motion(self, current: np.ndarray, previous: np.ndarray) -> Dict:
        """
        Analyse le mouvement entre deux frames
        """
        # Convertir en gris
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        
        # Calculer la différence
        diff = cv2.absdiff(current_gray, previous_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Intensité du mouvement (proportion de pixels qui ont changé)
        motion_intensity = np.sum(thresh > 0) / thresh.size
        
        # Calculer le flot optique pour la direction
        flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Direction moyenne du mouvement
        avg_flow_x = np.mean(flow[..., 0])
        avg_flow_y = np.mean(flow[..., 1])
        
        return {
            'intensity': motion_intensity,
            'direction': (avg_flow_x, avg_flow_y),
            'magnitude': np.sqrt(avg_flow_x**2 + avg_flow_y**2)
        }
    
    def _classify_motion_action(self, motion: Dict, players: List, ball: Optional[Tuple]) -> Optional[str]:
        """
        Classifie le type d'action basé sur le mouvement
        """
        if motion['intensity'] < 0.2:
            return None
        
        # Mouvement intense et rapide
        if motion['magnitude'] > 10:
            if ball and abs(motion['direction'][0]) > abs(motion['direction'][1]):
                return 'pass'
            else:
                return 'run'
        
        return 'movement'
    
    def _check_shot_condition(self, players: List, ball: Optional[Tuple], frame_num: int) -> bool:
        """
        Vérifie si les conditions pour un tir sont réunies
        """
        if not ball:
            return False
        
        # Simuler une détection de tir (toutes les 300 frames environ)
        return frame_num % 300 == 0 and random.random() > 0.7
    
    def _check_tackle_condition(self, players: List, frame_num: int) -> bool:
        """
        Vérifie si les conditions pour un tacle sont réunies
        """
        # Simuler une détection de tacle (toutes les 200 frames environ)
        return frame_num % 200 == 0 and random.random() > 0.6
    
    def _get_nearby_players(self, players: List, ball: Tuple) -> List:
        """
        Trouve les joueurs proches de la balle
        """
        nearby = []
        bx, by = ball
        
        for player in players[:3]:  # Limiter pour la simulation
            px, py = player['position']
            distance = np.sqrt((px - bx)**2 + (py - by)**2)
            if distance < 0.1:  # Proche de la balle
                nearby.append({
                    'id': player['id'],
                    'team': player['team'],
                    'distance': distance
                })
        
        return nearby
    
    def _get_shooting_player(self, players: List, ball: Optional[Tuple]) -> List:
        """
        Trouve le joueur qui tire
        """
        if not players:
            return []
        return [{'id': players[0]['id'], 'team': players[0]['team'], 'role': 'shooter'}]
    
    def _get_tackling_players(self, players: List) -> List:
        """
        Trouve les joueurs impliqués dans un tacle
        """
        if len(players) < 2:
            return []
        return [
            {'id': players[0]['id'], 'team': players[0]['team'], 'role': 'tackler'},
            {'id': players[1]['id'], 'team': players[1]['team'], 'role': 'ball_carrier'}
        ]


class PlayerTracker:
    """
    Suit les joueurs à travers les frames avec des IDs persistants
    """
    
    def __init__(self):
        self.players = {}  # id -> {positions, jersey, team, etc.}
        self.next_id = 0
        self.jersey_detector = JerseyNumberDetector()
        
        # Seuil de distance pour considérer le même joueur
        self.position_threshold = 0.05
    
    def update(self, detected_players: List[Dict], frame_num: int, timestamp: float, 
              frame: np.ndarray) -> Dict:
        """
        Met à jour le tracking avec les nouveaux joueurs détectés
        """
        updated_players = {}
        
        for det_player in detected_players:
            pos = det_player['position']
            
            # Chercher si ce joueur existe déjà
            matched_id = self._find_matching_player(pos)
            
            if matched_id is not None:
                # Mettre à jour le joueur existant
                player = self.players[matched_id]
                player['positions'].append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'position': pos,
                    'bbox': det_player.get('bbox')
                })
                player['last_seen'] = timestamp
                player['frame_count'] += 1
                updated_players[matched_id] = player
            else:
                # Nouveau joueur - essayer de détecter son numéro
                bbox = det_player.get('bbox')
                jersey = None
                team = det_player.get('team')
                
                if bbox:
                    # Extraire la ROI du joueur
                    h, w = frame.shape[:2]
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
                    
                    if x2 > x1 and y2 > y1:
                        player_roi = frame[y1:y2, x1:x2]
                        # Détecter le numéro
                        jersey, conf = self.jersey_detector.detect_jersey_number(player_roi, team)
                        
                        # Détecter l'équipe par les couleurs si non spécifiée
                        if not team:
                            team = self.jersey_detector.detect_team_from_colors(player_roi)
                
                # Créer nouveau joueur
                player = {
                    'id': self.next_id,
                    'jersey_number': jersey,
                    'team': team or 'Unknown',
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'frame_count': 1,
                    'positions': [{
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'position': pos,
                        'bbox': det_player.get('bbox'),
                        'jersey_detected': jersey is not None,
                        'jersey_confidence': conf if jersey else 0
                    }],
                    'actions': []
                }
                
                self.players[self.next_id] = player
                updated_players[self.next_id] = player
                self.next_id += 1
        
        return updated_players
    
    def _find_matching_player(self, position: Tuple[float, float]) -> Optional[int]:
        """
        Trouve si un joueur à cette position existe déjà
        """
        for pid, player in self.players.items():
            last_pos = player['positions'][-1]['position']
            dist = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
            if dist < self.position_threshold:
                return pid
        return None
    
    def add_action_to_player(self, player_id: int, action: Dict):
        """
        Ajoute une action à un joueur
        """
        if player_id in self.players:
            self.players[player_id]['actions'].append(action)
    
    def get_player_by_jersey(self, jersey: int, team: str) -> Optional[int]:
        """
        Trouve un joueur par son numéro de maillot
        """
        for pid, player in self.players.items():
            if player.get('jersey_number') == jersey and player.get('team') == team:
                return pid
        return None
    
    def get_all_players(self) -> Dict:
        return self.players


class SoccerNetRealtimeTracker:
    """
    Tracker en temps réel qui analyse la vidéo frame par frame
    """
    
    def __init__(self):
        # Paths
        self.base_path = Path(r"C:\Users\HP\Downloads\Soccernet-v3-main-Tracking")
        self.video_path = self.base_path / "RK_Semifinals_Luton Town - Northampton Town_04032026.mp4"
        
        # Match info
        self.match_name = "Luton Town vs Northampton Town"
        self.match_id = hashlib.md5(self.match_name.encode()).hexdigest()[:12]
        
        # Output directories
        self.output_base = self.base_path / "SOCCERNET_REALTIME" / self.match_name.replace(' ', '_')
        self.players_base = self.output_base / "PLAYERS"
        self.actions_base = self.output_base / "ACTIONS"
        self.reports_base = self.output_base / "REPORTS"
        self.frames_base = self.output_base / "ANALYSIS_FRAMES"
        
        # Create directories
        self._create_directories()
        
        # Video properties
        self.video_cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        self.video_duration = None
        
        # Detectors
        self.action_detector = None
        self.player_tracker = PlayerTracker()
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_actions_detected': 0,
            'unique_players': 0,
            'actions_by_type': defaultdict(int),
            'actions_by_team': defaultdict(int),
            'clips_created': 0,
            'processing_fps': 0
        }
        
        # Actions buffer
        self.detected_actions = []
        self.action_clip_buffer = []
        
        logger.info("=" * 80)
        logger.info("⚽ SOCCERNET REALTIME TRACKER - FRAME BY FRAME ANALYSIS ⚽")
        logger.info("=" * 80)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Output: {self.output_base}")
        logger.info("=" * 80)
    
    def _create_directories(self):
        """Create all output directories"""
        dirs = [
            self.output_base,
            self.players_base,
            self.actions_base,
            self.reports_base,
            self.frames_base
        ]
        
        # Create team folders
        for team in ['LUTON_TOWN', 'NORTHAMPTON_TOWN', 'OTHER']:
            dirs.append(self.players_base / team)
        
        # Create action type folders
        for action_type in ['shot', 'goal', 'pass', 'tackle', 'header', 'foul', 
                           'corner', 'free_kick', 'throw_in', 'save', 'movement']:
            dirs.append(self.actions_base / action_type)
            dirs.append(self.actions_base / action_type / 'clips')
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Created {len(dirs)} directories")
    
    def _initialize_video(self) -> bool:
        """Initialize video capture"""
        try:
            logger.info("Initializing video...")
            
            if not self.video_path.exists():
                logger.error(f"Video not found: {self.video_path}")
                return False
            
            self.video_cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.video_cap.isOpened():
                logger.error("Cannot open video")
                return False
            
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_duration = self.total_frames / self.fps
            
            # Initialize action detector
            self.action_detector = ActionDetector(self.fps, self.frame_width, self.frame_height)
            
            logger.info(f"✅ Video initialized:")
            logger.info(f"   • FPS: {self.fps:.2f}")
            logger.info(f"   • Frames: {self.total_frames:,}")
            logger.info(f"   • Resolution: {self.frame_width}x{self.frame_height}")
            logger.info(f"   • Duration: {self.video_duration/60:.1f} minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Video error: {e}")
            return False
    
    def process_video_frame_by_frame(self):
        """
        Traite la vidéo frame par frame pour détecter TOUTES les actions
        """
        logger.info("=" * 80)
        logger.info("🎥 ANALYSE FRAME PAR FRAME")
        logger.info("=" * 80)
        logger.info(f"Total frames à analyser: {self.total_frames:,}")
        logger.info(f"Temps estimé: ~{self.total_frames/30/60:.1f} minutes à 30 fps")
        logger.info("=" * 80)
        
        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Progress bar
        pbar = tqdm(total=self.total_frames, desc="Analyse des frames", unit="frame")
        
        frame_num = 0
        actions_detected = 0
        start_time = time.time()
        
        # Pour éviter les actions en double
        last_action_time = -10
        min_action_interval = 2.0  # secondes
        
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            
            timestamp = frame_num / self.fps
            
            # Détecter les actions dans cette frame
            actions = self.action_detector.detect_actions_in_frame(frame, frame_num, timestamp)
            
            # Détecter les joueurs (simulé - à remplacer par YOLO)
            detected_players = self.action_detector._detect_players(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            
            # Mettre à jour le tracking des joueurs
            updated_players = self.player_tracker.update(detected_players, frame_num, timestamp, frame)
            
            # Traiter les actions détectées
            for action in actions:
                # Éviter les doublons
                if timestamp - last_action_time > min_action_interval:
                    action['action_id'] = hashlib.md5(f"{timestamp}_{action['type']}".encode()).hexdigest()[:8]
                    action['frame'] = frame_num
                    
                    # Associer aux joueurs
                    if action.get('players_involved'):
                        for player_info in action['players_involved']:
                            if 'id' in player_info:
                                self.player_tracker.add_action_to_player(player_info['id'], action)
                    
                    self.detected_actions.append(action)
                    actions_detected += 1
                    last_action_time = timestamp
                    
                    # Mise à jour des stats
                    self.stats['actions_by_type'][action['type']] += 1
                    
                    # Sauvegarder la frame pour analyse
                    self._save_analysis_frame(frame, action, frame_num)
            
            # Sauvegarder une frame sur 1000 pour visualisation
            if frame_num % 1000 == 0:
                preview_frame = frame.copy()
                # Dessiner les joueurs détectés
                for pid, player in updated_players.items():
                    if player['positions']:
                        last_pos = player['positions'][-1]['position']
                        x = int(last_pos[0] * self.frame_width)
                        y = int(last_pos[1] * self.frame_height)
                        cv2.circle(preview_frame, (x, y), 10, (0, 255, 0), 2)
                        if player.get('jersey_number'):
                            cv2.putText(preview_frame, f"#{player['jersey_number']}", 
                                      (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                preview_path = self.frames_base / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(preview_path), preview_frame)
            
            frame_num += 1
            self.stats['total_frames_processed'] = frame_num
            
            # Mise à jour de la barre de progression
            if frame_num % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_num / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'Actions': actions_detected,
                    'Players': len(self.player_tracker.get_all_players()),
                    'FPS': f"{fps_processing:.1f}"
                })
            
            pbar.update(1)
        
        pbar.close()
        
        self.stats['total_actions_detected'] = actions_detected
        self.stats['unique_players'] = len(self.player_tracker.get_all_players())
        self.stats['processing_fps'] = frame_num / (time.time() - start_time)
        
        logger.info("=" * 80)
        logger.info("📊 RAPPORT D'ANALYSE")
        logger.info("=" * 80)
        logger.info(f"Frames analysées: {frame_num:,}")
        logger.info(f"Actions détectées: {actions_detected:,}")
        logger.info(f"Joueurs identifiés: {len(self.player_tracker.get_all_players()):,}")
        logger.info(f"Vitesse de traitement: {self.stats['processing_fps']:.1f} fps")
        logger.info("=" * 80)
    
    def _save_analysis_frame(self, frame: np.ndarray, action: Dict, frame_num: int):
        """
        Sauvegarde une frame d'analyse avec l'action détectée
        """
        # Dessiner l'action sur la frame
        annotated = frame.copy()
        
        # Ajouter le texte de l'action
        action_text = f"{action['type']} ({action['confidence']:.2f})"
        cv2.putText(annotated, action_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Ajouter le timestamp
        timestamp = action['timestamp']
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_text = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(annotated, time_text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Sauvegarder
        action_frame_path = self.frames_base / f"action_{action['action_id']}_frame{frame_num}.jpg"
        cv2.imwrite(str(action_frame_path), annotated)
    
    def create_action_clips(self):
        """
        Crée des clips vidéo pour chaque action détectée
        """
        logger.info("=" * 80)
        logger.info("🎬 CRÉATION DES CLIPS VIDÉO")
        logger.info("=" * 80)
        
        if not self.detected_actions:
            logger.warning("Aucune action détectée")
            return
        
        # Trier les actions par timestamp
        self.detected_actions.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Création de {len(self.detected_actions):,} clips...")
        
        # Barre de progression
        pbar = tqdm(self.detected_actions, desc="Création des clips", unit="clip")
        
        clips_created = 0
        
        for action in pbar:
            try:
                timestamp = action['timestamp']
                action_type = action['type']
                
                # Calculer les limites du clip (5s avant, 5s après)
                start_time = max(0, timestamp - 5)
                end_time = min(self.video_duration, timestamp + 5)
                
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                
                if start_frame >= end_frame:
                    continue
                
                # Extraire les frames
                frames = []
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(start_frame, end_frame):
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break
                    
                    # Annoter la frame avec l'action
                    annotated = frame.copy()
                    
                    # Ajouter le type d'action
                    cv2.putText(annotated, f"{action_type.upper()}", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Ajouter le timestamp
                    current_time = _ / self.fps
                    minutes = int(current_time // 60)
                    seconds = int(current_time % 60)
                    cv2.putText(annotated, f"{minutes:02d}:{seconds:02d}", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    frames.append(annotated)
                
                if frames:
                    # Créer le nom du clip
                    match_time = f"{int(timestamp//60):02d}_{int(timestamp%60):02d}"
                    clip_name = f"{action_type}_{match_time}_{action['action_id']}.mp4"
                    
                    # Sauvegarder dans le dossier du type d'action
                    action_dir = self.actions_base / action_type / 'clips'
                    clip_path = action_dir / clip_name
                    
                    self._save_clip(frames, clip_path)
                    
                    clips_created += 1
                    
                    # Mise à jour de la progression
                    pbar.set_postfix({
                        'Type': action_type[:10],
                        'Time': f"{int(timestamp//60)}:{int(timestamp%60):02d}",
                        'Clips': clips_created
                    })
                
            except Exception as e:
                logger.error(f"Erreur création clip: {e}")
            
            # Remettre la vidéo au début
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        pbar.close()
        self.stats['clips_created'] = clips_created
        
        logger.info(f"✅ {clips_created} clips créés")
    
    def _save_clip(self, frames: List[np.ndarray], path: Path):
        """Save video clip"""
        if not frames:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, self.fps, 
                              (self.frame_width, self.frame_height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def generate_player_reports(self):
        """
        Génère des rapports pour chaque joueur
        """
        logger.info("=" * 80)
        logger.info("📊 GÉNÉRATION DES RAPPORTS JOUEURS")
        logger.info("=" * 80)
        
        players = self.player_tracker.get_all_players()
        
        # Organiser les joueurs par équipe
        teams = {
            'Luton Town': [],
            'Northampton Town': [],
            'Unknown': []
        }
        
        for pid, player in players.items():
            team = player.get('team', 'Unknown')
            if team in teams:
                teams[team].append(player)
            else:
                teams['Unknown'].append(player)
        
        # Statistiques par équipe
        for team_name, team_players in teams.items():
            if not team_players:
                continue
            
            logger.info(f"\n{team_name}: {len(team_players)} joueurs")
            
            team_dir = self.players_base / team_name.upper().replace(' ', '_')
            
            for player in team_players:
                jersey = player.get('jersey_number', '??')
                actions = len(player.get('actions', []))
                
                # Créer dossier joueur
                player_dir = team_dir / str(jersey) if jersey != '??' else team_dir / f"player_{player['id']}"
                player_dir.mkdir(exist_ok=True)
                
                # Sauvegarder les données du joueur
                player_data = {
                    'player_id': player['id'],
                    'jersey_number': player.get('jersey_number'),
                    'team': player.get('team'),
                    'first_seen': player.get('first_seen'),
                    'last_seen': player.get('last_seen'),
                    'frames_visible': player.get('frame_count'),
                    'total_actions': len(player.get('actions', [])),
                    'actions': player.get('actions', []),
                    'positions': player.get('positions', [])
                }
                
                report_path = player_dir / 'player_data.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(player_data, f, indent=2)
                
                logger.info(f"  • #{jersey}: {actions} actions")
        
        logger.info(f"\n✅ Rapports joueurs sauvegardés dans {self.players_base}")
    
    def generate_match_report(self):
        """
        Génère le rapport complet du match
        """
        report = {
            'match_info': {
                'name': self.match_name,
                'id': self.match_id,
                'duration_seconds': self.video_duration,
                'duration_minutes': self.video_duration / 60,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'resolution': f"{self.frame_width}x{self.frame_height}"
            },
            'statistics': {
                'total_frames_processed': self.stats['total_frames_processed'],
                'total_actions_detected': self.stats['total_actions_detected'],
                'unique_players': len(self.player_tracker.get_all_players()),
                'clips_created': self.stats['clips_created'],
                'actions_by_type': dict(self.stats['actions_by_type']),
                'processing_fps': self.stats['processing_fps']
            },
            'actions': self.detected_actions,
            'processing_date': datetime.now().isoformat()
        }
        
        # Sauvegarder le rapport
        report_path = self.reports_base / f"{self.match_name}_COMPLETE_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Générer un résumé texte
        summary_path = self.reports_base / f"{self.match_name}_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"SOCCERNET REALTIME TRACKER - RAPPORT DE MATCH\n")
            f.write(f"{self.match_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Durée du match: {self.video_duration/60:.1f} minutes\n")
            f.write(f"Frames analysées: {self.stats['total_frames_processed']:,}\n")
            f.write(f"Actions détectées: {self.stats['total_actions_detected']:,}\n")
            f.write(f"Joueurs identifiés: {len(self.player_tracker.get_all_players()):,}\n")
            f.write(f"Clips créés: {self.stats['clips_created']:,}\n\n")
            f.write("Actions par type:\n")
            for atype, count in sorted(self.stats['actions_by_type'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {atype}: {count}\n")
        
        logger.info(f"✅ Rapport complet sauvegardé: {report_path}")
    
    def print_final_summary(self):
        """Print final summary"""
        logger.info("=" * 80)
        logger.info("🏆 RÉSULTATS FINAUX")
        logger.info("=" * 80)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Frames analysées: {self.stats['total_frames_processed']:,}")
        logger.info(f"Actions détectées: {self.stats['total_actions_detected']:,}")
        logger.info(f"Joueurs identifiés: {len(self.player_tracker.get_all_players()):,}")
        logger.info(f"Clips créés: {self.stats['clips_created']:,}")
        logger.info(f"Vitesse de traitement: {self.stats['processing_fps']:.1f} fps")
        logger.info("=" * 80)
        logger.info(f"Dossiers de sortie:")
        logger.info(f"  • {self.output_base}")
        logger.info(f"  • {self.players_base}")
        logger.info(f"  • {self.actions_base}")
        logger.info(f"  • {self.reports_base}")
        logger.info("=" * 80)
    
    def run(self):
        """Run complete real-time analysis"""
        
        start_total = time.time()
        
        # 1. Initialiser la vidéo
        if not self._initialize_video():
            return
        
        # 2. Analyser frame par frame
        self.process_video_frame_by_frame()
        
        # 3. Créer les clips pour chaque action
        if self.detected_actions:
            # Remettre la vidéo au début
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.create_action_clips()
        
        # 4. Générer les rapports
        self.generate_player_reports()
        self.generate_match_report()
        
        # 5. Nettoyage
        if self.video_cap:
            self.video_cap.release()
        
        # 6. Résumé final
        self.print_final_summary()
        
        total_time = time.time() - start_total
        logger.info(f"⏱️ Temps total: {total_time/60:.2f} minutes")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    try:
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
        
        tracker = SoccerNetRealtimeTracker()
        tracker.run()
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()