#!/usr/bin/env python3
"""
briqx_processor.py - OPTIMIZED Processor for Luton Town vs Northampton Town
Detects ALL actions from the match using SoccerNet-v3 data
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import glob
import re

# Import SoccerNet modules
try:
    from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY, INVERSE_FRAME_CLASS_DICTIONARY
except ImportError:
    logging.warning("SoccerNet module not found, using basic mode")
    FRAME_CLASS_DICTIONARY = {}
    INVERSE_FRAME_CLASS_DICTIONARY = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('briqx_optimized_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlayerInfo:
    """Information about a player"""
    def __init__(self, jersey_number: int, team: str, player_id: str = None):
        self.jersey_number = jersey_number
        self.team = team
        self.player_id = player_id or f"{team}_{jersey_number}"
        self.team_color = self._get_team_color(team)
        self.actions = []
        self.bboxes = []
        self.timestamps = []
        self.events = []  # All events involving this player
    
    def _get_team_color(self, team: str) -> str:
        """Determine team color based on team name"""
        team_lower = team.lower()
        if 'luton' in team_lower:
            return 'ORANGE'
        elif 'northampton' in team_lower:
            return 'MAROON'
        elif 'sivert' in team_lower or 'red' in team_lower:
            return 'RED'
        elif 'celtic' in team_lower or 'green' in team_lower:
            return 'GREEN'
        else:
            return 'BLUE'
    
    def add_action(self, action_data: Dict):
        """Add an action for this player"""
        self.actions.append(action_data)
        self.events.append({
            'type': 'action',
            'data': action_data,
            'timestamp': action_data.get('game_time', 0)
        })
    
    def add_bbox(self, bbox: List[float], timestamp: float, frame_num: int, confidence: float = 1.0):
        """Add a bounding box for this player"""
        bbox_data = {
            'frame': frame_num,
            'timestamp': timestamp,
            'bbox': bbox,
            'center_x': (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0,
            'center_y': (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0,
            'width': bbox[2] - bbox[0] if len(bbox) >= 4 else 0,
            'height': bbox[3] - bbox[1] if len(bbox) >= 4 else 0,
            'confidence': confidence
        }
        self.bboxes.append(bbox_data)
        self.timestamps.append(timestamp)

class ActionDetector:
    """Detects actions from video frames using computer vision"""
    
    def __init__(self, fps: float, frame_width: int, frame_height: int):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.action_types = [
            'shot', 'goal', 'pass', 'tackle', 'interception',
            'foul', 'corner', 'free_kick', 'penalty', 'offside',
            'save', 'clearance', 'cross', 'dribble', 'throw_in'
        ]
        
        # Action detection thresholds
        self.thresholds = {
            'shot': 0.7,
            'goal': 0.9,
            'pass': 0.6,
            'tackle': 0.65,
            'interception': 0.6,
            'foul': 0.7,
            'corner': 0.8,
            'free_kick': 0.75,
            'penalty': 0.85,
            'offside': 0.7,
            'save': 0.8,
            'clearance': 0.6,
            'cross': 0.65,
            'dribble': 0.6,
            'throw_in': 0.8
        }
    
    def detect_actions_in_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> List[Dict]:
        """
        Detect actions in a single frame
        This is where you would integrate your ML model
        For now, returns empty list - replace with your model
        """
        detected_actions = []
        
        # TODO: Replace with your actual action detection model
        # Example integration point for your model:
        # model_output = your_model.predict(frame)
        # for detection in model_output:
        #     if detection.confidence > self.thresholds.get(detection.action_type, 0.5):
        #         detected_actions.append({
        #             'label': detection.action_type,
        #             'confidence': detection.confidence,
        #             'bbox': detection.bbox,
        #             'frame': frame_num,
        #             'timestamp': timestamp
        #         })
        
        return detected_actions

class SoccerNetProcessor:
    def __init__(self):
        # Path configuration
        self.base_path = Path(r"C:\Users\HP\Downloads\Soccernet-v3-main-Tracking")
        self.video_path = self.base_path / "RK_Semifinals_Luton Town - Northampton Town_04032026.mp4"
        
        # Extract match name
        self.match_name = self._extract_match_name()
        
        # SoccerNet data paths
        self.soccernet_data_path = Path("C:/SoccerNetData")
        self.labels_base_path = self.soccernet_data_path
        
        # Output folders
        self.spotting_data_path = self.base_path / "SpottingData"
        self.output_base = self.base_path / "Analyse_Clips" / self.match_name
        
        # Create all directories
        self._create_all_directories()
        
        # Players dictionary
        self.players: Dict[str, PlayerInfo] = {}
        
        # Video properties
        self.video_cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        self.video_duration = None
        
        # Find match-specific label files
        self.match_labels = self._find_match_labels()
        
        # Action detector
        self.action_detector = None
        
        # Processing statistics
        self.stats = {
            'total_frames': 0,
            'frames_processed': 0,
            'actions_detected': 0,
            'players_detected': 0,
            'clips_created': 0,
            'processing_time': 0,
            'actions_by_type': {},
            'actions_by_team': {},
            'actions_by_player': {}
        }
        
        logger.info("=" * 80)
        logger.info("SOCCERNET-V3 OPTIMIZED PROCESSOR - LUTON TOWN VS NORTHAMPTON TOWN")
        logger.info("=" * 80)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Duration: {self.video_duration if self.video_duration else 'Unknown'} minutes")
        logger.info(f"Output folder: {self.output_base}")
        logger.info("=" * 80)
    
    def _extract_match_name(self) -> str:
        """Extract match name from video filename"""
        filename = self.video_path.stem
        # Clean up filename
        filename = filename.replace('RK_Semifinals_', '')
        filename = filename.replace('_04032026', '')
        return filename.strip()
    
    def _create_all_directories(self):
        """Create the entire folder structure"""
        directories = [
            self.spotting_data_path,
            self.output_base,
            self.output_base / "LUTON_TOWN",
            self.output_base / "NORTHAMPTON_TOWN",
            self.output_base / "OTHER",
            self.output_base / "ALL_ACTIONS",
            self.spotting_data_path / "actions",
            self.spotting_data_path / "tracking",
            self.spotting_data_path / "players",
            self.spotting_data_path / "match_analysis"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ All folders created for match: {self.match_name}")
    
    def _find_match_labels(self) -> List[Path]:
        """Find ALL label files that might contain match data"""
        label_files = []
        
        # Search patterns for match-specific labels
        patterns = [
            # Luton Town matches
            self.labels_base_path / "**/*Luton*/**/Labels-v3.json",
            self.labels_base_path / "**/*LUTON*/**/Labels-v3.json",
            self.labels_base_path / "**/*luton*/**/Labels-v3.json",
            # Northampton matches
            self.labels_base_path / "**/*Northampton*/**/Labels-v3.json",
            self.labels_base_path / "**/*NORTHAMPTON*/**/Labels-v3.json",
            self.labels_base_path / "**/*northampton*/**/Labels-v3.json",
            # Championship/League matches
            self.labels_base_path / "**/championship*/**/Labels-v3.json",
            self.labels_base_path / "**/league*/**/Labels-v3.json",
            self.labels_base_path / "**/england*/**/Labels-v3.json",
            # All label files (fallback)
            self.labels_base_path / "**/Labels-v3.json"
        ]
        
        for pattern in patterns:
            found = glob.glob(str(pattern), recursive=True)
            label_files.extend([Path(f) for f in found])
        
        # Deduplicate
        label_files = list(set(label_files))
        
        # Filter by date if possible (2024-2026 matches)
        filtered_files = []
        for file in label_files:
            file_str = str(file).lower()
            # Prioritize recent matches
            if any(year in file_str for year in ['2024', '2025', '2026']):
                filtered_files.append(file)
            elif any(team in file_str for team in ['luton', 'northampton']):
                filtered_files.append(file)
        
        if filtered_files:
            return filtered_files
        return label_files
    
    def _initialize_video(self) -> bool:
        """Initialize video capture"""
        try:
            logger.info("Initializing video...")
            
            if not self.video_path.exists():
                logger.error(f"Video file not found: {self.video_path}")
                return False
                
            self.video_cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.video_cap.isOpened():
                logger.error("Unable to open video")
                return False
            
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_duration = self.total_frames / self.fps
            
            # Initialize action detector
            self.action_detector = ActionDetector(self.fps, self.frame_width, self.frame_height)
            
            logger.info(f"✓ Video initialized successfully:")
            logger.info(f"  - FPS: {self.fps:.2f}")
            logger.info(f"  - Frames: {self.total_frames:,}")
            logger.info(f"  - Resolution: {self.frame_width}x{self.frame_height}")
            logger.info(f"  - Duration: {self.video_duration:.2f} seconds ({self.video_duration/60:.2f} minutes)")
            
            self.stats['total_frames'] = self.total_frames
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            return False
    
    def load_all_match_actions(self):
        """Load ALL actions from SoccerNet-v3 labels"""
        
        logger.info("=" * 80)
        logger.info("LOADING SOCCERNET-V3 MATCH DATA")
        logger.info("=" * 80)
        
        if not self.match_labels:
            logger.warning("No label files found. Will use frame-by-frame detection.")
            return
        
        total_actions_loaded = 0
        unique_players = set()
        
        # Process each label file
        for labels_file in tqdm(self.match_labels, desc="Loading match data"):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract match metadata
                game_metadata = data.get('GameMetadata', {})
                
                # Process all actions
                if 'actions' in data:
                    for action_name, action_data in data['actions'].items():
                        # Process each action instance
                        game_time = action_data.get('game_time', 0)
                        
                        # Only include actions within match duration
                        if game_time > self.video_duration:
                            continue
                        
                        # Process bounding boxes
                        bboxes = action_data.get('bboxes', [])
                        for bbox in bboxes:
                            jersey = bbox.get('jersey_number')
                            team = bbox.get('team', 'Unknown')
                            
                            # Try to identify team from context
                            if team == 'Unknown' or not team:
                                if 'luton' in str(labels_file).lower():
                                    team = 'Luton Town'
                                elif 'northampton' in str(labels_file).lower():
                                    team = 'Northampton Town'
                            
                            if jersey is not None:
                                player_key = f"{team}_{jersey}"
                                
                                if player_key not in self.players:
                                    self.players[player_key] = PlayerInfo(jersey, team)
                                    unique_players.add(player_key)
                                
                                # Create action data
                                action_info = {
                                    'label': action_name,
                                    'game_time': game_time,
                                    'half': action_data.get('half', 1),
                                    'period': action_data.get('period', '1H'),
                                    'position': action_data.get('position', []),
                                    'bbox': bbox.get('bbox', []),
                                    'confidence': 1.0,
                                    'source': str(labels_file),
                                    'match_time': self._format_match_time(game_time)
                                }
                                
                                self.players[player_key].add_action(action_info)
                                total_actions_loaded += 1
                                
                                # Update statistics
                                self.stats['actions_by_type'][action_name] = self.stats['actions_by_type'].get(action_name, 0) + 1
                                self.stats['actions_by_team'][team] = self.stats['actions_by_team'].get(team, 0) + 1
                                self.stats['actions_by_player'][player_key] = self.stats['actions_by_player'].get(player_key, 0) + 1
                
                # Process replays
                if 'replays' in data:
                    for replay_name, replay_data in data['replays'].items():
                        game_time = replay_data.get('game_time', 0)
                        
                        if game_time > self.video_duration:
                            continue
                        
                        bboxes = replay_data.get('bboxes', [])
                        for bbox in bboxes:
                            jersey = bbox.get('jersey_number')
                            team = bbox.get('team', 'Unknown')
                            
                            if jersey is not None:
                                player_key = f"{team}_{jersey}"
                                
                                if player_key not in self.players:
                                    self.players[player_key] = PlayerInfo(jersey, team)
                                    unique_players.add(player_key)
                                
                                action_info = {
                                    'label': f"{replay_name}_replay",
                                    'game_time': game_time,
                                    'half': replay_data.get('half', 1),
                                    'period': replay_data.get('period', '1H'),
                                    'position': replay_data.get('position', []),
                                    'bbox': bbox.get('bbox', []),
                                    'confidence': 1.0,
                                    'source': str(labels_file),
                                    'is_replay': True,
                                    'match_time': self._format_match_time(game_time)
                                }
                                
                                self.players[player_key].add_action(action_info)
                                total_actions_loaded += 1
                                
            except Exception as e:
                logger.debug(f"Error processing {labels_file}: {e}")
                continue
        
        self.stats['actions_detected'] = total_actions_loaded
        self.stats['players_detected'] = len(self.players)
        
        logger.info("=" * 80)
        logger.info("MATCH DATA LOADING REPORT")
        logger.info("=" * 80)
        logger.info(f"Label files processed: {len(self.match_labels)}")
        logger.info(f"Total actions loaded: {total_actions_loaded:,}")
        logger.info(f"Unique players detected: {len(self.players):,}")
        
        # Display actions by type
        if self.stats['actions_by_type']:
            logger.info("\nActions by type:")
            for action_type, count in sorted(self.stats['actions_by_type'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {action_type}: {count}")
        
        # Display teams
        logger.info("\nTeams:")
        for team, count in self.stats['actions_by_team'].items():
            team_players = [p for p in self.players.values() if p.team == team]
            logger.info(f"  • {team}: {count} actions, {len(team_players)} players")
        
        logger.info("=" * 80)
    
    def _format_match_time(self, seconds: float) -> str:
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds_remainder = int(seconds % 60)
        return f"{minutes:02d}:{seconds_remainder:02d}"
    
    def scan_video_for_actions(self):
        """Scan the entire video frame by frame to detect ALL actions"""
        
        logger.info("=" * 80)
        logger.info("SCANNING VIDEO FOR ALL ACTIONS - FRAME BY FRAME")
        logger.info("=" * 80)
        logger.info(f"Total frames to analyze: {self.total_frames:,}")
        logger.info(f"This will take approximately: {self.total_frames/30/60:.1f} minutes at 30 fps")
        logger.info("=" * 80)
        
        if not self.video_cap:
            return
        
        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_actions = []
        frame_num = 0
        detection_interval = max(1, int(self.fps))  # Check every second
        
        # Progress bar for frame scanning
        pbar = tqdm(total=self.total_frames, desc="Scanning frames", unit="frame")
        
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            
            timestamp = frame_num / self.fps
            
            # Detect actions at regular intervals
            if frame_num % detection_interval == 0:
                detected = self.action_detector.detect_actions_in_frame(frame, frame_num, timestamp)
                
                for action in detected:
                    action['detection_method'] = 'frame_scan'
                    frame_actions.append(action)
                    
                    # Try to associate with player
                    if 'bbox' in action:
                        # Here you would associate with player based on bbox position
                        pass
            
            frame_num += 1
            pbar.update(1)
            
            # Update progress info
            if frame_num % 1000 == 0:
                pbar.set_postfix({
                    'Actions': len(frame_actions),
                    'Time': self._format_match_time(timestamp)
                })
        
        pbar.close()
        
        logger.info(f"✓ Frame scanning complete: {len(frame_actions)} potential actions detected")
        
        # Merge with existing actions
        if frame_actions:
            self._merge_detected_actions(frame_actions)
        
        return frame_actions
    
    def _merge_detected_actions(self, new_actions: List[Dict]):
        """Merge newly detected actions with existing ones"""
        
        # Group actions by timestamp (within 2 seconds)
        time_threshold = 2.0
        merged_count = 0
        
        for new_action in new_actions:
            timestamp = new_action['timestamp']
            found_match = False
            
            # Check if similar action already exists
            for player in self.players.values():
                for existing_action in player.actions:
                    if abs(existing_action['game_time'] - timestamp) < time_threshold:
                        if existing_action['label'] == new_action['label']:
                            found_match = True
                            break
            
            if not found_match:
                # Create temporary player if needed
                player_key = f"Unknown_{len(self.players)+1}"
                if player_key not in self.players:
                    self.players[player_key] = PlayerInfo(0, "Unknown")
                
                self.players[player_key].add_action({
                    'label': new_action['label'],
                    'game_time': timestamp,
                    'half': 1 if timestamp < 2700 else 2,  # 45 minutes = 2700 seconds
                    'period': '1H' if timestamp < 2700 else '2H',
                    'confidence': new_action.get('confidence', 0.5),
                    'detection_method': 'frame_scan',
                    'match_time': self._format_match_time(timestamp)
                })
                merged_count += 1
        
        if merged_count > 0:
            logger.info(f"✓ Added {merged_count} new actions from frame scanning")
            self.stats['actions_detected'] += merged_count
    
    def create_clips_for_all_actions(self):
        """Create video clips for ALL detected actions"""
        
        logger.info("=" * 80)
        logger.info("CREATING VIDEO CLIPS FOR ALL ACTIONS")
        logger.info("=" * 80)
        
        # Collect all actions from all players
        all_actions = []
        for player_key, player in self.players.items():
            for action in player.actions:
                all_actions.append({
                    'player': player,
                    'player_key': player_key,
                    'action': action
                })
        
        # Sort by timestamp
        all_actions.sort(key=lambda x: x['action']['game_time'])
        
        logger.info(f"Total actions to process: {len(all_actions):,}")
        
        if not all_actions:
            logger.warning("No actions to process")
            return
        
        # Remove duplicates (actions within 3 seconds of each other)
        unique_actions = []
        last_time = -10
        
        for item in all_actions:
            current_time = item['action']['game_time']
            if current_time - last_time > 3.0:  # Minimum 3 seconds between clips
                unique_actions.append(item)
                last_time = current_time
        
        logger.info(f"Unique actions after deduplication: {len(unique_actions):,}")
        
        # Progress bar for clip creation
        pbar = tqdm(unique_actions, desc="Creating clips", unit="action")
        
        clips_created = 0
        errors = 0
        
        for item in pbar:
            try:
                player = item['player']
                action = item['action']
                
                # Determine team folder
                if 'luton' in player.team.lower():
                    team_folder = "LUTON_TOWN"
                elif 'northampton' in player.team.lower():
                    team_folder = "NORTHAMPTON_TOWN"
                else:
                    team_folder = "OTHER"
                
                # Create player folder
                player_dir = self.output_base / team_folder / str(player.jersey_number)
                clips_dir = player_dir / "clips"
                clips_dir.mkdir(parents=True, exist_ok=True)
                
                # Also save to all_actions folder
                all_actions_dir = self.output_base / "ALL_ACTIONS"
                all_actions_dir.mkdir(exist_ok=True)
                
                # Extract clip (5 seconds before, 5 seconds after)
                game_time = action['game_time']
                start_time = max(0, game_time - 5)
                end_time = min(self.video_duration, game_time + 5)
                
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                
                if start_frame >= end_frame:
                    continue
                
                # Extract frames
                frames = []
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(start_frame, end_frame):
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                if frames:
                    # Create clip filename
                    match_time = self._format_match_time(game_time)
                    safe_label = action['label'].replace(' ', '_').replace('/', '-')
                    clip_name = f"{safe_label}_{match_time.replace(':', '-')}_{int(game_time)}s.mp4"
                    
                    # Save in player folder
                    clip_path = clips_dir / clip_name
                    self._save_clip(frames, clip_path)
                    
                    # Save in all actions folder (with player info)
                    all_actions_clip = all_actions_dir / f"{player.team}_{player.jersey_number}_{clip_name}"
                    self._save_clip(frames, all_actions_clip)
                    
                    # Save action metadata
                    action_file = player_dir / "tracking" / f"action_{int(game_time)}s.json"
                    action_file.parent.mkdir(exist_ok=True)
                    
                    action_data = {
                        **action,
                        'player': {
                            'jersey': player.jersey_number,
                            'team': player.team,
                            'team_color': player.team_color
                        },
                        'clip_info': {
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': 10,
                            'path': str(clip_path)
                        }
                    }
                    
                    with open(action_file, 'w', encoding='utf-8') as f:
                        json.dump(action_data, f, indent=2, ensure_ascii=False)
                    
                    clips_created += 1
                    
                    # Update progress
                    pbar.set_postfix({
                        'Team': player.team[:10],
                        '#': player.jersey_number,
                        'Time': match_time,
                        'Action': safe_label[:15]
                    })
            
            except Exception as e:
                logger.error(f"Error creating clip: {e}")
                errors += 1
            
            # Reset video position for next clip
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        pbar.close()
        
        self.stats['clips_created'] = clips_created
        
        logger.info("=" * 80)
        logger.info("CLIP CREATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Actions processed: {len(unique_actions):,}")
        logger.info(f"Clips created: {clips_created:,}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Success rate: {clips_created/len(unique_actions)*100:.1f}%")
        logger.info("=" * 80)
    
    def _save_clip(self, frames: List[np.ndarray], output_path: Path):
        """Save a video clip"""
        if not frames:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all actions"""
        
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 80)
        
        report = {
            'match_info': {
                'name': self.match_name,
                'video_path': str(self.video_path),
                'duration_seconds': self.video_duration,
                'duration_minutes': self.video_duration / 60,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'resolution': f"{self.frame_width}x{self.frame_height}"
            },
            'statistics': self.stats,
            'players': {},
            'actions_timeline': [],
            'teams_summary': {},
            'processing_date': datetime.now().isoformat()
        }
        
        # Player details
        for player_key, player in self.players.items():
            report['players'][player_key] = {
                'jersey_number': player.jersey_number,
                'team': player.team,
                'team_color': player.team_color,
                'total_actions': len(player.actions),
                'actions': [
                    {
                        'label': a['label'],
                        'game_time': a['game_time'],
                        'match_time': self._format_match_time(a['game_time']),
                        'half': a.get('half', 1)
                    }
                    for a in player.actions
                ]
            }
            
            # Add to timeline
            for action in player.actions:
                report['actions_timeline'].append({
                    'timestamp': action['game_time'],
                    'match_time': self._format_match_time(action['game_time']),
                    'player': player_key,
                    'jersey': player.jersey_number,
                    'team': player.team,
                    'action_type': action['label']
                })
        
        # Sort timeline
        report['actions_timeline'].sort(key=lambda x: x['timestamp'])
        
        # Teams summary
        for player_key, player in self.players.items():
            if player.team not in report['teams_summary']:
                report['teams_summary'][player.team] = {
                    'total_actions': 0,
                    'players': set(),
                    'actions_by_type': {}
                }
            
            report['teams_summary'][player.team]['total_actions'] += len(player.actions)
            report['teams_summary'][player.team]['players'].add(player.jersey_number)
            
            for action in player.actions:
                action_type = action['label']
                report['teams_summary'][player.team]['actions_by_type'][action_type] = \
                    report['teams_summary'][player.team]['actions_by_type'].get(action_type, 0) + 1
        
        # Convert sets to lists for JSON
        for team in report['teams_summary']:
            report['teams_summary'][team]['players'] = list(report['teams_summary'][team]['players'])
        
        # Save reports
        report_file = self.output_base / 'complete_match_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save in SpottingData
        spotting_report = self.spotting_data_path / 'match_analysis' / f'{self.match_name}_complete_report.json'
        with open(spotting_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate readable summary
        summary_file = self.output_base / 'match_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MATCH SUMMARY: {self.match_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Duration: {self.video_duration/60:.1f} minutes\n")
            f.write(f"Total Actions: {self.stats['actions_detected']:,}\n")
            f.write(f"Clips Created: {self.stats['clips_created']:,}\n\n")
            
            f.write("ACTIONS BY TYPE:\n")
            for action_type, count in sorted(self.stats['actions_by_type'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {action_type}: {count}\n")
            
            f.write("\nACTIONS BY TEAM:\n")
            for team, count in self.stats['actions_by_team'].items():
                f.write(f"  {team}: {count}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"✓ Complete report saved: {report_file}")
        logger.info(f"✓ Summary saved: {summary_file}")
        
        return report
    
    def print_final_statistics(self):
        """Print final processing statistics"""
        
        logger.info("=" * 80)
        logger.info("FINAL PROCESSING STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Total Frames: {self.stats['total_frames']:,}")
        logger.info(f"Players Detected: {self.stats['players_detected']:,}")
        logger.info(f"Total Actions: {self.stats['actions_detected']:,}")
        logger.info(f"Clips Created: {self.stats['clips_created']:,}")
        
        if self.stats['actions_by_type']:
            logger.info("\nTop 10 Action Types:")
            top_actions = sorted(self.stats['actions_by_type'].items(), key=lambda x: x[1], reverse=True)[:10]
            for action_type, count in top_actions:
                logger.info(f"  • {action_type}: {count}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Output folder: {self.output_base}")
        logger.info("=" * 80)
    
    def process(self):
        """Execute the complete optimized processing"""
        
        import time
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"STARTING OPTIMIZED PROCESSING FOR: {self.match_name}")
        logger.info("=" * 80)
        
        # 1. Initialize video
        if not self._initialize_video():
            return
        
        # 2. Load ALL match actions from SoccerNet labels
        self.load_all_match_actions()
        
        # 3. Scan video for additional actions (if needed)
        if self.stats['actions_detected'] < 100:  # If few actions found, scan video
            logger.info("Few actions found in labels. Scanning video directly...")
            self.scan_video_for_actions()
        
        # 4. Create clips for all actions
        if self.stats['actions_detected'] > 0:
            # Reset video to beginning
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.create_clips_for_all_actions()
        else:
            logger.warning("No actions detected. Cannot create clips.")
        
        # 5. Generate comprehensive report
        self.generate_comprehensive_report()
        
        # 6. Cleanup
        if self.video_cap:
            self.video_cap.release()
        
        # 7. Print final statistics
        self.print_final_statistics()
        
        elapsed_time = time.time() - start_time
        self.stats['processing_time'] = elapsed_time
        
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Average processing speed: {self.stats['total_frames']/elapsed_time:.1f} frames/second")
        logger.info("=" * 80)

def main():
    """Main entry point"""
    try:
        # Configuration for Windows
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
        
        processor = SoccerNetProcessor()
        processor.process()
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()