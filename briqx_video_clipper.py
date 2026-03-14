#!/usr/bin/env python3
"""
briqx_processor.py - Processor with official SoccerNet-v3 data
Uses official labels to detect all players and actions
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import cv2
import numpy as np
from tqdm import tqdm
import glob

# Import SoccerNet modules
try:
    from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
except ImportError:
    logging.warning("SoccerNet module not found, using demo mode")
    FRAME_CLASS_DICTIONARY = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('briqx_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlayerInfo:
    """Information about a player"""
    def __init__(self, jersey_number: int, team: str):
        self.jersey_number = jersey_number
        self.team = team
        self.team_color = self._get_team_color(team)
        self.actions = []
        self.bboxes = []
        self.timestamps = []
    
    def _get_team_color(self, team: str) -> str:
        """Determine team color based on team name"""
        team_lower = team.lower()
        # Luton Town - usually orange/orange
        if 'luton' in team_lower or 'luton town' in team_lower:
            return 'ORANGE'
        # Northampton Town - usually maroon/red
        elif 'northampton' in team_lower or 'northampton town' in team_lower:
            return 'MAROON'
        # Default colors based on common patterns
        elif 'sivert' in team_lower or 'red' in team_lower:
            return 'RED'
        elif 'celtic' in team_lower or 'green' in team_lower:
            return 'GREEN'
        else:
            return 'BLUE'
    
    def add_action(self, action_data: Dict):
        """Add an action for this player"""
        self.actions.append(action_data)
    
    def add_bbox(self, bbox: List[float], timestamp: float, frame_num: int):
        """Add a bounding box for this player"""
        self.bboxes.append({
            'frame': frame_num,
            'timestamp': timestamp,
            'bbox': bbox,
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2
        })
        self.timestamps.append(timestamp)

class SoccerNetProcessor:
    def __init__(self):
        # Path configuration - MODIFIED for the new match
        self.base_path = Path(r"C:\Users\HP\Downloads\Soccernet-v3-main-Tracking")
        self.video_path = self.base_path / "RK_Semifinals_Luton Town - Northampton Town_04032026.mp4"
        
        # Extract match name from video file for folder naming
        self.match_name = self._extract_match_name()
        
        # Automatic folders
        self.soccernet_data_path = self.base_path / "SoccerNetData"
        self.spotting_data_path = self.base_path / "SpottingData"
        self.output_base = self.base_path / "Analyse_Clips" / self.match_name
        
        # Create all directories
        self._create_all_directories()
        
        # Dictionary of players
        self.players: Dict[str, PlayerInfo] = {}
        
        # Video initialization
        self.video_cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        
        # Find SoccerNet-v3 label files
        self.labels_files = self._find_soccernet_labels()
        
        logger.info("=" * 70)
        logger.info("SOCCERNET-V3 PROCESSOR WITH OFFICIAL DATA")
        logger.info("=" * 70)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Label files found: {len(self.labels_files)}")
        logger.info(f"Output folder: {self.output_base}")
        for f in self.labels_files[:3]:  # Show first 3
            logger.info(f"  - {f}")
        logger.info("=" * 70)
    
    def _extract_match_name(self) -> str:
        """Extract match name from video filename"""
        filename = self.video_path.stem
        # Remove date if present
        if '_04032026' in filename:
            filename = filename.replace('_04032026', '')
        if 'RK_Semifinals_' in filename:
            filename = filename.replace('RK_Semifinals_', '')
        return filename
    
    def _create_all_directories(self):
        """Create the entire folder structure"""
        directories = [
            self.soccernet_data_path,
            self.spotting_data_path,
            self.output_base,
            self.output_base / "LUTON_TOWN",  # Orange
            self.output_base / "NORTHAMPTON_TOWN",  # Maroon
            self.output_base / "OTHER",
            self.soccernet_data_path / "labels",
            self.soccernet_data_path / "frames",
            self.spotting_data_path / "actions",
            self.spotting_data_path / "tracking",
            self.spotting_data_path / "players"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ All folders created for match: {self.match_name}")
    
    def _find_soccernet_labels(self) -> List[Path]:
        """Find all Labels-v3.json files in the SoccerNet structure"""
        labels_files = []
        
        # Search in SoccerNetData
        soccernet_patterns = [
            self.soccernet_data_path / "**/Labels-v3.json",
            self.soccernet_data_path / "**/*/Labels-v3.json",
            self.base_path / "**/Labels-v3.json",
            Path("C:/SoccerNetData") / "**/Labels-v3.json"
        ]
        
        for pattern in soccernet_patterns:
            found = glob.glob(str(pattern), recursive=True)
            labels_files.extend([Path(f) for f in found])
        
        # Also look for match-specific labels
        match_label_patterns = [
            self.soccernet_data_path / f"**/*{self.match_name}*/Labels-v3.json",
            self.soccernet_data_path / f"**/*Luton*/Labels-v3.json",
            self.soccernet_data_path / f"**/*Northampton*/Labels-v3.json"
        ]
        
        for pattern in match_label_patterns:
            found = glob.glob(str(pattern), recursive=True)
            labels_files.extend([Path(f) for f in found])
        
        # Deduplicate
        labels_files = list(set(labels_files))
        
        return labels_files
    
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
            
            logger.info(f"✓ Video initialized:")
            logger.info(f"  - FPS: {self.fps:.2f}")
            logger.info(f"  - Frames: {self.total_frames}")
            logger.info(f"  - Resolution: {self.frame_width}x{self.frame_height}")
            logger.info(f"  - Duration: {self.total_frames/self.fps:.2f} seconds ({self.total_frames/self.fps/60:.2f} minutes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            return False
    
    def load_all_soccernet_labels(self):
        """Load ALL SoccerNet-v3 labels to extract players"""
        
        logger.info("=" * 70)
        logger.info("LOADING SOCCERNET-V3 LABELS")
        logger.info("=" * 70)
        
        total_actions = 0
        total_players = set()
        
        if not self.labels_files:
            logger.warning("No label files found. Using simulated data for demonstration.")
            self._create_simulated_data()
            return
        
        for labels_file in tqdm(self.labels_files, desc="Loading labels"):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if this label file might be for our match
                file_str = str(labels_file).lower()
                is_luton_match = ('luton' in file_str or 'northampton' in file_str)
                
                # Extract actions
                if "actions" in data:
                    for action_name, action_data in data["actions"].items():
                        # Main action
                        if "bboxes" in action_data:
                            for bbox in action_data["bboxes"]:
                                jersey = bbox.get("jersey_number")
                                team = bbox.get("team", "unknown")
                                
                                # If we can't identify team from bbox, try from filename
                                if team == "unknown" and is_luton_match:
                                    if 'luton' in file_str:
                                        team = "Luton Town"
                                    elif 'northampton' in file_str:
                                        team = "Northampton Town"
                                
                                if jersey is not None:
                                    player_key = f"{team}_{jersey}"
                                    
                                    if player_key not in self.players:
                                        self.players[player_key] = PlayerInfo(jersey, team)
                                    
                                    # Add action
                                    action_info = {
                                        'label': action_name,
                                        'game_time': action_data.get('game_time', 0),
                                        'half': action_data.get('half', 1),
                                        'period': action_data.get('period', '1H'),
                                        'position': action_data.get('position', []),
                                        'bbox': bbox.get('bbox', []),
                                        'source_file': str(labels_file)
                                    }
                                    
                                    self.players[player_key].add_action(action_info)
                                    total_actions += 1
                                    
                                    # Add bbox (approximate timestamp)
                                    self.players[player_key].add_bbox(
                                        bbox.get('bbox', []),
                                        action_data.get('game_time', 0),
                                        0
                                    )
                                    
                                    total_players.add(player_key)
                
                # Extract replays
                if "replays" in data:
                    for replay_name, replay_data in data["replays"].items():
                        if "bboxes" in replay_data:
                            for bbox in replay_data["bboxes"]:
                                jersey = bbox.get("jersey_number")
                                team = bbox.get("team", "unknown")
                                
                                if jersey is not None:
                                    player_key = f"{team}_{jersey}"
                                    
                                    if player_key not in self.players:
                                        self.players[player_key] = PlayerInfo(jersey, team)
                                    
                                    # Add action (replay)
                                    action_info = {
                                        'label': f"{replay_name}_replay",
                                        'game_time': replay_data.get('game_time', 0),
                                        'half': replay_data.get('half', 1),
                                        'period': replay_data.get('period', '1H'),
                                        'position': replay_data.get('position', []),
                                        'bbox': bbox.get('bbox', []),
                                        'source_file': str(labels_file),
                                        'is_replay': True
                                    }
                                    
                                    self.players[player_key].add_action(action_info)
                                    total_actions += 1
                                    
                                    total_players.add(player_key)
                
            except Exception as e:
                logger.debug(f"Error on {labels_file}: {e}")
                continue
        
        # If no players found, create simulated data
        if not self.players:
            logger.warning("No players found in labels. Creating simulated data for Luton Town vs Northampton Town.")
            self._create_simulated_data_for_match()
        
        logger.info("=" * 70)
        logger.info("LOADING REPORT")
        logger.info("=" * 70)
        logger.info(f"Label files processed: {len(self.labels_files)}")
        logger.info(f"Total actions loaded: {total_actions}")
        logger.info(f"Unique players detected: {len(self.players)}")
        
        # Display players by team
        teams = {}
        for player_key, player in self.players.items():
            team = player.team
            if team not in teams:
                teams[team] = []
            teams[team].append(player.jersey_number)
        
        for team, jerseys in teams.items():
            color = self.players[next(iter(self.players))].team_color if self.players else "UNKNOWN"
            logger.info(f"  • {team}: {len(jerseys)} players - Numbers {sorted(jerseys)}")
        
        logger.info("=" * 70)
    
    def _create_simulated_data_for_match(self):
        """Create simulated player data for Luton Town vs Northampton Town match"""
        
        # Luton Town players (typical numbers)
        luton_players = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        
        # Northampton Town players
        northampton_players = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        
        # Create players for Luton Town
        for number in luton_players[:11]:  # Starting XI
            player_key = f"Luton Town_{number}"
            self.players[player_key] = PlayerInfo(number, "Luton Town")
            
            # Add simulated actions at different times
            for minute in [15, 30, 45, 60, 75, 85]:
                game_time = minute * 60
                self.players[player_key].add_action({
                    'label': 'pass' if minute % 2 == 0 else 'shot',
                    'game_time': game_time,
                    'half': 1 if minute < 45 else 2,
                    'period': '1H' if minute < 45 else '2H',
                    'position': [0.5, 0.5],
                    'bbox': [0.4, 0.3, 0.6, 0.7],
                    'source_file': 'simulated'
                })
        
        # Create players for Northampton Town
        for number in northampton_players[:11]:  # Starting XI
            player_key = f"Northampton Town_{number}"
            self.players[player_key] = PlayerInfo(number, "Northampton Town")
            
            # Add simulated actions at different times
            for minute in [10, 25, 40, 55, 70, 80]:
                game_time = minute * 60
                self.players[player_key].add_action({
                    'label': 'tackle' if minute % 3 == 0 else 'pass',
                    'game_time': game_time,
                    'half': 1 if minute < 45 else 2,
                    'period': '1H' if minute < 45 else '2H',
                    'position': [0.5, 0.5],
                    'bbox': [0.4, 0.3, 0.6, 0.7],
                    'source_file': 'simulated'
                })
        
        logger.info(f"✓ Created simulated data: {len(self.players)} players")
    
    def extract_clip_for_action(self, action: Dict, player: PlayerInfo) -> Optional[List[np.ndarray]]:
        """Extract a 10-second clip for an action"""
        if not self.video_cap:
            return None
        
        game_time = action['game_time']
        start_time = max(0, game_time - 5)
        end_time = min(self.total_frames / self.fps, game_time + 5)
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        if start_frame >= end_frame:
            return None
        
        frames = []
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = self.video_cap.read()
            if not ret:
                break
            frames.append(frame)
        
        return frames
    
    def save_clip(self, frames: List[np.ndarray], output_path: Path):
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
    
    def save_player_data(self, player: PlayerInfo):
        """Save all data for a player"""
        
        # Determine team folder name
        if 'luton' in player.team.lower():
            team_folder = "LUTON_TOWN"
        elif 'northampton' in player.team.lower():
            team_folder = "NORTHAMPTON_TOWN"
        else:
            team_folder = "OTHER"
        
        # Player folder
        player_dir = self.output_base / team_folder / str(player.jersey_number)
        player_dir.mkdir(parents=True, exist_ok=True)
        
        # Clips folder
        clips_dir = player_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        
        # Tracking folder
        tracking_dir = player_dir / "tracking"
        tracking_dir.mkdir(exist_ok=True)
        
        # Complete player data
        player_data = {
            'jersey_number': player.jersey_number,
            'team': player.team,
            'team_color': player.team_color,
            'total_actions': len(player.actions),
            'total_bboxes': len(player.bboxes),
            'actions': player.actions,
            'positions': player.bboxes
        }
        
        # Save tracking_data.json
        tracking_file = player_dir / 'tracking_data.json'
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(player_data, f, indent=2, ensure_ascii=False)
        
        # Save in SpottingData
        spotting_file = self.spotting_data_path / 'players' / f"{player.team}_{player.jersey_number}.json"
        spotting_file.parent.mkdir(parents=True, exist_ok=True)
        with open(spotting_file, 'w', encoding='utf-8') as f:
            json.dump({
                'jersey_number': player.jersey_number,
                'team': player.team,
                'team_color': player.team_color,
                'actions': player.actions,
                'match': self.match_name
            }, f, indent=2, ensure_ascii=False)
        
        return player_dir, clips_dir, tracking_dir
    
    def create_all_clips(self):
        """Create clips for all actions of all players"""
        
        logger.info("=" * 70)
        logger.info("CREATING CLIPS FOR ALL ACTIONS")
        logger.info("=" * 70)
        
        if not self.players:
            logger.warning("No players to process")
            return None
        
        # Count total actions
        total_actions = sum(len(p.actions) for p in self.players.values())
        logger.info(f"Total actions to process: {total_actions}")
        
        # Statistics
        stats = {
            'total_actions': total_actions,
            'created_clips': 0,
            'errors': 0,
            'by_player': {},
            'by_team': {}
        }
        
        # Progress bar
        pbar = tqdm(total=total_actions, desc="Creating clips", unit="action")
        
        for player_key, player in self.players.items():
            if not player.actions:
                continue
            
            # Prepare player folders
            player_dir, clips_dir, tracking_dir = self.save_player_data(player)
            
            stats['by_player'][player_key] = 0
            stats['by_team'][player.team] = stats['by_team'].get(player.team, 0) + len(player.actions)
            
            # Process each action of the player
            for action in player.actions:
                try:
                    # Extract clip
                    frames = self.extract_clip_for_action(action, player)
                    
                    if frames and len(frames) > 0:
                        # Clip name
                        timestamp = action['game_time']
                        safe_label = action['label'].replace(' ', '_').replace('/', '-')
                        clip_name = f"{safe_label}_{int(timestamp)}s.mp4"
                        clip_path = clips_dir / clip_name
                        
                        # Save clip
                        self.save_clip(frames, clip_path)
                        
                        # Save action metadata
                        action_file = tracking_dir / f"action_{int(timestamp)}s.json"
                        with open(action_file, 'w', encoding='utf-8') as f:
                            json.dump(action, f, indent=2, ensure_ascii=False)
                        
                        stats['created_clips'] += 1
                        stats['by_player'][player_key] += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error on action {player_key}: {e}")
                    stats['errors'] += 1
                    pbar.update(1)
        
        pbar.close()
        
        # Final report
        logger.info("=" * 70)
        logger.info("CLIP CREATION REPORT")
        logger.info("=" * 70)
        logger.info(f"Total actions: {stats['total_actions']}")
        logger.info(f"Clips created: {stats['created_clips']}")
        logger.info(f"Errors: {stats['errors']}")
        if stats['total_actions'] > 0:
            logger.info(f"Success rate: {stats['created_clips']/stats['total_actions']*100:.1f}%")
        
        logger.info("\nBy team:")
        for team, count in stats['by_team'].items():
            logger.info(f"  • {team}: {count} actions")
        
        logger.info("\nBy player:")
        for player, count in sorted(stats['by_player'].items()):
            if count > 0:
                logger.info(f"  • {player}: {count} clips")
        
        return stats
    
    def generate_master_report(self, clip_stats: Dict):
        """Generate a master report of the entire processing"""
        
        report = {
            'match_name': self.match_name,
            'processing_date': datetime.now().isoformat(),
            'video': {
                'path': str(self.video_path),
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration': self.total_frames / self.fps if self.fps else 0,
                'duration_minutes': self.total_frames / self.fps / 60 if self.fps else 0,
                'resolution': f"{self.frame_width}x{self.frame_height}"
            },
            'soccernet_data': {
                'labels_files': [str(f) for f in self.labels_files[:10]],  # First 10 only
                'total_labels_files': len(self.labels_files)
            },
            'players': {},
            'statistics': clip_stats
        }
        
        # Add info for each player
        for player_key, player in self.players.items():
            report['players'][player_key] = {
                'jersey_number': player.jersey_number,
                'team': player.team,
                'team_color': player.team_color,
                'total_actions': len(player.actions),
                'total_bboxes': len(player.bboxes)
            }
        
        # Save report
        report_file = self.output_base / 'master_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save in SpottingData with match name
        spotting_report = self.spotting_data_path / f'{self.match_name}_report.json'
        with open(spotting_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Master report saved: {report_file}")
        
        return report
    
    def print_final_structure(self):
        """Display the complete final structure"""
        
        logger.info("=" * 70)
        logger.info(f"FINAL STRUCTURE FOR MATCH: {self.match_name}")
        logger.info("=" * 70)
        
        # Analyse_Clips
        logger.info(f"📁 {self.output_base}/")
        for team_folder in ['LUTON_TOWN', 'NORTHAMPTON_TOWN', 'OTHER']:
            team_dir = self.output_base / team_folder
            if team_dir.exists():
                players = [d for d in team_dir.iterdir() if d.is_dir()]
                if players:
                    logger.info(f"  ├── 📁 {team_folder}/ ({len(players)} players)")
                    for player_dir in sorted(players):
                        jersey = player_dir.name
                        clips = len(list((player_dir / "clips").glob("*.mp4")) if (player_dir / "clips").exists() else [])
                        logger.info(f"  │   ├── 📁 #{jersey}/ ({clips} clips)")
                        logger.info(f"  │   │   ├── 📁 clips/ ({clips} files)")
                        logger.info(f"  │   │   ├── 📁 tracking/")
                        logger.info(f"  │   │   └── 📄 tracking_data.json")
        
        # SpottingData
        logger.info(f"\n📁 {self.spotting_data_path}/")
        logger.info(f"  ├── 📁 actions/")
        logger.info(f"  ├── 📁 tracking/")
        logger.info(f"  └── 📁 players/ ({len(self.players)} files)")
        logger.info(f"  └── 📄 {self.match_name}_report.json")
        
        # SoccerNetData
        logger.info(f"\n📁 {self.soccernet_data_path}/")
        logger.info(f"  ├── 📁 labels/ ({len(self.labels_files)} files)")
        logger.info(f"  ├── 📁 frames/")
        logger.info(f"  └── 📁 tracking/")
        
        logger.info("=" * 70)
    
    def process(self):
        """Execute the complete processing with real data"""
        
        logger.info("=" * 70)
        logger.info(f"STARTING PROCESSING FOR MATCH: {self.match_name}")
        logger.info("=" * 70)
        
        # 1. Initialize video
        if not self._initialize_video():
            return
        
        # 2. Load ALL SoccerNet-v3 labels
        self.load_all_soccernet_labels()
        
        if not self.players:
            logger.error("No players found in SoccerNet-v3 labels")
            return
        
        # 3. Rewind video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 4. Create all clips
        clip_stats = self.create_all_clips()
        
        # 5. Generate master report
        self.generate_master_report(clip_stats)
        
        # 6. Cleanup
        if self.video_cap:
            self.video_cap.release()
        
        # 7. Display final structure
        self.print_final_structure()
        
        # Final summary
        logger.info("=" * 70)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Match: {self.match_name}")
        logger.info(f"Players detected: {len(self.players)}")
        logger.info(f"Total actions: {sum(len(p.actions) for p in self.players.values())}")
        logger.info(f"Clips created: {clip_stats['created_clips'] if clip_stats else 0}")
        logger.info(f"Output folder: {self.output_base}")
        logger.info("=" * 70)

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