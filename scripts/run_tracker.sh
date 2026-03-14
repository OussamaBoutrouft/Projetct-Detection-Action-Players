#!/bin/bash
cd /home/ec2-user/soccernet-tracker

# Lancer votre script en arrière-plan avec logging
nohup python3 src/briqx_video_clipper5.py > /home/ec2-user/soccernet-tracker/traitement.log 2>&1 &

# Sauvegarder le PID
echo $! > /home/ec2-user/soccernet-tracker/tracker.pid

echo "Traitement lancé avec PID $(cat /home/ec2-user/soccernet-tracker/tracker.pid)"