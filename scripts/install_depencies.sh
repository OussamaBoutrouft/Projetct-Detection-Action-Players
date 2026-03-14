#!/bin/bash
cd /home/ec2-user/soccernet-tracker

# Installation des dépendances système
sudo yum update -y
sudo yum install -y python3 python3-pip mesa-libGL mesa-libGL-devel libSM libXrender libXext

# Installation des dépendances Python
pip3 install -r requirements.txt

# Installation de l'agent CodeDeploy (si pas déjà présent)
sudo yum install -y ruby
wget https://aws-codedeploy-eu-west-3.s3.eu-west-3.amazonaws.com/latest/install
chmod +x ./install
sudo ./install auto

echo "Dépendances installées avec succès"