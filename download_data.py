from SoccerNet.Downloader import SoccerNetDownloader

# On définit où on veut enregistrer les données
path_to_data = "C:/SoccerNetData"

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=path_to_data)

# On télécharge uniquement les labels (fichiers JSON) pour le split "test"
# C'est rapide car ce ne sont que des fichiers texte (pas de vidéos lourdes)
mySoccerNetDownloader.downloadGames(files=["Labels-v3.json"], split=["test"], task="frames")

print("Téléchargement terminé !")