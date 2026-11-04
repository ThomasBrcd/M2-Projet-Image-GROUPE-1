import torch
import torch.nn as nn

class AutoEncodeurVisage(nn.Module):
    def __init__(self):
        super(AutoEncodeurVisage, self).__init__()
        
        # ==========================================
        # 1. L'ENCODEUR (Compression vers l'espace latent)
        # ==========================================
        # nn.Conv2d permet de balayer l'image pour en extraire les caractéristiques.
        # stride=2 divise la taille de l'image par 2 à chaque étape.
        self.encodeur = nn.Sequential(
            # Entrée : image RGB (3 canaux). Sortie : 16 canaux de caractéristiques
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), # Fonction d'activation (introduit de la non-linéarité)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Sortie finale de l'encodeur : le vecteur latent (très compressé)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # ==========================================
        # 2. LE DÉCODEUR (Reconstruction de l'image)
        # ==========================================
        # nn.ConvTranspose2d fait l'inverse : il agrandit l'image à chaque étape.
        self.decodeur = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # On revient à 3 canaux pour recréer une image couleur RGB
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Force les valeurs des pixels à rester entre 0 et 1
        )

    def forward(self, x):
        """
        C'est ici que l'on définit le trajet de la donnée.
        L'image (x) traverse l'encodeur, devient un vecteur latent, puis traverse le décodeur.
        """
        # 1. L'image devient une représentation mathématique compressée
        vecteur_latent = self.encodeur(x)
        
        # 2. On reconstruit l'image à partir de ce vecteur
        image_reconstruite = self.decodeur(vecteur_latent)
        
        # On retourne l'image reconstruite ET le vecteur latent (utile pour le modifier plus tard)
        return image_reconstruite, vecteur_latent

# ==========================================
# TEST DU MODÈLE (Vérification de l'architecture)
# ==========================================
if __name__ == "__main__":
    # Création du modèle
    modele = AutoEncodeurVisage()
    print(modele)
    
    # Création d'une "fausse" image aléatoire pour tester (Format PyTorch : Batch, Canaux, Hauteur, Largeur)
    # Ici : 1 image, 3 canaux (RGB), 64x64 pixels
    fausse_image = torch.randn(1, 3, 64, 64)
    
    # On passe l'image dans le réseau
    image_sortie, espace_latent = modele(fausse_image)
    
    print("\n--- Dimensions des données ---")
    print(f"Taille de l'image d'entrée   : {fausse_image.shape}")
    print(f"Taille du vecteur latent     : {espace_latent.shape}")
    print(f"Taille de l'image de sortie  : {image_sortie.shape}")