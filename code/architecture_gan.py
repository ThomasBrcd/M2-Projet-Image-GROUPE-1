import torch
import torch.nn as nn

# ==========================================
# 1. LE GÉNÉRATEUR CONDITIONNEL (128x128)
# ==========================================
class Generateur(nn.Module):
    def __init__(self, taille_latent=512):
        super(Generateur, self).__init__()
        
        # ENCODEUR : 4 couches (Extrait plus de détails)
        self.encodeur = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),      # -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),     # -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),    # -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # NOUVELLE COUCHE : Analyse des micro-détails
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),    # -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # DÉCODEUR : 4 couches (Reconstruit en 128x128)
        self.decodeur = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), # -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # NOUVELLE COUCHE : Upscaling final vers 128x128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),    # -> 128x128
            nn.Tanh()
        )

    def forward(self, image_originale, vecteur_identite):
        traits_visage = self.encodeur(image_originale)
        # L'espace latent passe à 512 dimensions pour s'adapter à la profondeur
        vecteur = vecteur_identite.view(-1, 512, 1, 1).expand_as(traits_visage)
        traits_modifies = traits_visage + vecteur
        return self.decodeur(traits_modifies)

# ==========================================
# 2. LE DISCRIMINATEUR (128x128)
# ==========================================
class Discriminateur(nn.Module):
    def __init__(self):
        super(Discriminateur, self).__init__()
        self.reseau = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),       # -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),     # -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),    # -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),    # -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # NOUVELLE COUCHE : Compression finale pour l'inspecteur
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),   # -> 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),     # -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.reseau(x).view(-1, 1).squeeze(1)