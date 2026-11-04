import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from facenet_pytorch import InceptionResnetV1

class EvaluateurAnonymisation:
    def __init__(self):
        # Chargement du modèle de reconnaissance faciale (FaceNet) pré-entraîné
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modele_reconnaissance = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def extraire_vecteur_identite(self, visage_bgr):
        """Transforme un visage en un vecteur mathématique d'identité (Embeddings)."""
        # Redimensionnement requis par FaceNet (160x160)
        visage_rgb = cv2.cvtColor(visage_bgr, cv2.COLOR_BGR2RGB)
        visage_resized = cv2.resize(visage_rgb, (160, 160))
        
        # Normalisation entre -1 et 1
        visage_norm = (visage_resized / 127.5) - 1.0
        
        # Format PyTorch (Batch, Canaux, H, W)
        tenseur = torch.tensor(visage_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            vecteur_identite = self.modele_reconnaissance(tenseur)
        return vecteur_identite

    def evaluer(self, visage_original, visage_anonymise):
        """Calcule le PSNR, le SSIM et la similarité d'identité entre deux visages."""
        
        # 1. Calcul du PSNR (Qualité globale)
        valeur_psnr = psnr(visage_original, visage_anonymise)
        
        # 2. Calcul du SSIM (Similarité structurelle)
        # channel_axis=2 précise que c'est une image couleur (BGR)
        valeur_ssim = ssim(visage_original, visage_anonymise, channel_axis=2)
        
        # 3. Calcul de la préservation d'identité
        vecteur_orig = self.extraire_vecteur_identite(visage_original)
        vecteur_anon = self.extraire_vecteur_identite(visage_anonymise)
        
        # Calcul de la distance cosinus entre les deux identités (-1 à 1)
        # 1 = Identité parfaitement identique, < 0.6 = Personnes différentes
        similarite_identite = torch.nn.functional.cosine_similarity(vecteur_orig, vecteur_anon).item()
        
        return valeur_psnr, valeur_ssim, similarite_identite

# ==========================================
# EXÉCUTION DU TEST
# ==========================================
if __name__ == "__main__":
    # Import de tes filtres depuis ton script principal
    from identif_visage import appliquer_flou, appliquer_autoencodeur, AutoEncodeurVisage
    
    # 1. Charger un visage test (utilise l'image recadrée de ton visage)
    # Assure-toi d'avoir une image contenant SEULEMENT ton visage pour ce test
    visage_orig = cv2.imread("mon_dataset/visages/visage1.jpg") 
    visage_orig = cv2.resize(visage_orig, (128, 128)) # Standardisation de la taille
    
    # 2. Préparer les versions anonymisées
    visage_flou = appliquer_flou(visage_orig)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modele_ae = AutoEncodeurVisage().to(device)
    modele_ae.load_state_dict(torch.load("poids_autoencodeur.pth", map_location=device))
    modele_ae.eval()
    
    visage_ia = appliquer_autoencodeur(visage_orig, modele_ae, device, force_bruit=0.8)
    
    # 3. Lancer l'évaluation
    evaluateur = EvaluateurAnonymisation()
    
    print("\n--- RÉSULTATS : MÉTHODE FLOU ---")
    p, s, i = evaluateur.evaluer(visage_orig, visage_flou)
    print(f"PSNR : {p:.2f} dB")
    print(f"SSIM : {s:.3f} (1.0 = identique)")
    print(f"Maintien de l'identité : {i:.3f} (Seuil de rupture vers 0.6)")
    
    print("\n--- RÉSULTATS : MÉTHODE IA GÉNÉRATIVE ---")
    p, s, i = evaluateur.evaluer(visage_orig, visage_ia)
    print(f"PSNR : {p:.2f} dB")
    print(f"SSIM : {s:.3f} (1.0 = identique)")
    print(f"Maintien de l'identité : {i:.3f} (Seuil de rupture vers 0.6)")