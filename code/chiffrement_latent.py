import cv2
import torch
import numpy as np
from torchvision import transforms
from modele_ia import AutoEncodeurVisage

def generer_bruit_secret(taille_vecteur, cle_secrete, device):
    """
    Génère un tenseur de bruit mathématique basé sur un mot de passe.
    Pour un même mot de passe, le bruit sera toujours rigoureusement identique.
    """
    # On utilise la clé comme "graine" (seed) pour le générateur aléatoire
    torch.manual_seed(cle_secrete)
    bruit = torch.randn(taille_vecteur, device=device) * 1.5 # Multiplié pour bien déformer
    return bruit

def demonstration_chiffrement(chemin_image, modele_chemin, cle_secrete=12345):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement du modèle
    modele = AutoEncodeurVisage().to(device)
    modele.load_state_dict(torch.load(modele_chemin, map_location=device))
    modele.eval()
    
    # 2. Préparation de l'image
    visage_bgr = cv2.imread(chemin_image)
    visage_bgr = cv2.resize(visage_bgr, (64, 64))
    visage_rgb = cv2.cvtColor(visage_bgr, cv2.COLOR_BGR2RGB)
    
    transformations = transforms.Compose([transforms.ToTensor()])
    tenseur_entree = transformations(visage_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- ÉTAPE A : ENCODAGE ---
        vecteur_latent_original = modele.encodeur(tenseur_entree)
        
        # --- ÉTAPE B : CHIFFREMENT ---
        bruit_secret = generer_bruit_secret(vecteur_latent_original.shape, cle_secrete, device)
        vecteur_chiffre = vecteur_latent_original + bruit_secret
        
        # Génération du visage anonyme
        sortie_anonyme = modele.decodeur(vecteur_chiffre)
        
        # --- ÉTAPE C : DÉCHIFFREMENT ---
        # On soustrait la clé mathématique exacte pour retrouver le vecteur d'origine
        vecteur_dechiffre = vecteur_chiffre - bruit_secret
        
        # Génération du visage restauré
        sortie_restauree = modele.decodeur(vecteur_dechiffre)
        
    # 3. Conversion PyTorch -> OpenCV pour l'affichage
    def tenseur_vers_cv2(tenseur):
        img = tenseur.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    img_anonyme = tenseur_vers_cv2(sortie_anonyme)
    img_restauree = tenseur_vers_cv2(sortie_restauree)
    
    # Affichage comparatif
    cv2.imshow("1. Visage Original", visage_bgr)
    cv2.imshow(f"2. Visage Anonyme (Cle: {cle_secrete})", img_anonyme)
    cv2.imshow("3. Visage Restaure (Dechiffre)", img_restauree)
    
    print("Appuyez sur une touche pour fermer les fenêtres.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test avec une photo recadrée de ton visage et le modèle entraîné
    demonstration_chiffrement("mon_dataset/visages/visage1.jpg", "poids_autoencodeur.pth", cle_secrete=9999)