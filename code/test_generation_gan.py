import torch
import cv2
import numpy as np
from architecture_gan import Generateur

def tester_generateur_gan(chemin_poids, nb_visages=3, taille_latent=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement de l'artiste (Générateur)
    generateur = Generateur(taille_latent).to(device)
    generateur.load_state_dict(torch.load(chemin_poids, map_location=device))
    generateur.eval()
    
    # 2. Création de "graines" d'identités (du bruit purement aléatoire)
    # On crée 'nb_visages' vecteurs de 100 nombres
    with torch.no_grad():
        bruit = torch.randn(nb_visages, taille_latent, device=device)
        images_generees = generateur(bruit)
        
    # 3. Affichage des résultats
    for i in range(nb_visages):
        # Le GAN sort des valeurs entre -1 et 1. Il faut les ramener entre 0 et 255.
        img = images_generees[i].cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1.0) / 2.0) * 255.0 
        img = img.astype(np.uint8)
        
        # Conversion RGB vers BGR pour OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Agrandissement (256x256) pour bien observer les détails générés (texture, contrastes)
        img_agrandie = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(f"Identite Synthetique {i+1}", img_agrandie)
        
    print("Appuyez sur une touche pour fermer les fenêtres.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tester_generateur_gan("poids_generateur.pth", nb_visages=4)