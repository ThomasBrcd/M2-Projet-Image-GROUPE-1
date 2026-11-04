import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modele_ia import AutoEncodeurVisage  # On importe le modèle créé précédemment

def entrainer_modele(dossier_dataset, epochs=10, batch_size=32):
    """
    Entraîne l'auto-encodeur sur un dossier d'images et sauvegarde les poids.
    - epochs : Nombre de fois où l'IA va voir l'intégralité du dataset.
    - batch_size : Nombre d'images traitées en même temps (pour aller plus vite).
    """
    # 1. Configuration (Détection automatique de la carte graphique si dispo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device.type.upper()}")
    
    # Instanciation du modèle et envoi sur le processeur/carte graphique
    modele = AutoEncodeurVisage().to(device)
    
    # Fonction de perte : Mean Squared Error (calcule l'écart de couleur pixel par pixel)
    critere = nn.MSELoss()
    
    # Optimiseur : Adam (l'algorithme qui met à jour les poids du réseau)
    # lr = learning rate (vitesse d'apprentissage, 0.001 est le standard)
    optimiseur = optim.Adam(modele.parameters(), lr=0.001)

    # 2. Préparation des images
    # On force toutes les images à faire 64x64 pixels et on les convertit en tenseurs mathématiques
    transformations = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Chargement du dataset (lit automatiquement toutes les images du dossier)
    try:
        dataset = datasets.ImageFolder(root=dossier_dataset, transform=transformations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print("Erreur de chargement du dataset. Vérifiez la structure de vos dossiers.")
        return

    print(f"Début de l'entraînement sur {len(dataset)} images...")

    # 3. La boucle d'entraînement
    for epoch in range(epochs):
        perte_totale = 0
        
        # On passe les images par paquets (batch)
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            
            # Étape A : Remise à zéro de la mémoire de l'optimiseur
            optimiseur.zero_grad()
            
            # Étape B : Forward (L'IA essaie de reconstruire l'image)
            images_reconstruites, _ = modele(batch_images)
            
            # Étape C : Évaluation (L'IA regarde à quel point elle a faux)
            perte = critere(images_reconstruites, batch_images)
            
            # Étape D : Backward (Calcul des corrections à apporter)
            perte.backward()
            
            # Étape E : Apprentissage (Application des corrections)
            optimiseur.step()
            
            perte_totale += perte.item()
            
        # Affichage de la progression
        perte_moyenne = perte_totale / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Marge d'erreur : {perte_moyenne:.4f}")

    # 4. Sauvegarde du cerveau de l'IA
    chemin_sauvegarde = "poids_autoencodeur.pth"
    torch.save(modele.state_dict(), chemin_sauvegarde)
    print(f"\nEntraînement terminé ! Le modèle est sauvegardé sous : {chemin_sauvegarde}")

if __name__ == "__main__":
    # Remplace par le chemin de ton dossier contenant les images
    chemin_dataset = "mon_dataset/"
    
    # On lance 10 cycles d'apprentissage par paquets de 32 images
    entrainer_modele(chemin_dataset, epochs=10, batch_size=32)