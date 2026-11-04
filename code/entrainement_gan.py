import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from architecture_gan import Generateur, Discriminateur
import os

def entrainer_cgan(dossier_dataset, epochs=100, batch_size=128, taille_latent=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement cGAN lancé sur : {device.type.upper()}")

    # 1. Initialisation
    generateur = Generateur(taille_latent).to(device)
    discriminateur = Discriminateur().to(device)

    critere = nn.BCELoss()
    opt_gen = optim.Adam(generateur.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminateur.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if os.path.exists("poids_generateur.pth") and os.path.exists("poids_discriminateur.pth"):
        generateur.load_state_dict(torch.load("poids_generateur.pth", map_location=device))
        discriminateur.load_state_dict(torch.load("poids_discriminateur.pth", map_location=device))
        print("Sauvegardes trouvées : Reprise de l'entraînement en cours.")
    else:
        print("Aucune sauvegarde trouvée : Début d'un nouvel entraînement.")

    # 2. Préparation des données
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(root=dossier_dataset, transform=transformations)
    # num_workers=2 permet d'utiliser plusieurs cœurs CPU pour charger les images plus vite vers le GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"Début de l'entraînement sur {len(dataset)} images...")

    # 3. Boucle d'apprentissage
    for epoch in range(epochs):
        for i, (images_reelles, _) in enumerate(dataloader):
            taille_batch = images_reelles.size(0)
            images_reelles = images_reelles.to(device)

            vrais_labels = torch.ones(taille_batch, device=device)
            faux_labels = torch.zeros(taille_batch, device=device)

            # --- A. ENTRAÎNEMENT DU DISCRIMINATEUR ---
            opt_disc.zero_grad()
            predictions_reelles = discriminateur(images_reelles)
            perte_disc_reelle = critere(predictions_reelles, vrais_labels)
            
            bruit = torch.randn(taille_batch, taille_latent, device=device)
            # Le générateur conditionnel reçoit l'image source ET le bruit
            fausses_images = generateur(images_reelles, bruit)
            
            predictions_fausses = discriminateur(fausses_images.detach()) 
            perte_disc_fausse = critere(predictions_fausses, faux_labels)
            
            perte_disc_totale = perte_disc_reelle + perte_disc_fausse
            perte_disc_totale.backward()
            opt_disc.step()

            # --- B. ENTRAÎNEMENT DU GÉNÉRATEUR ---
            opt_gen.zero_grad()
            predictions_piege = discriminateur(fausses_images)
            perte_gen = critere(predictions_piege, vrais_labels)
            
            perte_gen.backward()
            opt_gen.step()

        # Affichage et Sauvegarde automatique
        print(f"Epoch [{epoch+1}/{epochs}] - Perte Disc: {perte_disc_totale.item():.4f} | Perte Gen: {perte_gen.item():.4f}")
        torch.save(generateur.state_dict(), "poids_generateur.pth")
        torch.save(discriminateur.state_dict(), "poids_discriminateur.pth")

    print("\nEntraînement terminé !")

if __name__ == "__main__":
    chemin_dataset = "mon_dataset/"
    # 100 epochs est le point d'équilibre visé
    entrainer_cgan(chemin_dataset, epochs=100, batch_size=128)