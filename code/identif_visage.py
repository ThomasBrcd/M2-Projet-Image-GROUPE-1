import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from modele_ia import AutoEncodeurVisage
import numpy as np
from architecture_gan import Generateur

mp_face_mesh = mp.solutions.face_mesh


def creer_masque_facemesh(shape_fond, landmarks):
    """Crée un masque fluide avec une extension naturelle et courbée du front."""
    h, w = shape_fond[:2]
    masque = np.zeros((h, w, 3), dtype=np.float32)
    
    points_contour = []
    for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
        pt = landmarks.landmark[connection[0]]
        points_contour.append([int(pt.x * w), int(pt.y * h)])
        
    points_contour = np.array(points_contour, dtype=np.int32)
    
    # EXTENSION NATURELLE DU FRONT (Proportionnelle)
    y_min = np.min(points_contour[:, 1])
    y_max = np.max(points_contour[:, 1])
    y_centre = y_min + (y_max - y_min) / 2
    
    for i in range(len(points_contour)):
        # Si le point est dans la moitié haute du visage
        if points_contour[i][1] < y_centre:
            distance_y = y_centre - points_contour[i][1]
            # On étire proportionnellement pour garder la courbe (15% d'augmentation)
            points_contour[i][1] = int(y_centre - (distance_y * 1.15))
            
    # Sécurité pour les bordures de l'image
    points_contour[:, 1] = np.clip(points_contour[:, 1], 0, h - 1)
    
    # Création du polygone plein
    hull = cv2.convexHull(points_contour)
    cv2.fillConvexPoly(masque, hull, (1.0, 1.0, 1.0))
    
    # Flou gaussien pour fondre les bords
    masque = cv2.GaussianBlur(masque, (21, 21), 0)
    return masque


def fondre_visage(image_fond, visage_anonymise):
    """
    Incruste le visage anonymisé dans le fond original en utilisant 
    un masque ovale aux bords floutés (feathering).
    """
    h, w = image_fond.shape[:2]
    
    # 1. Créer un masque noir de la taille de la boîte
    masque = np.zeros((h, w, 3), dtype=np.float32)
    
    centre = (w // 2, int(h * 0.40))
    
    axes = (int(w / 2.2), int(h / 1.6))
    cv2.ellipse(masque, centre, axes, 0, 0, 360, (1.0, 1.0, 1.0), -1)
    
    # 3. Flouter fortement le masque pour créer un dégradé sur les bords
    masque = cv2.GaussianBlur(masque, (31, 31), 0)
    
    # 4. Fusionner mathématiquement les deux images selon ce masque
    fond_float = image_fond.astype(np.float32)
    visage_float = visage_anonymise.astype(np.float32)
    
    image_fusionnee = (visage_float * masque) + (fond_float * (1.0 - masque))
    
    return image_fusionnee.astype(np.uint8)


def appliquer_autoencodeur(visage_bgr, modele, device, force_bruit=1.0):
    """
    Passe la région du visage dans le réseau, modifie son espace latent 
    pour l'anonymiser, et retourne l'image générée.
    - force_bruit : Plus la valeur est haute, plus le visage sera modifié/déformé.
    """
    h_orig, w_orig = visage_bgr.shape[:2]
    
    # 1. OpenCV vers PyTorch
    visage_rgb = cv2.cvtColor(visage_bgr, cv2.COLOR_BGR2RGB)
    visage_redimensionne = cv2.resize(visage_rgb, (64, 64))
    
    transformations = transforms.Compose([transforms.ToTensor()])
    tenseur_entree = transformations(visage_redimensionne).unsqueeze(0).to(device)
    
    # 2. IA et MANIPULATION DE L'ESPACE LATENT
    with torch.no_grad():
        # A. L'image passe d'abord uniquement dans l'encodeur
        vecteur_latent = modele.encodeur(tenseur_entree)
        
        # B. Ajout du bruit pour altérer l'identité (Anonymisation)
        # torch.randn_like crée un bruit aléatoire de la même taille que le vecteur
        bruit = torch.randn_like(vecteur_latent) * force_bruit
        vecteur_latent_modifie = vecteur_latent + bruit
        
        # C. Le décodeur reconstruit le nouveau visage à partir du vecteur modifié
        tenseur_sortie = modele.decodeur(vecteur_latent_modifie)
        
    # 3. PyTorch vers OpenCV
    image_generee = tenseur_sortie.squeeze(0).cpu().numpy()
    image_generee = image_generee.transpose(1, 2, 0) 
    
    image_generee = (image_generee * 255).astype(np.uint8)
    image_generee_bgr = cv2.cvtColor(image_generee, cv2.COLOR_RGB2BGR)
    
    return cv2.resize(image_generee_bgr, (w_orig, h_orig))



# ==========================================
# 1. FONCTIONS DE TRAITEMENT (LES FILTRES)
# ==========================================

def appliquer_flou(visage, taille_noyau=(51, 51)):
    """Applique un flou gaussien sur la zone du visage."""
    # La taille du noyau doit être composée de nombres impairs
    return cv2.GaussianBlur(visage, taille_noyau, 0)

def appliquer_pixelisation(visage, resolution_blocs=10):
    """Applique un effet mosaïque en réduisant puis agrandissant l'image."""
    hauteur, largeur = visage.shape[:2]
    # Étape 1 : Réduire l'image
    visage_reduit = cv2.resize(visage, (resolution_blocs, resolution_blocs), interpolation=cv2.INTER_LINEAR)
    # Étape 2 : Agrandir à la taille initiale sans lisser (INTER_NEAREST)
    visage_pixelise = cv2.resize(visage_reduit, (largeur, hauteur), interpolation=cv2.INTER_NEAREST)
    return visage_pixelise

def appliquer_masque(visage):
    """Remplace la zone du visage par un rectangle noir complet."""
    visage_noir = visage.copy()
    # Met toutes les composantes de couleur (BGR) à 0
    visage_noir[:] = (0, 0, 0)
    return visage_noir


def transferer_couleur_lab(image_source, image_generee):
    """Calque l'éclairage et la carnation de l'image source sur le visage généré."""
    # Conversion en espace LAB (Luminance, Canal A, Canal B)
    source_lab = cv2.cvtColor(image_source, cv2.COLOR_BGR2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(image_generee, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calcul des moyennes (teint) et écart-types (contrastes)
    mean_src, std_src = cv2.meanStdDev(source_lab)
    mean_gen, std_gen = cv2.meanStdDev(gen_lab)

    # Alignement mathématique des canaux
    for i in range(3):
        gen_lab[:, :, i] = ((gen_lab[:, :, i] - mean_gen[i][0]) * 
                            (std_src[i][0] / (std_gen[i][0] + 1e-5))) + mean_src[i][0]

    gen_lab = np.clip(gen_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gen_lab, cv2.COLOR_LAB2BGR)


def appliquer_gan(visage_bgr, generateur, vecteur_identite, device):
    h_orig, w_orig = visage_bgr.shape[:2]
    
    visage_rgb = cv2.cvtColor(visage_bgr, cv2.COLOR_BGR2RGB)
    visage_resized = cv2.resize(visage_rgb, (128, 128)) # Passage en 128
    
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tenseur_entree = transformations(visage_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        tenseur_sortie = generateur(tenseur_entree, vecteur_identite)
        
    image_generee = tenseur_sortie.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image_generee = ((image_generee + 1.0) / 2.0) * 255.0
    image_generee_bgr = cv2.cvtColor(image_generee.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Redimensionnement au format d'origine de la boîte
    image_generee_bgr = cv2.resize(image_generee_bgr, (w_orig, h_orig))
    
    # APPLICATION DU TRANSFERT DE COULEUR
    return transferer_couleur_lab(visage_bgr, image_generee_bgr)


# ==========================================
# 2. FONCTION PRINCIPALE DE GESTION
# ==========================================

def anonymiser_image(chemin_image, methode="flou"):
    """
    Charge l'image, détecte les visages et applique le filtre choisi.
    Méthodes supportées : 'flou', 'pixel', 'masque'
    """
    # Chargement de l'image
    image = cv2.imread(chemin_image)
    if image is None:
        print(f"Erreur : Impossible de charger l'image '{chemin_image}'. Vérifiez le chemin.")
        return

    # Conversion en niveaux de gris pour la détection
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Chargement du modèle de détection de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    visages = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    # Traitement de chaque visage détecté
    for (x, y, w, h) in visages:
        # Extraction de la région d'intérêt (Region Of Interest - ROI)
        roi_visage = image[y:y+h, x:x+w]
        
        # Application du filtre selon le choix de l'utilisateur
        if methode == "flou":
            roi_anonymisee = appliquer_flou(roi_visage)
        elif methode == "pixel":
            roi_anonymisee = appliquer_pixelisation(roi_visage)
        elif methode == "masque":
            roi_anonymisee = appliquer_masque(roi_visage)
        else:
            print("Erreur : Méthode inconnue. Choisissez 'flou', 'pixel' ou 'masque'.")
            return
            
        # Remplacement du visage original par le visage anonymisé
        image[y:y+h, x:x+w] = roi_anonymisee

    # Affichage du résultat
    cv2.imshow(f"Resultat - Methode : {methode}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def anonymiser_image_ia(chemin_image, methode="gan"):
    """Applique le modèle génératif conditionnel sur une image fixe avec découpage FaceMesh."""
    image = cv2.imread(chemin_image)
    if image is None:
        print(f"Erreur : Impossible de charger l'image '{chemin_image}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Chargement du modèle et création de l'identité
    if methode == "gan":
        generateur_gan = Generateur(taille_latent=512).to(device)
        generateur_gan.load_state_dict(torch.load("poids_generateur.pth", map_location=device))
        generateur_gan.eval()
        vecteur_identite_fixe = torch.randn(1, 512, device=device)

    # 2. Configuration de FaceMesh pour image fixe (static_image_mode=True)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resultats = face_mesh.process(image_rgb)

        if resultats.multi_face_landmarks:
            for face_landmarks in resultats.multi_face_landmarks:
                h_img, w_img, _ = image.shape
                
                # A. Calcul de la boîte englobante dynamique
                x_min = min([lm.x for lm in face_landmarks.landmark])
                y_min = min([lm.y for lm in face_landmarks.landmark])
                x_max = max([lm.x for lm in face_landmarks.landmark])
                y_max = max([lm.y for lm in face_landmarks.landmark])
                
                x = max(0, int(x_min * w_img))
                y = max(0, int(y_min * h_img))
                w = min(w_img - x, int((x_max - x_min) * w_img))
                h = min(h_img - y, int((y_max - y_min) * h_img))
                
                # Agrandissement pour englober la racine des cheveux et le menton
                y = max(0, int(y - (h * 0.15)))
                h = int(h * 1.25)
                if y + h > h_img: h = h_img - y
                if x + w > w_img: w = w_img - x
                
                roi_visage = image[y:y+h, x:x+w]
                if roi_visage.size == 0: continue
                
                # B. Génération de la nouvelle texture
                if methode == "gan":
                    roi_anonymisee = appliquer_gan(roi_visage, generateur_gan, vecteur_identite_fixe, device)
                
                # C. Découpage exact et fusion
                masque_complet = creer_masque_facemesh(image.shape, face_landmarks)
                masque_roi = masque_complet[y:y+h, x:x+w]
                
                fond_float = roi_visage.astype(np.float32)
                visage_float = roi_anonymisee.astype(np.float32)
                fusion = (visage_float * masque_roi) + (fond_float * (1.0 - masque_roi))
                
                image[y:y+h, x:x+w] = fusion.astype(np.uint8)

    # 3. Affichage du résultat final
    cv2.imshow("Image Anonymisee - IA Generative", image)
    print("Appuyez sur une touche pour fermer l'image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def anonymiser_video(methode="flou"):
    """
    Ouvre la webcam, détecte les visages en temps réel et applique le filtre choisi.
    Méthodes supportées : 'flou', 'pixel', 'masque'
    """
    # 0 correspond à la webcam par défaut (tu peux mettre un chemin vidéo "ma_video.mp4" à la place)
    cap = cv2.VideoCapture(0)
    
    # Chargement du modèle de détection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Appuyez sur la touche 'q' pour quitter la vidéo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo.")
            break

        # Conversion en niveaux de gris pour la détection
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # On garde les mêmes paramètres anti-faux positifs que pour ton image
        visages = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

        for (x, y, w, h) in visages:
            roi_visage = frame[y:y+h, x:x+w]
            
            # Application du filtre
            if methode == "flou":
                roi_anonymisee = appliquer_flou(roi_visage)
            elif methode == "pixel":
                roi_anonymisee = appliquer_pixelisation(roi_visage)
            elif methode == "masque":
                roi_anonymisee = appliquer_masque(roi_visage)
            else:
                print("Erreur : Méthode inconnue.")
                return
                
            frame[y:y+h, x:x+w] = roi_anonymisee

        # Affichage du résultat en temps réel
        cv2.imshow(f"Webcam - Methode : {methode}", frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la webcam et fermer la fenêtre
    cap.release()
    cv2.destroyAllWindows()



def anonymiser_video_ia(methode="gan"):
    cap = cv2.VideoCapture(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if methode == "gan":
        generateur_gan = Generateur(taille_latent=512).to(device)
        generateur_gan.load_state_dict(torch.load("poids_generateur.pth", map_location=device))
        #generateur_gan = generateur_gan.half()
        vecteur_identite_fixe = torch.randn(1, 512, device=device)


    print("Appuyez sur 'q' pour quitter.")

    # Utilisation de FaceMesh au lieu de FaceDetection
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultats = face_mesh.process(frame_rgb)

            if resultats.multi_face_landmarks:
                for face_landmarks in resultats.multi_face_landmarks:
                    h_img, w_img, _ = frame.shape
                    
                    # 1. Calcul de la boîte englobante (ROI) dynamique à partir des points extrêmes
                    x_min = min([lm.x for lm in face_landmarks.landmark])
                    y_min = min([lm.y for lm in face_landmarks.landmark])
                    x_max = max([lm.x for lm in face_landmarks.landmark])
                    y_max = max([lm.y for lm in face_landmarks.landmark])
                    
                    x = max(0, int(x_min * w_img))
                    y = max(0, int(y_min * h_img))
                    w = min(w_img - x, int((x_max - x_min) * w_img))
                    h = min(h_img - y, int((y_max - y_min) * h_img))
                    
                    # Sécurité d'agrandissement pour englober la texture
                    y = max(0, int(y - (h * 0.15)))
                    h = int(h * 1.25)
                    if y + h > h_img: h = h_img - y
                    if x + w > w_img: w = w_img - x
                    
                    roi_visage = frame[y:y+h, x:x+w]
                    if roi_visage.size == 0: continue
                    
                    if methode == "gan":
                        roi_anonymisee = appliquer_gan(roi_visage, generateur_gan, vecteur_identite_fixe, device)
                    
                    # 2. Création du masque exact et extraction sur la ROI
                    masque_complet = creer_masque_facemesh(frame.shape, face_landmarks)
                    masque_roi = masque_complet[y:y+h, x:x+w]
                    
                    # 3. Fusion douce
                    fond_float = roi_visage.astype(np.float32)
                    visage_float = roi_anonymisee.astype(np.float32)
                    fusion = (visage_float * masque_roi) + (fond_float * (1.0 - masque_roi))
                    
                    frame[y:y+h, x:x+w] = fusion.astype(np.uint8)

            cv2.imshow(f"Webcam IA - Methode : {methode}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 3. EXÉCUTION DU SCRIPT
# ==========================================

if __name__ == "__main__":
    # Remplace "mon_image.jpg" par le chemin réel de ton image
    nom_fichier = "mon_dataset/visages/visage1.jpg"
    
    # Teste les différentes méthodes en changeant le paramètre
    #anonymiser_image(nom_fichier, methode="flou")
    #anonymiser_image(nom_fichier, methode="pixel")
    #anonymiser_image(nom_fichier, methode="masque")
    #anonymiser_video(methode="pixel")
    #anonymiser_video_ia(methode="gan")
    anonymiser_image_ia(nom_fichier, methode="gan")