# Tests et validation

## État des vérifications

| Vérification | Dernier résultat enregistré |
|---|---|
| Syntaxe Python de l'application et des tests | Réussie avec CPython 3.10 |
| Tests unitaires et interface (`not integration`) | 52 tests réussis |
| Aide CLI, diagnostic et état des modèles | Vérifiés |
| Tests NumPy/OpenCV du pipeline et des images | Réussis |
| Interface PySide6 | Ouverture, détection et anonymisation via workers vérifiées en mode offscreen |
| Inférence SimSwap 224 et parsing BiSeNet | Réussis sur CUDA avec les checkpoints officiels |
| Inférence SimSwap 512 et parsing BiSeNet | Réussis sur CUDA avec le checkpoint 512 beta officiel |
| Génération StyleGAN2-ADA | 5 portraits FFHQ générés et indexés |
| Intégration avec checkpoints préentraînés | 2 tests réussis (224 et 512) : remplacement ciblé, protection des autres visages et export PNG |
| Évaluation avec un reconnaisseur indépendant | Non exécutée |

Configuration vérifiée : Windows, CPython 3.10.22 x64, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, InsightFace 0.7.3, ONNX Runtime 1.18.1 et RTX 3060. InsightFace est compilé depuis ses sources officielles avec MSVC et le Windows SDK 10.0.26100. Le contrôle des dépendances `pip check` ne signale aucune incompatibilité.

Les essais d'intégration utilisent la photographie `vendor/SimSwap/demo_file/multi_people.jpg`, une référence recadrée sur la personne de gauche et un donneur FFHQ synthétique. Pour chaque résolution, trois visages sont détectés et un seul est remplacé. Les deux autres régions faciales sont conservées exactement, puis l'image est enregistrée en PNG. Les deux résultats satisfont le seuil expérimental de similarité de `0.30` après remplacement.

Le chargement du checkpoint 512 beta filtre les 162 paramètres historiques `conv_ada1`, `conv_ada2` et `bn_ada1` absents du réseau d'inférence, conformément au chargeur officiel. Ses 137 paramètres actifs sont présents et ont les dimensions attendues. Les tests de chargement refusent les poids actifs manquants, les mauvaises dimensions et les autres clés inconnues ; le chargement du modèle 224 conserve sa vérification stricte.

Les portraits FFHQ ont été générés sur CUDA avec les opérations de référence PyTorch fournies par NVIDIA ; les extensions CUDA personnalisées n'ont pas été compilées. Ces vérifications portent sur les images de démonstration et ne constituent pas une évaluation générale de la qualité ou de la confidentialité.

## Tests unitaires et interface

Installer l'environnement décrit dans le [README](README.md), puis exécuter :

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_manager -v
.\.venv\Scripts\python.exe -m pytest -m "not integration"
```

La suite comprend les vérifications suivantes :

- Absence de visage ou de correspondance, sélection de la référence et correction manuelle.
- Photographies de groupe, occurrences multiples et visages proches du seuil de reconnaissance.
- Protection des visages non sélectionnés et conservation des dimensions.
- Sélection d'identités synthétiques, incompatibilité des embeddings et contrôle après remplacement.
- Nouvelles tentatives, échec de redétection et erreurs CUDA.
- Checkpoints absents, tronqués, modifiés ou incompatibles ; provenance et chargement pickle.
- Formats d'image, orientation EXIF, export et protection des fichiers d'entrée.
- Ouverture de l'interface en mode Qt `offscreen`.

## Intégration avec les modèles préentraînés

Installer les checkpoints de détection, reconnaissance, SimSwap 224 et 512, ArcFace SimSwap et BiSeNet. Préparer ensuite :

1. Une photo de référence.
2. Une photo de groupe contenant la personne recherchée et au moins une autre personne à conserver.
3. Un portrait synthétique attesté contenant un seul visage.

Copier `tests/integration.example.json` vers `tests/integration.local.json`. Adapter les chemins des images, l'indice de référence, le nombre de correspondances attendu et les seuils. Les chemins sont relatifs au fichier JSON.

```powershell
$env:FACEANON_INTEGRATION_CONFIG = (Resolve-Path .\tests\integration.local.json).Path
.\.venv\Scripts\python.exe -m pytest -m integration -v
```

Les tests sont paramétrés pour les résolutions 224 et 512. Ils vérifient le remplacement effectif, la redétection des visages, les scores après remplacement, la préservation exacte des régions non ciblées et l'export PNG. Avec la segmentation activée, un repli géométrique fait échouer le test. Ajouter `-k 224` ou `-k 512` à la commande pour sélectionner une seule résolution.

Sans `FACEANON_INTEGRATION_CONFIG`, le test est ignoré (`SKIPPED`). Une fois configuré, un modèle manquant, une erreur d'inférence ou un contrôle refusé produit un échec.

## Évaluation avec un second reconnaisseur

Importer un autre modèle ArcFace ONNX à l'emplacement `models/evaluation/recognizer.onnx`, puis activer `independent_validation` dans la configuration d'intégration. Régler son seuil avec `independent_validation_max_similarity`.

Le second modèle extrait ses propres embeddings. Son architecture, ses poids et ses données d'entraînement doivent être choisis pour permettre une évaluation indépendante ; un hash différent ne suffit pas à établir cette indépendance.

Les seuils de similarité sont expérimentaux. L'évaluation doit inclure un contrôle visuel et plusieurs conditions de pose, de taille, d'éclairage et d'occlusion.
