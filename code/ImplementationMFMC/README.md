# FaceAnon

FaceAnon est une application de bureau pour anonymiser une personne dans une photographie en remplaçant son visage par une identité synthétique. Une photo de référence permet de retrouver la personne dans une photo de groupe et de sélectionner les occurrences à traiter.

Le traitement s'effectue localement avec **InsightFace**, **SimSwap** et **BiSeNet**. Les identités fictives proviennent d'une banque locale, importée ou générée avec **StyleGAN2-ADA**.

## Fonctionnalités

- Détection et comparaison des visages à partir d'une photo de référence.
- Sélection automatique des correspondances et correction manuelle.
- Remplacement de plusieurs occurrences d'une même personne dans une image.
- Identité synthétique aléatoire, stable pendant la session ou choisie dans la banque.
- Fusion avec segmentation faciale et contrôle de similarité après remplacement.
- Interface PySide6 avec vues avant/après, zoom, masques et progression.
- Export PNG, JPEG et WebP, et interface en ligne de commande.

## Prérequis

Le guide ci-dessous cible **Windows 11 x64**. Un GPU NVIDIA est recommandé pour le traitement ; un mode CPU plus lent est également disponible. La configuration GPU vérifiée utilise une RTX 3060 de 12 Go.

Prévoir **environ 20 Go d'espace libre pendant l'installation complète GPU** pour les outils, l'environnement Python, les téléchargements et les caches. Les checkpoints et les portraits s'ajoutent au code du dépôt.

### Logiciels externes à télécharger

| Logiciel et lien officiel | Nécessaire pour | Choix à faire |
|---|---|---|
| [Git pour Windows](https://git-scm.com/install/windows) | Cloner le dépôt | Installateur x64 ; options par défaut adaptées à une utilisation en ligne de commande |
| [Python 3.10.11](https://www.python.org/downloads/release/python-31011/) · [installateur Windows x64](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) | Exécuter FaceAnon | **Windows installer (64-bit)**, avec `pip` et le lanceur `py` |
| [Build Tools pour Visual Studio](https://visualstudio.microsoft.com/downloads/) | Compiler InsightFace | Rubrique **Tools for Visual Studio → Build Tools**, puis charge de travail C++ décrite ci-dessous |
| [Windows SDK](https://learn.microsoft.com/en-us/windows/apps/windows-sdk/downloads) | Fournir les en-têtes et bibliothèques C/C++ | À sélectionner dans Visual Studio Installer ; installateur séparé disponible si nécessaire. Version `10.0.26100` utilisée pour la validation |
| [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) · [installateur x64](https://aka.ms/vc14/vc_redist.x64.exe) | Charger les DLL de PyTorch et ONNX Runtime | Package **x64** |
| [Pilotes NVIDIA](https://www.nvidia.com/en-us/drivers/) | Utiliser le GPU | Pilote compatible avec la carte et CUDA 12.1 ; inutile pour le mode CPU |
| [Visual Studio Code](https://code.visualstudio.com/download) | Éditer le code et utiliser un terminal intégré | Facultatif ; choisir la version Windows x64 |
| [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) | Compiler les extensions CUDA de StyleGAN2 | Facultatif, réservé à ce besoin de compilation |

Les **Build Tools** fournissent le compilateur ; **Visual Studio Code** est l'éditeur. Le SDK et le redistribuable Visual C++ remplissent aussi des rôles différents : installer le redistribuable seul ne permet pas de compiler InsightFace.

### Installer Python

1. Ouvrir l'installateur **Python 3.10.11 x64** indiqué dans le tableau.
2. Cocher **Add python.exe to PATH**. Dans les options d'installation, conserver **pip** et le **Python Launcher (`py`)**.
3. Terminer l'installation, puis ouvrir un **nouveau terminal PowerShell**.
4. Vérifier l'interpréteur :

```powershell
py -3.10 --version
py -3.10 -c "import platform, sys; print(sys.executable); print(platform.architecture()[0])"
```

Le résultat doit indiquer **Python 3.10.x** et **64bit**. Les commandes suivantes sélectionnent explicitement Python 3.10, même si d'autres versions sont installées. Utiliser la distribution Windows native ; les distributions MSYS2/MinGW et les installations 32 bits ne correspondent pas au profil du projet.

Python 3.10 est conservé pour la compatibilité de ce profil de dépendances ; cette branche est en fin de support. Les versions Python plus récentes nécessitent une qualification séparée avec les modèles historiques.

### Installer les outils C++ et le SDK

Dans **Visual Studio Installer**, installer les Build Tools 2022 ou plus récents, ou choisir **Modifier** sur une installation existante. Sélectionner la charge de travail **Développement Desktop en C++** (*Desktop development with C++*).

Vérifier dans les détails de l'installation que les composants suivants sont cochés :

- **MSVC — outils de génération C++ x64/x86**.
- **Windows 11 SDK**, notamment la version `10.0.26100` si elle est proposée.

La [procédure Microsoft pour installer les outils C++](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation) détaille ces écrans. Si le SDK manque ensuite, son [installateur autonome](https://learn.microsoft.com/en-us/windows/apps/windows-sdk/downloads) permet de l'ajouter. Les installateurs peuvent demander des droits administrateur ; l'installation des paquets Python et le lancement de FaceAnon s'effectuent ensuite dans un terminal normal.

Fermer et rouvrir les terminaux après l'installation des outils. Le SDK apporte notamment `stdio.h` et les bibliothèques nécessaires à la compilation d'InsightFace.

### GPU : pilote, runtime et Toolkit

Pour le mode GPU, installer le pilote adapté depuis [NVIDIA](https://www.nvidia.com/en-us/drivers/), puis vérifier sa disponibilité :

```powershell
nvidia-smi
```

Le tableau doit afficher la carte NVIDIA. Le **runtime CUDA 12.1 est fourni avec les roues PyTorch `cu121`** installées plus bas. Le CUDA Toolkit complet n'est donc pas un prérequis pour utiliser SimSwap. Le nombre « CUDA Version » affiché par `nvidia-smi` peut être différent de `12.1` : il décrit la capacité du pilote, pas la version de la roue PyTorch.

Le Toolkit devient utile pour compiler les opérations CUDA personnalisées de StyleGAN2. Les opérations de référence PyTorch permettent également la génération, avec des performances différentes. Voir l'[explication du runtime fourni par PyTorch](https://discuss.pytorch.org/t/should-i-install-the-extra-cudatoolkit-and-cudnn/194528/2), les [versions PyTorch disponibles](https://pytorch.org/get-started/previous-versions/) et la [documentation de compatibilité des pilotes CUDA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### Versions des bibliothèques Python

| Dépendance | Version |
|---|---|
| PyTorch / torchvision | 2.2.2 / 0.17.2 |
| Runtime PyTorch CUDA | cu121 |
| NumPy | 1.26.4 |
| InsightFace | 0.7.3 |
| ONNX Runtime | 1.18.1, CPU |
| OpenCV headless | 4.10.0.84 |
| PySide6 | 6.7.3 |

Les versions sont fixées dans [requirements.txt](requirements.txt). Le profil utilise CUDA pour PyTorch et le CPU pour ONNX Runtime, ce qui permet de gérer séparément leurs dépendances CUDA/cuDNN.

## Installation

### 1. Récupérer le dépôt et ouvrir le bon dossier

Choisir un **chemin court**, par exemple `C:\src\FaceAnon`, pour éviter les limites de longueur des chemins lors de la compilation. Remplacer `URL_DU_DEPOT` par l'URL Git fournie par le bouton **Code** du dépôt :

```powershell
git clone "URL_DU_DEPOT" C:\src\FaceAnon
Set-Location C:\src\FaceAnon\code\ImplementationMFMC
```

Si le dépôt est déjà cloné, ouvrir simplement un terminal dans son dossier `code/ImplementationMFMC`. Toutes les commandes qui suivent sont à exécuter **dans ce dossier**, où se trouvent `app.py` et `requirements.txt`.

```powershell
Get-Item app.py, requirements.txt
Test-Path .\vendor\SimSwap\models\fs_networks.py
Test-Path .\vendor\stylegan2-ada-pytorch\legacy.py
```

Les deux dernières commandes doivent renvoyer `True`. Les sources tierces sont incluses dans `vendor/` ; un dossier absent indique que la copie du projet est incomplète.

**Un clone Git ne contient pas l'environnement `.venv`, les checkpoints `models/` ni la banque `identity_bank/`.** Ces éléments sont locaux à chaque installation et doivent être préparés avec les étapes ci-dessous.

### 2. Créer l'environnement Python

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 Cython==3.0.11
```

Toutes les commandes utilisent directement le Python de `.venv` : **il n'est pas nécessaire d'activer l'environnement** avec `Activate.ps1` ni de modifier la stratégie d'exécution PowerShell.

### 3. Installer PyTorch — choisir GPU ou CPU

**Avec un GPU NVIDIA**, utiliser :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

**Sans GPU NVIDIA**, utiliser à la place :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

Exécuter **une seule** de ces deux commandes. Le téléchargement de la version GPU est volumineux et peut prendre plusieurs minutes.

### 4. Installer les autres dépendances

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

InsightFace est compilé à cette étape. `--no-build-isolation` permet d'utiliser les versions NumPy et Cython préparées précédemment. Le contrôle final doit afficher `No broken requirements found.`

Si la compilation échoue, consulter la section [Dépannage de l'installation](#dépannage-de-linstallation) avant de relancer la commande. Éviter les roues InsightFace provenant de sites non officiels.

### 5. Vérifier l'installation

```powershell
.\.venv\Scripts\python.exe app.py doctor
```

Cette commande affiche les versions, les checkpoints disponibles et le résultat d'une opération CUDA. Pour la configuration GPU, `cuda_available` et `cuda_tensor_test` doivent être `true`. En mode CPU, l'absence de CUDA est normale. À ce stade, des checkpoints absents sont aussi normaux : ils seront installés à l'étape suivante.

### 6. Préparer les modèles, la banque et lancer FaceAnon

Suivre les sections **Modèles préentraînés** et **Banque d'identités synthétiques** ci-dessous. Pour un premier traitement, il faut les cinq modèles de base du profil 224 et au moins une identité synthétique admissible.

Une fois ces éléments prêts :

```powershell
.\.venv\Scripts\python.exe app.py
```

Pour les lancements suivants, ouvrir un terminal dans `code/ImplementationMFMC` et réutiliser cette commande ; aucune réinstallation n'est nécessaire.

Dans VS Code, ouvrir ce dossier, installer l'extension **Python** de Microsoft si besoin, puis utiliser **Ctrl+Maj+P → Python: Select Interpreter → Enter interpreter path** et sélectionner `.venv\Scripts\python.exe`.

## Modèles préentraînés

Les sources SimSwap et StyleGAN2-ADA sont incluses dans `vendor/`. Leurs révisions sont consignées dans [vendor/sources.json](vendor/sources.json). Les checkpoints sont à télécharger séparément depuis les sources officielles.

Le bouton **Modèles et CUDA** permet de consulter leur état, de les télécharger avec une barre de progression ou d'importer un fichier local. En ligne de commande :

```powershell
.\.venv\Scripts\python.exe app.py models download detector
.\.venv\Scripts\python.exe app.py models download recognizer
.\.venv\Scripts\python.exe app.py models download simswap224
.\.venv\Scripts\python.exe app.py models download arcface --acknowledge-pickle-risk
.\.venv\Scripts\python.exe app.py models download bisenet
```

Le modèle 512 est optionnel. Pour l'ajouter et le rendre sélectionnable dans l'interface :

```powershell
.\.venv\Scripts\python.exe app.py models download simswap512
```

Si l'application est déjà ouverte, fermer le gestionnaire de modèles ou relancer l'application après l'installation.

| Clé | Fichier local | Utilisation |
|---|---|---|
| `detector` | `models/insightface/det_10g.onnx` | Détection SCRFD, archive buffalo_l |
| `recognizer` | `models/insightface/w600k_r50.onnx` | Comparaison et validation des identités |
| `simswap224` | `models/simswap/224/latest_net_G.pth` | Générateur SimSwap 224, variante `people` |
| `arcface` | `models/simswap/arcface_checkpoint.tar` | Encodeur d'identité propre à SimSwap |
| `bisenet` | `models/parsing/79999_iter.pth` | Segmentation pour la fusion |
| `simswap512` | `models/simswap/512/550000_net_G.pth` | Générateur 512, optionnel |
| `stylegan` | `models/stylegan/ffhq.pkl` | Génération de portraits synthétiques, optionnelle |
| `independent` | `models/evaluation/recognizer.onnx` | Second modèle de validation, optionnel |

Sources officielles pour un téléchargement manuel : [InsightFace buffalo_l](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip), [SimSwap 224 / ArcFace / BiSeNet](https://github.com/neuralchen/SimSwap/releases/tag/1.0), [SimSwap 512 beta](https://github.com/neuralchen/SimSwap/releases/tag/512_beta) et [FFHQ StyleGAN2-ADA](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl). Pour les archives ZIP, extraire le fichier indiqué dans le tableau avant de l'importer ; pour le générateur 224, choisir le `latest_net_G.pth` de la variante `people`.

Pour consulter l'état des fichiers ou importer un checkpoint téléchargé manuellement :

```powershell
.\.venv\Scripts\python.exe app.py models status
.\.venv\Scripts\python.exe app.py models import simswap224 --file .\downloads\latest_net_G.pth
.\.venv\Scripts\python.exe app.py models import arcface --file .\downloads\arcface_checkpoint.tar --acknowledge-pickle-risk
```

Les checkpoints ArcFace SimSwap et FFHQ utilisent le format pickle, qui peut exécuter du code au chargement. Leur import nécessite une confirmation ; utiliser uniquement les fichiers des sources officielles. Le registre `models/provenance.json` conserve les hashes locaux pour détecter les modifications ultérieures. Ces hashes ne remplacent pas une somme de contrôle fournie par l'éditeur.

Le checkpoint SimSwap 512 beta contient 162 paramètres historiques de couches absentes du réseau d'inférence officiel. L'adaptateur écarte uniquement ces clés connues, comme le prévoit le [chargeur SimSwap](https://github.com/neuralchen/SimSwap/blob/main/models/base_model.py). Les poids actifs restent chargés strictement : un paramètre manquant, une dimension incorrecte ou une autre clé inconnue entraîne un refus du checkpoint.

## Banque d'identités synthétiques

Une banque contenant au moins une identité admissible est nécessaire au remplacement. Chaque portrait doit représenter un seul visage fictif.

Choisir **l'une des deux méthodes** suivantes. La génération StyleGAN2 permet de préparer sa propre banque si aucun dossier de portraits synthétiques n'est disponible. Quelques portraits suffisent pour commencer ; plusieurs candidats augmentent les possibilités de sélection et de nouvelles tentatives.

### Importer des portraits existants

```powershell
.\.venv\Scripts\python.exe app.py bank import --folder .\portraits_synthetiques --confirm-synthetic
.\.venv\Scripts\python.exe app.py bank list
```

L'import demande de confirmer l'origine synthétique des images. Les portraits, leurs embeddings et leur provenance sont conservés dans `identity_bank/`. Après un changement de reconnaisseur ou une modification des images, recalculer les embeddings :

```powershell
.\.venv\Scripts\python.exe app.py bank reindex
```

### Générer une banque avec StyleGAN2-ADA

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements-stylegan.txt
.\.venv\Scripts\python.exe app.py models download stylegan --acknowledge-pickle-risk
.\.venv\Scripts\python.exe app.py bank generate --start-seed 0 --count 30 --device auto
```

La banque générée est réutilisable sans relancer StyleGAN2. Les seeds sont enregistrées pour la reproductibilité. Les extensions CUDA de StyleGAN2 peuvent nécessiter le CUDA Toolkit 12.1 et MSVC ; `--device cpu` permet d'utiliser le mode CPU.

L'import et la génération sont aussi accessibles depuis la fenêtre principale.

## Utilisation

```powershell
.\.venv\Scripts\python.exe app.py
```

1. Dans **Entrées**, choisir une photo de référence et la photographie à anonymiser.
2. Cliquer sur **Détecter les visages**. Si la référence contient plusieurs personnes, sélectionner celle recherchée.
3. Vérifier les correspondances : les cadres verts et les cases cochées indiquent les visages sélectionnés. Cliquer sur les cadres ou les cases pour ajuster la sélection.
4. Dans **Anonymisation**, choisir le mode d'identité, régler les seuils si nécessaire, puis cliquer sur **Anonymiser**.
5. Examiner le résultat dans la vue **Avant / Après**. La molette permet de zoomer ; le glisser permet de déplacer l'image. Les cadres et les masques peuvent être affichés séparément.
6. Cliquer sur **Enregistrer sous** pour exporter le résultat.

L'orientation EXIF est appliquée au chargement. L'export conserve les dimensions et le cadrage ainsi obtenus, retire les métadonnées et empêche l'écrasement des photos d'entrée. Utiliser **PNG** pour conserver les pixels sans perte ; JPEG et WebP peuvent introduire des changements liés à la compression.

Une même identité fictive est utilisée pour toutes les occurrences sélectionnées dans une photo. Le mode **stable pendant la session** conserve cette association en mémoire. **Nouvelle identité et nouvel essai** efface l'association et relance le traitement avec un choix aléatoire.

Après remplacement, les visages sont redétectés et comparés à leur identité originale. Si aucun essai ne satisfait les contrôles, l'original est conservé. Le bouton **Masquage opaque de secours** permet alors d'appliquer un masquage aux seuls visages sélectionnés.

## Ligne de commande

```powershell
.\.venv\Scripts\python.exe app.py anonymize --reference .\photos\reference.jpg --image .\photos\groupe.jpg --output .\outputs\groupe.png
```

Pour choisir la personne de référence et les visages cibles manuellement :

```powershell
.\.venv\Scripts\python.exe app.py anonymize --reference .\photos\reference.jpg --reference-index 1 --image .\photos\groupe.jpg --output .\outputs\groupe.png --select 0 3 --attempts 3 --device auto
```

Les indices CLI commencent à **0**. Sans `--select`, le traitement utilise les correspondances automatiques. Si la référence contient plusieurs visages, `--reference-index` est obligatoire.

| Option | Description |
|---|---|
| `--mode manual --identity IDENTIFIANT` | Choisir une identité de la banque |
| `--device auto`, `cpu` ou `cuda` | Sélectionner le mode de calcul PyTorch |
| `--size 224` ou `512` | Choisir la résolution SimSwap |
| `--match-threshold` | Seuil de correspondance avec la référence |
| `--candidate-max` | Similarité maximale du donneur synthétique |
| `--validation-max` | Similarité maximale après remplacement |
| `--fallback mask` | Autoriser un masquage de secours en cas d'échec |

La CLI produit un compte rendu JSON sans embeddings. Les codes de retour sont `0` pour un succès, `2` pour une erreur de configuration ou de dépendance, `3` en l'absence de résultat validé et `4` pour un export de secours. Consulter `python app.py anonymize --help` pour toutes les options.

## Tests et validation

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_manager -v
.\.venv\Scripts\python.exe -m pytest -m "not integration"
```

Les tests unitaires utilisent des backends simulés pour vérifier le ciblage, la fusion, la sélection des identités et la gestion des erreurs. Le test d'interface utilise Qt en mode `offscreen`.

Les tests d'intégration couvrent SimSwap 224 et 512. Ils nécessitent leurs checkpoints, une photo de référence, une photo de groupe et un portrait synthétique. Copier [tests/integration.example.json](tests/integration.example.json) vers `tests/integration.local.json`, adapter les chemins et le nombre de correspondances attendu, puis lancer :

```powershell
$env:FACEANON_INTEGRATION_CONFIG = (Resolve-Path .\tests\integration.local.json).Path
.\.venv\Scripts\python.exe -m pytest -m integration -v
```

Les chemins d'images sont relatifs au fichier JSON. Ajouter `-k 224` ou `-k 512` à la commande pour tester une seule résolution. Sans configuration, les tests d'intégration sont ignorés (`SKIPPED`). Les procédures détaillées et l'état des vérifications sont décrits dans [VALIDATION.md](VALIDATION.md).

## Limites et confidentialité

Les seuils par défaut (`0.45` pour la correspondance, `0.30` pour le donneur et la validation) sont expérimentaux et doivent être calibrés sur les images traitées. Une faible similarité ne garantit pas l'anonymat : les vêtements, le corps ou le contexte peuvent encore permettre une réidentification. Les petits visages, profils prononcés, occlusions et éclairages difficiles nécessitent une vérification visuelle.

Les régions des visages détectés non sélectionnés sont exclues de la fusion. Une détection ou une sélection erronée peut toutefois affecter le résultat. Si BiSeNet est indisponible, un masque géométrique est utilisé avec un avertissement sur la qualité.

Les photos sont traitées localement et les embeddings des personnes réelles restent en mémoire. Les accès réseau servent au téléchargement des modèles. Les embeddings persistants concernent uniquement les portraits de la banque synthétique.

Pour une évaluation supplémentaire, importer un reconnaisseur ArcFace ONNX distinct avec `models import independent --file CHEMIN`, puis utiliser `--independent-model --independent-validation-max SEUIL`. Le second modèle recalcule ses propres embeddings et son seuil doit être calibré séparément.

## Dépannage de l'installation

| Symptôme | Vérification et solution |
|---|---|
| `py` introuvable ou Python 3.10 absent | Réouvrir un terminal après l'installation de Python. Relancer l'installateur Python 3.10 x64 et vérifier le composant **Python Launcher**. `py -0p` liste les interpréteurs disponibles. |
| `No module named pip` | Vérifier `py -3.10 --version`, puis ajouter **pip** via l'option **Modify** de l'installateur Python. Recréer l'environnement avec cet interpréteur. |
| `Microsoft Visual C++ 14.0 or greater is required`, compilateur introuvable | Dans Visual Studio Installer, vérifier la charge **Développement Desktop en C++** et les outils **MSVC x64/x86**. Réessayer depuis un **Developer PowerShell for VS** après installation. |
| `fatal error C1083` mentionnant `stdio.h`, `windows.h` ou un en-tête système | Ajouter le **Windows SDK**, puis rouvrir le terminal de développement. Le redistribuable Visual C++ ne fournit pas ces en-têtes. |
| `C1083` sur un fichier généré ou `.obj`, `Invalid argument`, chemin trop long | Cloner le dépôt dans un chemin court tel que `C:\src\FaceAnon`. Un chemin de compilation de plus de 260 caractères peut bloquer MSVC. Recréer `.venv` si le dépôt a été déplacé. |
| `DLL load failed`, `WinError 126`, erreur au chargement de `c10_cuda.dll` | Vérifier le **Visual C++ Redistributable x64**, la distribution Python x64 et le pilote NVIDIA. Réouvrir le terminal, puis relancer `app.py doctor`. |
| CUDA indisponible malgré une carte NVIDIA | Vérifier d'abord `nvidia-smi`, puis le résultat de `app.py doctor`. Une roue PyTorch CPU ne fournit pas CUDA ; installer la variante `cu121` dans le bon environnement. |
| `ModuleNotFoundError` alors que les paquets semblent installés | Utiliser systématiquement `.\.venv\Scripts\python.exe` depuis le dossier du projet. Dans VS Code, sélectionner le même interpréteur. |
| Erreur indiquant une incompatibilité NumPy ou une extension compilée pour une autre version | Vérifier que NumPy est bien en version **1.26.4**, puis reconstruire InsightFace avec ce NumPy et `--no-build-isolation`. Éviter une mise à jour globale non contrôlée des dépendances. |
| Checkpoint absent, refusé ou page HTML téléchargée | Utiliser le gestionnaire de modèles ou les liens officiels de la section **Modèles préentraînés**, puis importer le vrai fichier de poids extrait. Pour ArcFace/FFHQ, passer par l'import avec confirmation du chargement pickle. |
| SimSwap 512 non sélectionnable | Installer `simswap512`, puis fermer le gestionnaire de modèles ou relancer l'application pour actualiser la liste. |
| `Aucune identité synthétique admissible` | Consulter `app.py bank list`. Préparer une banque si elle est vide ; ajouter d'autres portraits synthétiques si les candidats sont trop proches de la personne ciblée. |
| Mémoire CUDA insuffisante | Libérer la VRAM occupée par d'autres programmes, essayer SimSwap 224 ou choisir **CPU** dans l'interface. |

Pour vérifier les outils dans un terminal **Developer PowerShell for VS** ouvert depuis le menu Démarrer :

```powershell
Get-Command cl
$env:WindowsSdkDir
$env:WindowsSDKVersion
```

La première commande doit retrouver le compilateur Microsoft ; les deux suivantes doivent indiquer le dossier et la version du SDK. Revenir ensuite dans `code/ImplementationMFMC` et reprendre l'installation des dépendances.

Avant de signaler un problème, joindre le message d'erreur complet et le résultat de :

```powershell
.\.venv\Scripts\python.exe app.py doctor
.\.venv\Scripts\python.exe -m pip check
```

## Structure du projet

```text
app.py                      Point d'entrée graphique et CLI
src/gui/                    Fenêtre, vues, workers et gestionnaire de modèles
src/core/                   Pipeline, ciblage, banque, fusion et validation
src/backends/               Adaptateurs InsightFace, SimSwap et StyleGAN2
src/utils/                  Configuration, images et gestion des checkpoints
vendor/                     Sources et licences des projets tiers
models/                     Checkpoints locaux
identity_bank/              Portraits synthétiques et embeddings
tests/                      Tests unitaires, interface et intégration
```

Le pipeline est utilisable sans interface graphique via `ProcessingService().pipeline(Settings())`, `analyze(...)` et `run(...)`. Les interfaces `Recognizer` et `FaceSwapper` séparent le traitement des backends. L'ArcFace ONNX utilisé pour la reconnaissance et l'ArcFace historique attendu par SimSwap conservent chacun leur prétraitement et leurs embeddings.

## Références et licences

- [My Face My Choice — WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Ciftci_My_Face_My_Choice_Privacy_Enhancing_Deepfakes_for_Social_Media_WACV_2023_paper.pdf)
- [SimSwap](https://github.com/neuralchen/SimSwap) et [préparation des modèles](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md)
- [InsightFace](https://github.com/deepinsight/insightface)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)

Les modèles préentraînés et les sources tierces ont leurs propres conditions d'utilisation, notamment des restrictions non commerciales. Consulter [LICENSES.md](LICENSES.md) avant utilisation ou redistribution.
