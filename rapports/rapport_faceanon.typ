// Compilation depuis le dossier ImplementationMFMC :
// typst compile rapport/rapport_faceanon.typ rapport/rapport_faceanon.pdf
// Document autonome : aucun paquet Typst ni fichier image externe requis.

#let ink = rgb("#183447")
#let teal = rgb("#147d74")
#let muted = rgb("#566b78")
#let pale = rgb("#edf5f5")
#let rule = rgb("#d5e1e5")

#set document(title: "FaceAnon — Rapport de réalisation", author: "Projet M2")
#set page(
  paper: "a4",
  margin: (top: 21mm, bottom: 19mm, left: 19mm, right: 19mm),
  numbering: "1 / 1",
  header: text(size: 8pt, fill: muted)[FaceAnon #h(1fr) Projet M2 · Traitement d’images],
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "fr", fill: ink)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.")
#show heading: set text(fill: teal)
#show raw: set text(size: 9pt)
#set list(indent: 12pt, body-indent: 5pt)

#let note(body) = block(width: 100%, inset: 10pt, radius: 4pt, fill: pale, body)
#let step(n, title, description) = block(
  width: 100%, inset: 8pt, radius: 3pt, fill: pale, stroke: 0.4pt + rule,
)[
  #text(size: 9.5pt, weight: "bold", fill: teal)[#n — #title]
  #v(2pt)
  #text(size: 9pt)[#description]
]

#text(size: 27pt, weight: "bold")[FaceAnon]
#v(3pt)
#text(size: 15pt)[Rapport de réalisation de l’application]
#v(3pt)
#text(size: 9pt, fill: muted)[8 novembre 2026]
#v(10pt)

#note[
  *Résumé.* FaceAnon est une application de bureau permettant de remplacer les
  occurrences d’une personne dans une photographie par une identité fictive.
  Le travail réalisé porte sur l’intégration de réseaux préentraînés, le ciblage
  des visages, la gestion des identités de remplacement, la fusion dans l’image
  et la validation du résultat. Une interface graphique et une interface en ligne
  de commande donnent accès au même pipeline local.
]

= Objectif et périmètre

Le point de départ est une photographie de référence et une photographie à
traiter, éventuellement composée de plusieurs personnes. L’application doit
retrouver la personne recherchée et modifier les seuls visages sélectionnés.
Cette démarche s’inscrit dans l’approche de remplacement d’identité illustrée
par _My Face My Choice_ ; elle ne constitue pas une reproduction de ses résultats
scientifiques.

La réalisation repose sur l’assemblage et l’adaptation de modèles publiés.
Le traitement des photographies reste local et porte sur des images fixes.

= Chaîne de traitement

#grid(
  columns: (1fr, 1fr, 1fr), gutter: 6pt,
  step(1, "Entrées", "Lecture des images et application de l’orientation EXIF."),
  step(2, "Ciblage", "Détection, landmarks et comparaison avec la référence."),
  step(3, "Identité", "Choix d’un portrait dans la banque synthétique locale."),
  step(4, "Génération", "Alignement du visage et remplacement avec SimSwap."),
  step(5, "Fusion", "Segmentation, contours progressifs et réinsertion à la résolution d’origine."),
  step(6, "Validation", "Nouvelle détection, comparaison d’identité et export si les contrôles passent."),
)

La reconnaissance utilise la similarité cosinus entre embeddings normalisés.
Le seuil est configurable et la sélection automatique peut être corrigée
manuellement. Une référence contenant plusieurs visages demande un choix
explicite. Plusieurs occurrences de la même personne peuvent être remplacées
dans une seule photographie.

Les candidats synthétiques trop proches de l’identité originale sont écartés.
La sélection peut être aléatoire, stable pendant la session ou manuelle. Un
même donneur est utilisé pour toutes les occurrences sélectionnées d’une photo.

= Fonctions accessibles à l’utilisateur

- Chargement de photographies PNG, JPEG et WebP, avec aperçu et sélection des visages.
- Réglages de reconnaissance, de validation, de segmentation et de résolution 224/512.
- Vue avant/après avec zoom, déplacement, cadres et masques facultatifs.
- Gestion des checkpoints, import ou génération de la banque et export sans écraser les entrées.

#pagebreak()

= Architecture et choix techniques

Le code est organisé par responsabilités. `src/gui/` contient la fenêtre PySide6,
les vues d’images et les workers. `src/core/` regroupe la logique de traitement,
la banque, la fusion et la validation. `src/backends/` adapte les réseaux officiels,
tandis que `src/utils/` gère la configuration, les fichiers et les checkpoints.
Le point d’entrée `app.py` lance l’interface ou les commandes automatisées.

#block[
  #set text(size: 9.5pt)
  #table(
    columns: (1.15fr, 1fr, 2.1fr),
    inset: 5pt,
    stroke: 0.4pt + rule,
    fill: (x, y) => if y == 0 { pale } else { none },
    table.header([*Composant*], [*Profil retenu*], [*Responsabilité*]),
    [Python / PySide6], [3.10 / 6.7.3], [Application de bureau et interface réactive.],
    [InsightFace / ONNX], [0.7.3 / 1.18.1], [Détection, landmarks, reconnaissance et contrôle final.],
    [PyTorch / CUDA], [2.2.2 / cu121], [Inférence GPU, modèles en mode évaluation et sans gradients.],
    [SimSwap], [224 et 512 beta], [Transfert d’identité à partir des poids officiels.],
    [BiSeNet], [Parsing facial], [Masque limitant la fusion aux régions faciales utiles.],
    [StyleGAN2-ADA], [Checkpoint FFHQ], [Production indépendante de portraits synthétiques réutilisables.],
  )
]

Une séparation essentielle concerne les deux encodeurs d’identité : les
embeddings ONNX servent au ciblage et à la validation, alors que l’ArcFace
historique de SimSwap conditionne le générateur. Chacun conserve son propre
prétraitement. Leurs embeddings ne sont pas interchangeables.

Les modèles sont chargés une fois puis réutilisés. Les opérations coûteuses
s’exécutent dans des workers ; la progression et les résultats sont transmis au
thread graphique par des signaux. Le pipeline reste utilisable sans Qt pour les
tests et les traitements en ligne de commande.

= Qualité de fusion et gestion des échecs

Le visage généré est réinséré à l’aide de la transformation inverse de
l’alignement. Le masque BiSeNet est adouci et une correction photométrique
modérée facilite le raccord. Les cadres des visages détectés non sélectionnés
sont exclus de la fusion. La photographie complète conserve ses dimensions
après correction de l’orientation EXIF.

Un échec de segmentation déclenche un masque géométrique accompagné d’un
avertissement. Après remplacement, chaque visage ciblé doit être redétecté et
satisfaire le contrôle d’identité. Plusieurs donneurs peuvent être essayés. Si
aucun résultat complet n’est admissible, l’original est conservé et un masquage
opaque de secours peut être demandé explicitement.

= Difficultés techniques résolues

*Installation Windows.* La compilation d’InsightFace a nécessité les outils
C++ de Microsoft et le Windows SDK. Les chemins de compilation trop longs ont
été raccourcis. Le profil de dépendances est fixé et le README décrit les
installateurs officiels, les composants nécessaires et les diagnostics CPU/GPU.

*Compatibilité SimSwap 512.* Le checkpoint beta contient 162 paramètres
historiques absents du réseau d’inférence officiel. L’adaptateur filtre uniquement
ces clés connues, conformément au chargement amont. Les 137 paramètres actifs
sont présents et compatibles ; les poids manquants, les mauvaises dimensions
et les autres clés inconnues restent refusés.

*Robustesse des fichiers et de la détection.* Les checkpoints sont publiés avec
les droits du dossier de destination sous Windows. Un second passage avec
marges améliore la détection des portraits très serrés, avec retour des coordonnées
dans l’image originale.

#pagebreak()

= Vérifications et résultats

Les résultats consignés dans `VALIDATION.md` distinguent les tests avec backends
simulés et les essais réellement exécutés avec les poids préentraînés.

#block[
  #set text(size: 9.5pt)
  #table(
    columns: (1.25fr, 2.75fr),
    inset: 6pt,
    stroke: 0.4pt + rule,
    fill: (x, y) => if y == 0 { pale } else { none },
    table.header([*Vérification*], [*Résultat enregistré*]),
    [Tests unitaires et interface], [52 tests réussis : ciblage, banque, fusion, formats d’image, checkpoints et erreurs.],
    [Intégration réelle], [2 tests réussis, pour SimSwap 224 et 512, avec BiSeNet et export PNG.],
    [Interface graphique], [Ouverture, détection et anonymisation via workers vérifiées en mode offscreen.],
    [Génération synthétique], [5 portraits FFHQ générés sur CUDA puis indexés dans la banque.],
    [Environnement], [RTX 3060, CPython 3.10.22, PyTorch 2.2.2+cu121 ; contrôle des dépendances sans incompatibilité signalée.],
  )
]

Les essais d’intégration utilisent une photographie de démonstration du dépôt
SimSwap contenant trois personnes, une référence recadrée sur la personne de
gauche et un portrait FFHQ. Pour les deux résolutions, un visage est remplacé ;
les régions des deux autres visages sont conservées exactement. Les résultats
satisfont le seuil expérimental de similarité de 0,30 et l’export PNG est relu
pour vérifier la sauvegarde.

Les tests unitaires couvrent notamment les images sans visage ou sans
correspondance, les occurrences multiples, les faux positifs proches du seuil,
la sélection manuelle, les grandes images, les poids incompatibles et les échecs
CUDA. Ils vérifient la logique du logiciel ; ils ne mesurent pas la qualité
perceptuelle du générateur sur une population représentative.

= Limites et perspectives

Les seuils par défaut — 0,45 pour reconnaître une correspondance et 0,30 pour
filtrer le donneur ou contrôler le résultat — restent expérimentaux. Une faible
similarité d’embeddings ne constitue pas une garantie d’anonymat : le contexte,
les vêtements et d’autres indices peuvent permettre une réidentification.

La validation réelle porte sur des images de démonstration. Une évaluation plus
large doit inclure des tailles, poses, éclairages et occlusions variés, ainsi
qu’un contrôle visuel des raccords. L’intégration d’un second reconnaisseur est
prévue dans le pipeline, mais cette évaluation indépendante n’a pas encore été
exécutée. Les extensions CUDA personnalisées de StyleGAN2 n’ont pas été
compilées ; les portraits ont été obtenus avec les opérations de référence
PyTorch fournies par NVIDIA.

Les checkpoints doivent provenir de sources fiables ; certains chargements
utilisent le format pickle. Les conditions d’utilisation des modèles, dont les
restrictions non commerciales, sont regroupées dans `LICENSES.md`.

= Bilan

Le travail réalisé aboutit à une application locale réunissant interface,
reconnaissance ciblée, remplacement 224/512, banque d’identités et contrôle du
résultat. Les principales contributions logicielles sont l’intégration cohérente
des modèles, la protection des visages non sélectionnés, la gestion explicite
des échecs et la reproductibilité de l’installation. La suite du travail porte
sur l’élargissement des essais et la mesure indépendante de la confidentialité.

#v(5pt)
#text(size: 9pt, weight: "bold")[Sources et traçabilité]
#parbreak()
#text(size: 8.5pt)[
  Code et résultats du projet : `src/`, `tests/`, `README.md`, `VALIDATION.md` et
  `vendor/sources.json`. Implémentations de référence :
  #link("https://github.com/neuralchen/SimSwap")[SimSwap],
  #link("https://github.com/deepinsight/insightface")[InsightFace] et
  #link("https://github.com/NVlabs/stylegan2-ada-pytorch")[StyleGAN2-ADA].
  Référence scientifique :
  #link("https://openaccess.thecvf.com/content/WACV2023/html/Ciftci_My_Face_My_Choice_Privacy_Enhancing_Deepfakes_for_Social_Media_WACV_2023_paper.html")[_My Face My Choice_, WACV 2023].
]
