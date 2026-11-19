// Compilation depuis le dossier ImplementationMFMC :
// typst compile rapport/rapport_faceanon.typ rapport/rapport_faceanon.pdf
// Document autonome : aucun paquet Typst ni fichier image externe requis.

#let ink = rgb("#183447")
#let teal = rgb("#147d74")
#let muted = rgb("#566b78")
#let pale = rgb("#edf5f5")
#let rule = rgb("#d5e1e5")

#set document(title: "Rapport semaine 2", author: "Projet M2")
#set page(
  paper: "a4",
  margin: (top: 21mm, bottom: 19mm, left: 19mm, right: 19mm),
  numbering: "1 / 1",
  header: text(size: 8pt, fill: muted)[Groupe 1 #h(1fr) M2 · Projet image],
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

#text(size: 27pt, weight: "bold")[Rapport d'avancement]
#v(3pt)
#text(size: 15pt)[Semaine 2]
#v(3pt)
#text(size: 9pt, fill: muted)[15 novembre 2026]
#v(10pt)

#note[
  *Résumé.* L'application produite la semaine dernière, nous a fourni un pipeline stable, qui remplit l'objectif du projet. Mais le projet ne s'arrête pas là. Nous avons donc trouvé une piste qui nous permet de modifier, et d'améliorer l'un des modèles du pipeline.
]

= Objectifs disponibles et notre Choix
  Le deeplearning, appliqué à l'image, demande souvent de gros modèles, entrainé sur des jeux de donnée massif (expliquant la suggestion de la part de nos encadrant, concernant le fait de ne *pas* entrainer un modèle _from scratch_).

  De plus, le modèle le plus proche d'un GAN (SimSwap), fût entrainé sur un jeu de donné massif, comparé au dataset 'lamsimon/celebahq'[kaggle], qui ne contient que 30k images, ce qui rend irréaliste une éventuelle amélioration qualitative du modèle, étant donné nos contraintes matérielles.

  Cependant, il y à une chose que l'on peut faire, c'est utiliser ce modèle comme base. Et cela donne accès à deux voies de développement.

  - La distilation : qui revient à aligner un modèle sur le modèle que nous avons, en s'assurant que le modèle de destination est plus compact que le modèle d'origine.

  - La feature extraction : qui revient à extraire le modèle et l'inserer dans un modèle plus généraliste.

  La deuxième approche, très intérésante pour produire des modèles plus généralistes, n'est pas pertinente ici.
  
  Nous sommes parti sur la piste de la distilation, tout d'abbord en planifiant le format du nouveau modèle, puis en préparant le dataset qui va permettre de sampler SimSwap.


= Contraintes de format
  Pour permettre cette approche, il nous faut impérativement respecter le prétraitement de notre pipeline, et collecter les générations de simSwap.
  #figure(image("pipeline.png", width: 25%, height: 25%), caption: "Shema du routage des flux de données")
  Donc il nous faut des triplets (X0, X1, Y), tel que X0: image à éditée alignée ffqh, X1 : identité synthétique à greffer, Y : image éditée.

  Cela évite de "bruteforce" les I/O du modèle pour constituer un dataset. Cependant, pour une question de précision de convergence, un bruteforce serait plus précis.

  Un notebook jupyter à été écrit pour repliquer le prétraitement, et le faire traiter par SimSwap512, et enregistrer les sorties. Nous nous retrouvons avec un dataset de 150 Go (approx.), contenant 59999 triplets.
  #figure(image("pre-dataset.png", width: 25%, height: 25%), caption: "Echantillon de quelques triplets de notre dataset enrichi")

= Modèle et entrainement
  #figure(image("archi.png", width: 25%, height: 25%), caption: "Architecture de notre modèle")
  Nous partons sur 50 epoch, avec checkpoint automatique (pour entrainer en plusieurs sessions et ne pas craindre les crashs), et un early stop pour ne pas entrainer le modèle une fois un mimimum local atteint.

  Nous utilisons torch-cuda, et les estimations du temps d'entrainement sont de 1h30/epoch (approx.), et donc 3j (approx) d'entrainement pour atteindre le seuil max d'entrainement.

  Attention ceci dit. Comme souvent en deeplearning, rien ne peux garantir que le dataset est assez gros pour sampler suffisament le modèle maitre, et rien ne garanti non plus des performances aussi bonne.