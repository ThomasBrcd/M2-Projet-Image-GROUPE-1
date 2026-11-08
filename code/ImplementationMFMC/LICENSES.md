# Modèles, sources et conditions d'utilisation

Les checkpoints ne sont pas redistribués avec FaceAnon. Leur téléchargement et leur utilisation restent soumis aux conditions des auteurs. L'application ne donne aucune autorisation commerciale supplémentaire.

| Élément | Provenance | Conditions/limitations à vérifier |
|---|---|---|
| SimSwap | [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap) | Le dépôt contient une licence **CC BY-NC 4.0** ; certains fichiers des réseaux portent aussi une mention **CC BY-NC-SA 4.0**. Conserver les mentions et respecter les conditions des fichiers concernés. |
| Générateur 224/512 et ArcFace SimSwap | Releases officielles SimSwap | Usage et redistribution selon les conditions des auteurs ; recherche/non commercial. Le module ArcFace historique est un pickle exécutable. |
| InsightFace, code Python | [deepinsight/insightface](https://github.com/deepinsight/insightface) | Code sous licence **MIT** ; cela ne rend pas ses poids librement utilisables commercialement. |
| buffalo_l / SCRFD / ArcFace ONNX | Model zoo officiel InsightFace | Poids préentraînés annoncés pour **recherche non commerciale uniquement**, téléchargement manuel ou automatique. |
| BiSeNet face parsing | Code inclus dans SimSwap, dérivé de [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | Préserver les mentions amont et vérifier les conditions du code, des poids et des données d'entraînement séparément. |
| StyleGAN2-ADA | [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) | **NVIDIA Source Code License**, restrictions non commerciales/recherche ; texte intégral conservé dans le snapshot. |
| FFHQ préentraîné | CDN NVIDIA lié par le dépôt officiel StyleGAN2-ADA | Vérifier la licence du checkpoint et les conditions liées aux données ; un portrait généré peut toujours ressembler à une personne réelle. |
| PySide6/Qt | [Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html) | LGPLv3/GPLv3 ou licence commerciale suivant les composants et la distribution ; examiner les obligations de redistribution. |
| Autres dépendances | Projets identifiés par requirements.txt | NumPy, PyTorch, OpenCV, ONNX Runtime, Pillow, etc. conservent leurs propres licences et notices. |

Les sources vendoriées sont conservées sans modification ; les adaptations FaceAnon résident dans `src/backends/`. Les révisions utilisées sont dans [vendor/sources.json](vendor/sources.json). Les notices complètes disponibles sont notamment [SimSwap/LICENSE](vendor/SimSwap/LICENSE) et [StyleGAN2-ADA/LICENSE.txt](vendor/stylegan2-ada-pytorch/LICENSE.txt).

La banque accepte uniquement des images déclarées synthétiques ou produites par FFHQ. L'attestation utilisateur est une information de provenance, pas une preuve algorithmique. Les images de démonstration réelles incluses dans le dépôt SimSwap ne sont jamais automatiquement importées comme donneurs.

Les embeddings et scores sont des indicateurs expérimentaux. FaceAnon ne certifie pas l'anonymat d'une photo et ne supprime pas les indices contextuels permettant une réidentification. Choisir des images pour lesquelles le traitement est autorisé et respecter les droits des personnes concernées.
