# Audit du dépôt — Détection explicable d'accidents et du risque routier

## Périmètre audité

Cet audit compare le dépôt avec le rapport de projet fourni. Le dépôt implémente une chaîne prototype en quatre scripts :

1. `preprocess_traffic_video.py` : détection/suivi YOLOv8, extraction des variables, règles de risque et boîte noire.
2. `view_processed_video.py` : relecture interactive d'une vidéo et du JSON prétraité.
3. `export_processed_video.py` : export d'une vidéo annotée.
4. `generate_dashboard.py` : agrégation temporelle, explications textuelles et dashboard HTML.

## Correspondance avec le rapport

### Points bien couverts

- **YOLOv8 / Ultralytics** : le prétraitement utilise bien `YOLO(args.model)` avec `model.track(...)`, un modèle par défaut `yolov8n.pt`, `conf=0.4`, `iou=0.45` et `imgsz=640`.
- **Classes routières filtrées** : seules les classes `car`, `truck`, `bus` et `motorcycle` sont conservées.
- **Caméra fixe et mesures 2D** : les métriques sont exprimées en pixels, cohérentes avec une approche non calibrée.
- **Variables géométriques** : boîte, centre, point bas-centre, trajectoire par identifiant.
- **Variables dynamiques** : vitesse, accélération, cap et changement d'angle.
- **Variables relationnelles** : distance au plus proche véhicule, vitesse relative et IoU de contact.
- **Règles explicites** : les seuils décrits dans le rapport sont présents sous forme de constantes lisibles.
- **Séparation prétraitement / exploitation** : le JSON prétraité est réutilisé par le viewer, l'export vidéo et le dashboard.
- **Boîte noire** : un historique d'environ cinq secondes est enregistré autour des accidents confirmés.
- **Dashboard explicable** : le dashboard agrège les fenêtres temporelles, produit une explication par règles et affiche un graphe causal simplifié.

### Écarts ou limites importantes

1. **Le score affiché n'est pas calibré statistiquement**  
   Le dashboard l'appelle `risk_probability`, mais la formule est heuristique. Le rapport précise ce point, mais l'interface peut être interprétée comme une vraie probabilité. Il serait plus prudent d'afficher « score normalisé » ou « indice de risque ».

2. **La causalité reste descriptive**  
   Le graphe causal repose sur une structure fixe et des corrélations simples entre fenêtres. Ce n'est pas une découverte causale ni une estimation d'effet causal. Le rapport le reconnaît, mais le dashboard devrait rendre cette limite très visible.

3. **Pas d'évaluation quantitative**  
   Le dépôt ne contient pas de protocole de test sur TU-DAT, pas d'annotations de référence, pas de métriques de faux positifs/faux négatifs, ni de script d'évaluation. C'est la plus grande limite pour soutenir des claims de détection.

4. **Mesures sensibles à la perspective**  
   Les seuils en pixels peuvent fonctionner sur une vidéo donnée, mais ils ne se généralisent pas automatiquement à d'autres angles de caméra, résolutions ou profondeurs de scène. Une homographie ou une normalisation par zone améliorerait la robustesse.

5. **Suivi dépendant des identifiants YOLO/ByteTrack**  
   Les vitesses, accélérations et historiques supposent que les IDs restent stables. Après occlusion ou choc, les IDs peuvent changer ; le code tente une persistance d'accident par proximité, mais il n'y a pas de ré-identification robuste.

6. **Contact par IoU fragile**  
   Une IoU positive entre boîtes 2D peut indiquer un vrai contact, une occlusion, une file de véhicules vue en perspective ou une erreur de détection. Le code ajoute une règle anti-file (`is_queueing_perspective_case`), mais cela reste heuristique.

7. **Données lourdes non versionnées de manière structurée**  
   Le README montre des commandes avec `processed/...`, mais le dépôt ne définit pas de structure de données de sortie, d'exemple JSON minimal, ni de schéma documenté.

8. **Peu de garde-fous d'exécution**  
   Les scripts supposent que les JSON ont la bonne structure. En cas de fichier vide, incomplet ou sans fenêtre agrégée, certaines vues du dashboard peuvent échouer côté JavaScript.

## Problème corrigé pendant l'audit

`extract_frame_data(...)` retournait seulement `vehicles` quand aucune boîte suivie n'était disponible (`boxes is None`, zéro boîte ou pas d'ID). Or `main()` attend toujours deux valeurs : `vehicles, newly_confirmed_ids`. Sur une frame sans tracking YOLO, le script pouvait donc lever une erreur de dépaquetage. La fonction retourne maintenant toujours le tuple `(vehicles, newly_confirmed_ids)` avec une liste vide d'accidents confirmés dans ce cas.

## Recommandations prioritaires

### Priorité haute

- Ajouter un script `evaluate.py` capable de comparer les alertes aux annotations TU-DAT : timestamp/frame d'accident, tolérance temporelle, précision, rappel, F1, faux positifs par minute.
- Renommer `risk_probability` ou ajouter un libellé très explicite indiquant qu'il s'agit d'un score heuristique non calibré.
- Documenter le format JSON produit : champs vidéo, champs frame, champs véhicule et format de la boîte noire.
- Ajouter des tests unitaires pour les fonctions déterministes : distance, IoU, changement d'angle, score de risque, agrégation de fenêtres et génération d'explications.

### Priorité moyenne

- Ajouter un mode de calibration/homographie optionnel pour convertir certains pixels en coordonnées route approximatives.
- Rendre les seuils configurables dans un fichier YAML/JSON plutôt que uniquement dans le code.
- Ajouter un exemple de petit JSON synthétique pour générer le dashboard sans devoir relancer YOLO.
- Gérer explicitement les vidéos sans accident détecté et les JSON sans véhicules.

### Priorité basse

- Harmoniser les accents dans les messages et l'interface (`vehicule` → `véhicule`, etc.) si l'encodage cible est UTF-8 partout.
- Ajouter des captures d'écran du dashboard dans le README.
- Ajouter un `pyproject.toml` avec formatage/linting et commandes de test standardisées.

## Conclusion

Le dépôt correspond globalement au prototype décrit : il contient bien une chaîne YOLOv8 → tracking → variables interprétables → règles de risque → boîte noire → dashboard explicable. En revanche, il doit être présenté comme un prototype heuristique. Pour un rendu de recherche plus solide, les deux manques principaux sont une évaluation quantitative sur annotations et une clarification visuelle du fait que les scores/probabilités et les liens causaux ne sont pas calibrés statistiquement.
