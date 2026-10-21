# ⏱️ Pocket Clock

**Combien de temps la poche protège-t-elle le quarterback ?**
*Et comment la faire tenir, ou tomber.*

Projet réalisé pour le NFL Big Data Bowl, événement régional AWS, par Dima Dhehibi.

---

## En bref

Pocket Clock chronomètre, image par image, le temps entre le snap et le moment où le premier pass rusher arrive à moins de 2 yards du quarterback. Sur 8 532 passes de la saison 2021, la poche tient **3,7 s en médiane**, et le taux de sack passe de **0,3 % à 19 %** dès qu'elle est percée. Le play action fait gagner 1,3 s de poche, alors que chaque rusher ajouté par la défense en retire. L'application transforme ce chrono en décisions : un plan de match qui recommande combien de rushers envoyer ou s'il faut appeler le play action, un mode Direct qui rejoue un vrai match avec des alertes en temps réel, une fiche de scouting par adversaire et des classements de joueurs. Les coachs de ligne y voient où leur protection lâche, les recruteurs repèrent les rushers d'élite, et les commentateurs ont un chrono à afficher en direct.

---

## Ce que révèlent les données

| Constat | Chiffre |
|---|---|
| Tenue médiane de la poche | **3,7 s** |
| Poches déjà percées 2,5 s après le snap | **14,8 %** |
| Taux de sack, poche intacte → poche percée | **0,3 % → 19,3 %** |
| Passes complètes, poche intacte → poche percée | **62,9 % → 35,0 %** |
| Temps de poche gagné grâce au play action | **+1,3 s** |
| Tenue médiane à 4 rushers → blitz à 6 ou plus | **3,7 s → 3,0 s** |

La métrique retrouve d'elle-même les meilleurs pass rushers de la saison : Trey Hendrickson, Myles Garrett, Yannick Ngakoue, Matt Judon et Von Miller sont les premiers à atteindre le QB sur près d'un snap sur cinq. C'est un bon signe de validité.

---

## Comment la métrique est calculée

1. **Départ du chrono** : l'image du snap.
2. **Fin de la fenêtre** : le lancer, le sack ou la course du QB.
3. À chaque image, soit 10 par seconde, on mesure la distance entre le QB et le pass rusher le plus proche. Les rôles viennent des données PFF.
4. **Pocket Clock** : le premier instant où cette distance passe sous 2 yards, soit environ une longueur de bras plus un pas.
5. Si la poche tient jusqu'au lancer, on ne sait pas combien de temps elle aurait encore tenu. On utilise donc l'estimateur de **Kaplan-Meier**, issu de la statistique médicale, pour construire une courbe de survie de la poche sans biaiser le résultat.

---

## Le tableau de bord

Au lancement, une **présentation en 4 écrans** explique simplement ce qu'est la poche, le problème et les fonctionnalités. Elle reste accessible depuis la barre latérale.

Une application en trois temps : **Jour de match**, **Préparation**, **Méthode**.

- **Mode Direct** : un vrai match de 2021 rejoué action par action, comme si les données arrivaient en direct. Terrain animé pour 6 matchs vedettes, chrono de poche, recommandation recalculée avant chaque snap, alertes qui apparaissent au fil du match, puis animation de fin et bilan. Lecture ×0,5 à ×4, boutons « Précédente » et « Suivante » ou flèches gauche et droite pour naviguer entre les actions, barre d'espace pour lecture et pause.

Le reste de l'application, du plus synthétique au plus détaillé :

- **Ce qu'un coach doit retenir** : quatre phrases, lisibles en cinq secondes.
- **Plan de match en direct** : le coach choisit son équipe, l'adversaire, la tentative et la distance. La page recommande instantanément combien de rushers envoyer en défense, ou s'il faut appeler le play action en attaque, avec le risque de sack et les yards attendus. Les options peu observées sont signalées.
- **Fiche de scouting** : on choisit l'adversaire de la semaine et on voit six indicateurs avec leur rang sur 32, en attaque comme en défense.
- **Rejeu animé** : la poche qui se referme, avec le chrono qui tourne et la distance du rusher le plus proche.
- **Courbes de survie** : chaque équipe comparée à la moyenne NFL.
- **Ce que vaut une demi-seconde** : l'issue de la passe selon le moment où la poche cède.
- **Contexte tactique** : play action, nombre de rushers, couverture, type de dropback.
- **Classements** : pass rushers « premier arrivé », joueurs de ligne offensive « bloc tenu », équipes.
- **Fatigue** : un résultat contre-intuitif, expliqué plus bas.
- **Fiabilité** : validation hors échantillon, calibration et test du seuil, affichés dans la page.
- **Mode vidéo** : une démo de 85 secondes qui se joue toute seule, avec souris, zooms et sous-titres, pour présenter l'app.

---

## Testé et vérifié

- **15 tests automatiques** vérifient le chrono sur des actions synthétiques dont on connaît la réponse : arrivée du rusher, poche intacte, sack, rien compté avant le snap ni après le lancer, rayon modifiable. Ils vérifient aussi le modèle : Kaplan-Meier, effet d'équipe retrouvé, équipe peu observée ramenée vers la moyenne.
- **Validation hors échantillon** : le modèle de recommandation est entraîné sur les semaines 1 à 6 et évalué sur les semaines 7 et 8. Il est bien calibré : quand il annonce 22 % de poches percées, on en observe 20 % ; quand il annonce 43 %, on en observe 43 %.
- **Choix du seuil** : entre 1,5 et 2 yards, les classements restent proches. Au-delà, la majorité des poches comptent comme percées et la mesure ne distingue plus rien. 2 yards est le seuil le plus large qui garde un vrai signal.
- **Valeur d'une poche** : une poche percée coûte 3,8 yards par action, 7,8 yards quand elle tient contre 4,0 quand elle casse.

---

## Lancer le projet

**Voir le résultat** : ouvrir `index.html` dans un navigateur. Toutes les données sont intégrées, rien à installer.

**Lancer les tests** : `python -m pytest tests`

**Tout recalculer d'un coup** :

```bash
./run.sh        # Mac / Linux
run.bat         # Windows, double-clic
```

Le script télécharge les données si le dossier `data` est absent, environ 830 Mo, installe les librairies, calcule la métrique sur les 122 matchs en deux minutes environ et ouvre le tableau de bord.

**Le notebook** : `jupyter notebook pocket_clock.ipynb`. Il déroule toute la démarche en quatorze étapes, avec les graphiques. Sans installation : Google Colab, Fichier puis Importer un notebook.

---

## Structure du projet

| Fichier | Rôle |
|---|---|
| `pocket_clock.ipynb` | La démarche pas à pas, pédagogique et exécutée |
| `pocket_clock.py` | Calcule le Pocket Clock sur toutes les actions |
| `model.py` | Modèle de recommandation et courbes de survie |
| `validate.py` | Validation hors échantillon, calibration, sensibilité au seuil |
| `tests/` | Tests automatiques |
| `export.py` | Agrège les indicateurs, les classements et les rejeux |
| `build_dashboard.py` | Intègre les données dans le tableau de bord |
| `dashboard_template.html` | Le tableau de bord, sans les données |
| `index.html` | Le tableau de bord final, prêt à être hébergé |
| `run.sh` · `run.bat` | Lancement en une commande |
| `pocket_clock_video.srt` | Sous-titres de la vidéo de présentation |

---

## Limites et pistes

- **Rayon fixe de 2 yards** : un rusher tenu au contact peut se trouver à 2 yards sans menacer réellement le QB. Prochaine étape : intégrer la vitesse et l'orientation du rusher vers le QB.
- **QB mobiles** : un quarterback qui sort de la poche allonge mécaniquement le chrono.
- **Fatigue** : contrairement à l'intuition, les rushers atteignent plus souvent le QB en fin de match. Les joueurs qui cumulent le plus de snaps sont les titulaires d'élite, et les fins de match amènent des situations de passe évidentes. Pour isoler la fatigue, il faudrait comparer chaque joueur à lui-même, à situation de jeu égale.
- **Échantillon** : huit semaines de la saison 2021. Les classements individuels restent indicatifs.

---

## Données

- Tracking des joueurs : **NFL Next Gen Stats**
- Rôles, pressions et blocs : **Pro Football Focus**
- Source : [nfl-big-data-bowl-regional-event-data](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data)
