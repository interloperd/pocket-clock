#!/usr/bin/env bash
# Lance tout le projet Pocket Clock d'un coup (Mac / Linux)
set -e
cd "$(dirname "$0")"
if [ ! -d data/tracking ]; then
  echo "Téléchargement des données (~830 Mo)..."
  git clone --depth 1 https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data.git _repo
  mv _repo/data ./data && rm -rf _repo
fi
python3 -m pip install -q -r requirements.txt
echo "Calcul du Pocket Clock sur les 122 matchs (~2 min)..."
python3 pocket_clock.py data > /dev/null
python3 validate.py > /dev/null
python3 export.py data > /dev/null
python3 build_dashboard.py
open pocket_clock_dashboard.html 2>/dev/null || xdg-open pocket_clock_dashboard.html 2>/dev/null || true
