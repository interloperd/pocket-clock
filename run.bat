@echo off
REM Lance tout le projet Pocket Clock d'un coup (Windows)
cd /d "%~dp0"
if not exist data\tracking (
  echo Telechargement des donnees (~830 Mo)...
  git clone --depth 1 https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data.git _repo
  move _repo\data data
  rmdir /s /q _repo
)
python -m pip install -q -r requirements.txt
echo Calcul du Pocket Clock sur les 122 matchs (~2 min)...
python pocket_clock.py data > nul
python validate.py > nul
python export.py data > nul
python build_dashboard.py
start pocket_clock_dashboard.html
