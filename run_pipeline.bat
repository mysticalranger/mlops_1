@echo off
echo Running preprocessing script...
python scripts/preprocess.py

echo Training model...
python scripts/train.py

echo Running model monitoring...
python monitoring.py

echo Starting Flask API...
python app.py


/ filepath: c:\Users\riyan\Downloads\project_1\wellness_app_project\Makefile