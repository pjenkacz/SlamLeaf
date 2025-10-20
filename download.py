import os
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise SystemExit("Brak ROBOFLOW_API_KEY. Ustaw zmienną środowiskową i uruchom ponownie.")


WORKSPACE = "joseph-nelson"
PROJECT   = "plantdoc"
VERSION   = 4          
FORMAT    = "yolov11"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download(FORMAT, location="C:/pd")

print("Pobrano do:", dataset.location)  
