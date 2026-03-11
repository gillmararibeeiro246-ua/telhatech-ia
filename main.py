from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import shutil

app = FastAPI()

# carregar modelo
model = YOLO("best.pt")

# pasta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# abrir interface
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    caminho = f"temp_{file.filename}"

    with open(caminho, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(caminho)

    total = 0
    ceramica = 0
    brasilit = 0

    for r in results:
        for c in r.boxes.cls:
            total += 1

            if int(c) == 0:
                ceramica += 1
            elif int(c) == 1:
                brasilit += 1

    return {
        "Total de telhados": total,
        "Telhados de cerâmica": ceramica,
        "Telhados de brasilit": brasilit
    }