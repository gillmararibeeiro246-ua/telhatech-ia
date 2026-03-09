from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

model = YOLO("best.pt")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Detector de Telhados</title>
    </head>
    <body>

    <h2>Detector de Telhados</h2>

    <input type="file" id="image">
    <button onclick="enviar()">Analisar</button>

    <pre id="resultado"></pre>

    <script>
    async function enviar(){

        let file = document.getElementById("image").files[0]

        let formData = new FormData()
        formData.append("file", file)

        let res = await fetch("/predict", {
            method:"POST",
            body:formData
        })

        let data = await res.json()

        document.getElementById("resultado").innerText =
        `Total de telhados: ${data.total}

Telhados de cerâmica: ${data.ceramica}
Telhados de brasilit: ${data.brasilit}`
    }
    </script>

    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(path)

    ceramica = 0
    brasilit = 0

    for r in results:
        for c in r.boxes.cls:
            if int(c) == 0:
                ceramica += 1
            elif int(c) == 1:
                brasilit += 1

    total = ceramica + brasilit

    os.remove(path)

    return {
        "total": total,
        "ceramica": ceramica,
        "brasilit": brasilit
    }