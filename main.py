from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

app = FastAPI()

# permitir acessar arquivos da pasta static
app.mount("/static", StaticFiles(directory="static"), name="static")


# rota da página inicial
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return FileResponse("static/index.html")


# rota para análise da imagem
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # carregar modelo somente quando analisar
    from ultralytics import YOLO
    model = YOLO("best.pt")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)

    total = 0
    ceramica = 0
    brasilit = 0

    for r in results:
        for c in r.boxes.cls:
            total += 1

            if int(c) == 0:
                ceramica += 1

            if int(c) == 1:
                brasilit += 1

    return {
        "Total de telhados": total,
        "Telhados de cerâmica": ceramica,
        "Telhados de brasilit": brasilit
    }


# rota de histórico
@app.get("/history")
def history():
    return []