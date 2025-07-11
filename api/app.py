import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io

app = FastAPI()

# Model URI (GCS path)
MODEL_URI = "gs://mlops-animal-classifier/models/models/m-f2878fa41e5c4dc089e112c447a862ab/artifacts"

# Load model from MLflow or GCS (make sure you have authentication set up)
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def hello_world():
    return {"message" : "Hello, world!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    return JSONResponse(content={"predicted_class": preds.item()})

# To run, save as app.py and run:
# uvicorn app:app --host 0.0.0.0 --port 8080
