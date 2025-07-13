import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from fastapi.middleware.cors import CORSMiddleware
import torch.nn.functional as F

app = FastAPI()

# Add this CORS middleware to allow all origins:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc)
    allow_headers=["*"],  # allow all headers
)

# Model URI (GCS path)
MODEL_URI = "gs://mlops-animal-classifier/models/models/m-f2878fa41e5c4dc089e112c447a862ab/artifacts/data/model.pth"

# Load model from MLflow or GCS (make sure you have authentication set up)
def load_model_cpu(uri):
    local_path = mlflow.artifacts.download_artifacts(uri)
    model = torch.load(local_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model_cpu(MODEL_URI)

model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.get("/hello_world")
async def hello_world():
    return {"message" : "Hello, world!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    prediction = preds.item()
    prediction_dict = {
        0 : "Dog",
        1 : "Horse",
        2 : "Elephant",
        3 : "Butterfly",
        4 : "Chicken",
        5 : "Cat",
        6 : "Cow",
        7 : "Sheep",
        8 : "Squirrel",
        9 : "Spider"
    }
    animal_name = prediction_dict.get(prediction, "Unknown")
    probs = probabilities.squeeze().tolist()

    return JSONResponse(content={"Animal Prediction": animal_name, "Probabilities": probs })

# To run, save as app.py and run:
# uvicorn app:app --host 0.0.0.0 --port 8080
