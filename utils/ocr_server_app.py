from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import (
    load_model as load_detection_model,
    load_processor as load_detection_processor,
)
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import (
    load_processor as load_recognition_processor,
)
import base64
import io

app = FastAPI()

# Load models and processors
langs = ["en"]
det_processor, det_model = load_detection_processor(), load_detection_model()
rec_model, rec_processor = load_recognition_model(), load_recognition_processor()


class InputData(BaseModel):
    numeric: Optional[List[float]] = None
    categ: Optional[List[int]] = None
    text: Optional[List[str]] = None
    images: Optional[List[str]] = None
    time_tick: Optional[int] = None
    label: Optional[float] = None
    key: Optional[str] = None


class PredictionOutput(BaseModel):
    key: str
    score: float
    featureScore: List[float]
    classProbabilities: List[float]
    predictedClass: int
    embeddings: List[float]
    textOutput: str


def predict(input_data: dict) -> PredictionOutput:
    try:
        images = input_data.get("images")
        if not images or len(images) == 0:
            raise ValueError("Missing required field: 'images'.")

        # Process the first image (you can modify this to handle multiple images if needed)
        base64_image = images[0]

        # Decode base64 image
        image_data = base64.b64decode(base64_image)

        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))

        # Convert to RGB if it's not already
        img = img.convert("RGB")

        # Run OCR
        predictions = run_ocr(
            [img], [langs], det_model, det_processor, rec_model, rec_processor
        )
        text_dump = str(predictions)

        return PredictionOutput(
            key=input_data.get("key"),
            score=0,
            featureScore=[0.0],
            classProbabilities=[0.0],
            predictedClass=0,
            embeddings=[0.0],
            textOutput=text_dump,
        )
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e


@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: InputData):
    try:
        input_dict = input_data.dict(exclude_unset=True)
        prediction = predict(input_dict)
        print(f"Prediction output: {prediction}")
        return prediction
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in predict_endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}"
        ) from e


@app.get("/")
async def root():
    return {"message": "OCR Prediction API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
