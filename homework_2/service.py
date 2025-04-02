import bentoml
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL.Image import Image

MODEL_NAME = "google/vit-base-patch16-224"

@bentoml.service
class ViTClassifier:
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    
    @bentoml.api
    def classify(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return {"Predicted class": self.model.config.id2label[predicted_class_idx]}
