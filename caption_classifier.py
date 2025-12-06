# caption_classifier.py
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class CaptionClassifier:
    def __init__(self,
                 model_dir: str = "saved_model",
                 device: str | None = None):
        """
        model_dir:
            Folder containing config.json, model.safetensors, tokenizer files, etc.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[BERT] Loading tokenizer and model from '{model_dir}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        ).to(self.device)
        self.model.eval()

        # Get id2label mapping from config if present
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        if "id2label" in cfg:
            # keys are usually strings: {"0": "normal", "1": "intrusion", ...}
            self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        else:
            # Fallback: generic mapping – change manually if needed
            num_labels = cfg.get("num_labels", 2)
            self.id2label = {i: f"label_{i}" for i in range(num_labels)}
        print(f"[BERT] id2label = {self.id2label}")

    def predict(self, caption: str) -> tuple[str, float]:
        """
        caption -> (label_str, confidence_float)
        """
        inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        conf, pred_id = torch.max(probs, dim=-1)
        label = self.id2label[int(pred_id)]
        confidence = float(conf)
        return label, confidence
