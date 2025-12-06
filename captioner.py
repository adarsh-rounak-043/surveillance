# captioner.py
"""
BLIP (v1) captioner using:
  "Salesforce/blip-image-captioning-base"

Usage:
  from captioner import BlipCaptioner
  captioner = BlipCaptioner()            # will use GPU if available
  caption = captioner.generate_caption(frame_bgr)
"""

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

class Blip2Captioner:
    def __init__(self,
                 model_name: str = "Salesforce/blip-image-captioning-large",
                 device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[BLIP] Loading '{model_name}' on {self.device}...")
        # Processor handles feature extraction + tokenizer for BLIP v1
        self.processor = BlipProcessor.from_pretrained(model_name)
        # Model for conditional generation
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _frame_to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        """
        Convert OpenCV BGR frame (H,W,3) to PIL RGB image.
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        # Convert BGR -> RGB
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def generate_caption(self,
                         frame_bgr: np.ndarray,
                         max_new_tokens: int = 30,
                         num_beams: int = 3) -> str:
        """
        Generate a caption for a single frame.

        Args:
            frame_bgr: OpenCV image (H,W,3) in BGR format
            max_new_tokens: max tokens to generate
            num_beams: beam size for generation (1 = greedy)

        Returns:
            caption (str)
        """
        image = self._frame_to_pil(frame_bgr)

        # Processor prepares pixel values and returns tensors
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Generate tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode generated ids to text
        # processor.decode typically works; fallback to tokenizer if needed
        try:
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        except Exception:
            # some versions expose tokenizer via processor
            caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()


# if __name__ == "__main__":
#     # Quick test/example: reads an image file "test.jpg" (place in same dir)
#     import cv2
#     import sys

#     captioner = BlipCaptioner()

#     if len(sys.argv) > 1:
#         img_path = sys.argv[1]
#     else:
#         img_path = "test.jpg"  # default test file

#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Could not read '{img_path}'. Please provide a valid image path.")
#         sys.exit(1)

#     cap = captioner.generate_caption(img, max_new_tokens=30, num_beams=3)
#     print("Caption:", cap)
