# Campus Surveillance — Caption‑Based Real‑Time Monitoring 📹🧠

## 📌 Description  
This project implements a basic **real‑time campus surveillance system** using **image captioning + text‑classification**.  
The pipeline:

- Captures frames from a live camera feed (webcam / IP‑camera)
- Generates a **natural language caption** for each frame using a BLIP model
- Classifies the caption into event types (normal / intrusion / violence / accident …)
- Displays a live preview with caption + event label overlay
- Prints alerts in console when suspicious events are detected

This prototype is built for exploring **caption‑driven surveillance** approaches, not just object detection.

---

## 🚀 Features  
✔ Real‑time CCTV monitoring  
✔ Vision‑language captioning (BLIP)  
✔ BERT‑based caption classifier  
✔ Alert system for critical events  
✔ Overlay text on video feed  
✔ Support for webcam + RTSP streams  

---

## 🧩 Tech Used
- Python 3.11
- PyTorch
- HuggingFace Transformers
- OpenCV
- BLIP for image captioning
- BERT for text classification

GPU recommended for performance (but CPU mode works slower).

---

## 📦 Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/adarsh-rounak-043/surveillance.git
cd surveillance
2️⃣ (Optional) Create a virtual environment:

bash
Copy code
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # Linux/Mac
3️⃣ Install dependencies:

bash
Copy code
pip install -r requirements.txt
4️⃣ Make sure your fine‑tuned BERT classifier exists in:

pgsql
Copy code
saved_model/
   ├─ config.json
   ├─ model.safetensors
   ├─ tokenizer.json
   ├─ vocab.json
   ├─ merges.txt
   └─ training_args.bin
▶️ Usage
Run live preview with a webcam:

bash
Copy code
python main_preview.py --camera 0
Or connect an IP/RTSP camera:

bash
Copy code
python main_preview.py --camera "rtsp://user:pass@<IP>:554/stream"
Press q to exit the preview.

👉 AI predictions appear over the video, and console prints alerts like:

less
Copy code
[ALERT] intrusion (0.88)
Caption: a person climbing the fence near the gate
📁 Project Structure
graphql
Copy code
campus_surveillance/
│
├─ main_preview.py            # Real-time monitoring and overlay display
├─ captioner.py               # BLIP image captioning model
├─ caption_classifier.py      # BERT text classification
├─ event_logic.py             # Alert conditions
├─ requirements.txt           # Dependencies
└─ saved_model/               # BERT model + tokenizer files
🧠 How It Works
Step	Module	Output
1. Capture frame	OpenCV	Raw image
2. Generate caption	BLIP	Text sentence
3. Classify caption	BERT	Event label + confidence
4. Display results	OpenCV overlay	Live preview
5. Alerts	Logic module	Console notification

🔧 Customization
Want to expand the system? Some ideas:

Enhancement	File to Modify
Add more event types	caption_classifier.py + re-train BERT
Save images/video on alert	main_preview.py
Different thresholds for alerts	event_logic.py
Multi‑camera streams	duplicate capture threads in main_preview.py
Face blurring for privacy	integrate detection models

⚠️ Limitations
Captioning alone may misclassify complex behaviors

CPU mode = slow performance for real‑time use

No tracking / identity recognition yet

Privacy concerns if deployed publicly → apply anonymization