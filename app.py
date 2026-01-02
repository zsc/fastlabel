import os
import io
import json
import random
import threading
import requests
import base64
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip

from flask import Flask, render_template, request, jsonify

# --- Configuration ---
CELEBA_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
IMAGE_FILENAME = "celeba_48x48.png"
TILE_SIZE = 48
COLS = 200
ROWS = 150
TOTAL_IMAGES = COLS * ROWS
BATCH_SIZE = 10  # Number of images to show per batch
NUM_CLASSES = 10 # 0-9
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Model Definitions ---

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# --- State Management ---

@dataclass
class State:
    labeled: Dict[int, int] = field(default_factory=dict)
    unlabeled: List[int] = field(default_factory=list)
    embed_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    model: Optional[MLPHead] = None
    optimizer: Optional[optim.Optimizer] = None
    steps_since_train: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Cache for the large image to avoid reopening it constantly
    # In a real production app, we might use memory mapping or a database
    source_image: Optional[Image.Image] = None

app_state = State()

# --- Helper Functions ---

def download_image_if_not_exists():
    if not os.path.exists(IMAGE_FILENAME):
        print(f"Downloading {IMAGE_FILENAME} from {CELEBA_URL}...")
        try:
            response = requests.get(CELEBA_URL, stream=True)
            response.raise_for_status()
            with open(IMAGE_FILENAME, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading image: {e}")
            raise e
    else:
        print(f"Found {IMAGE_FILENAME}.")

def load_source_image():
    if app_state.source_image is None:
        download_image_if_not_exists()
        app_state.source_image = Image.open(IMAGE_FILENAME).convert('RGB')
        # Initialize unlabeled pool
        # For this demo, let's limit to first 1000 to keep it snappy if needed, 
        # or use full if memory allows. 30k ints is fine.
        app_state.unlabeled = list(range(TOTAL_IMAGES))
        random.shuffle(app_state.unlabeled)

def get_image_crop(idx: int) -> Image.Image:
    if app_state.source_image is None:
        load_source_image()
    
    if app_state.source_image is None:
        raise RuntimeError("Failed to load source image")
    
    col = idx % COLS
    row = idx // COLS
    left = col * TILE_SIZE
    upper = row * TILE_SIZE
    right = left + TILE_SIZE
    lower = upper + TILE_SIZE
    
    return app_state.source_image.crop((left, upper, right, lower))

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- CLIP & Model Initialization ---

print(f"Initializing CLIP model on {DEVICE}...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(DEVICE)
clip_model.eval()
print("CLIP initialized.")

def get_embedding(idx: int) -> np.ndarray:
    """Get CLIP embedding for image at idx. Compute and cache if necessary."""
    if idx in app_state.embed_cache:
        return app_state.embed_cache[idx]
    
    img = get_image_crop(idx)
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    
    embedding = features.cpu().numpy()[0]
    app_state.embed_cache[idx] = embedding
    return embedding

def init_mlp_if_needed():
    if app_state.model is None:
        # Get embedding dimension from a sample
        sample_embed = get_embedding(0)
        input_dim = sample_embed.shape[0]
        app_state.model = MLPHead(input_dim, NUM_CLASSES).to(DEVICE)
        app_state.optimizer = optim.Adam(app_state.model.parameters(), lr=1e-3)

def train_step():
    """Perform a training step using all labeled data."""
    if not app_state.labeled:
        return

    init_mlp_if_needed()
    app_state.model.train()
    
    # Prepare data
    ids = list(app_state.labeled.keys())
    labels = list(app_state.labeled.values())
    
    embeddings = [get_embedding(i) for i in ids]
    
    X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    
    # Train for a few epochs per trigger
    epochs = 5 
    for _ in range(epochs):
        app_state.optimizer.zero_grad()
        outputs = app_state.model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        app_state.optimizer.step()
        
    print(f"Training step complete. Loss: {loss.item():.4f}, Labeled count: {len(ids)}")

def predict(idx: int) -> Tuple[int, float]:
    """Return (predicted_label, confidence) for an image index."""
    if app_state.model is None:
        return -1, 0.0
    
    embedding = get_embedding(idx)
    X = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    app_state.model.eval()
    with torch.no_grad():
        outputs = app_state.model(X)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    return predicted.item(), confidence.item()

# --- Flask App ---

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/next_batch')
def next_batch():
    load_source_image() # Ensure loaded
    
    with app_state.lock:
        # Strategy: Random for now (or could be uncertainty sampling if model exists)
        # For simplicity and speed in demo: just take next available from shuffled list
        
        batch_ids = []
        candidates = []
        
        # If we have a model, we could score unlabeled items. 
        # But scanning all 30k for uncertainty is slow without vector search/indexing.
        # Let's just pick a random subset to score, then pick best from there.
        pool_size = min(len(app_state.unlabeled), 100)
        candidates = app_state.unlabeled[:pool_size]
        
        # If model exists, maybe sort by uncertainty (entropy)? 
        # For now, just return random candidates to keep UI responsive.
        batch_ids = candidates[:BATCH_SIZE]
        
        response_data = []
        for idx in batch_ids:
            img = get_image_crop(idx)
            img_b64 = image_to_base64(img)
            pred_label, conf = predict(idx)
            
            response_data.append({
                "id": idx,
                "image": img_b64,
                "prediction": pred_label if pred_label != -1 else None,
                "confidence": float(f"{conf:.2f}") if pred_label != -1 else None
            })
            
    return jsonify(response_data)

@app.route('/api/submit_labels', methods=['POST'])
def submit_labels():
    data = request.json
    # data format: [{"id": 1, "label": 3}, ...]
    
    triggered_training = False
    
    with app_state.lock:
        for item in data:
            idx = item['id']
            label = int(item['label'])
            
            app_state.labeled[idx] = label
            if idx in app_state.unlabeled:
                app_state.unlabeled.remove(idx)
        
        # Check if we should train
        # For demo, train every time a batch is submitted (if we have enough data)
        # Or simple rule: if labeled count > classes
        if len(app_state.labeled) >= 2: # Minimal requirement
             train_step()
             triggered_training = True
    
    return jsonify({
        "success": True,
        "triggered_training": triggered_training,
        "labeled_count": len(app_state.labeled),
        "unlabeled_count": len(app_state.unlabeled)
    })

if __name__ == '__main__':
    # Initialize image
    load_source_image()
    # Start app
    app.run(debug=True, host='0.0.0.0', port=8008, use_reloader=False) 
    # use_reloader=False to avoid double initialization of CLIP which is heavy
