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
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from sklearn.cluster import KMeans

from flask import Flask, render_template, request, jsonify

# --- Configuration ---
CELEBA_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
IMAGE_FILENAME = "celeba_48x48.png"
TILE_SIZE = 48
COLS = 200
ROWS = 150
TOTAL_IMAGES = COLS * ROWS
BATCH_SIZE = 24  # Increased for higher density
NUM_CLASSES = 2 # Binary: 0 (Negative), 1 (Positive)
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
    lock: threading.Lock = field(default_factory=threading.Lock)
    source_image: Optional[Image.Image] = None
    
    # Track current batch context to know what the user was verifying
    current_batch_type: str = "neutral" # neutral, positive, negative

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

def load_source_image():
    if app_state.source_image is None:
        download_image_if_not_exists()
        app_state.source_image = Image.open(IMAGE_FILENAME).convert('RGB')
        # Initialize unlabeled pool
        app_state.unlabeled = list(range(TOTAL_IMAGES))
        random.shuffle(app_state.unlabeled)

def get_image_crop(idx: int) -> Image.Image:
    if app_state.source_image is None:
        load_source_image()
    if app_state.source_image is None: # explicit check for mypy/runtime safety
         raise RuntimeError("Image not loaded")
    
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
        sample_embed = get_embedding(0)
        input_dim = sample_embed.shape[0]
        app_state.model = MLPHead(input_dim, NUM_CLASSES).to(DEVICE)
        app_state.optimizer = optim.Adam(app_state.model.parameters(), lr=1e-3)

def train_step():
    if not app_state.labeled:
        return

    init_mlp_if_needed()
    if app_state.model is None or app_state.optimizer is None: return

    app_state.model.train()
    
    ids = list(app_state.labeled.keys())
    labels = list(app_state.labeled.values())
    
    # Minimal caching for speed, but re-fetching from dict is fast enough
    embeddings = [get_embedding(i) for i in ids]
    
    X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    
    epochs = 5 
    for _ in range(epochs):
        app_state.optimizer.zero_grad()
        outputs = app_state.model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        app_state.optimizer.step()
        
    print(f"Training step complete. Loss: {loss.item():.4f}, Labeled count: {len(ids)}")

def predict_batch(ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (probs, predictions) for a list of IDs."""
    if app_state.model is None:
        # Uniform probability if no model
        return np.ones((len(ids), NUM_CLASSES)) / NUM_CLASSES, np.zeros(len(ids))
    
    embeddings = [get_embedding(i) for i in ids]
    X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    
    app_state.model.eval()
    with torch.no_grad():
        outputs = app_state.model(X)
        probs = torch.softmax(outputs, dim=1) # (N, 2)
        _, preds = torch.max(probs, 1)
        
    return probs.cpu().numpy(), preds.cpu().numpy()

# --- Strategies ---

def strategy_random(k: int) -> Tuple[List[int], str]:
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 1000)] # sample from top 1000 for speed
    selected = random.sample(pool, min(k, len(pool)))
    return selected, "neutral"

def strategy_kmeans(k: int) -> Tuple[List[int], str]:
    # Cluster a subset of unlabeled data
    pool_size = min(len(app_state.unlabeled), 1000)
    pool = app_state.unlabeled[:pool_size]
    
    embeddings = np.array([get_embedding(i) for i in pool])
    
    # We want 'k' items. If k is small, we can just pick k randoms?
    # Or actually, the prompt says "KMeans Big Class".
    # Interpretation: Cluster into K clusters, pick center of each?
    # Or: Cluster into X clusters, pick one whole cluster?
    # Let's interpret as: Find diversity. Cluster into k, pick centers.
    
    n_clusters = min(k, pool_size)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100)
    kmeans.fit(embeddings)
    
    # Find closest samples to centers
    selected = []
    for center in kmeans.cluster_centers_:
        dists = np.linalg.norm(embeddings - center, axis=1)
        idx = np.argmin(dists)
        selected.append(pool[idx])
        
    return selected, "neutral"

def strategy_uncertainty(k: int) -> Tuple[List[int], str]:
    # Borderline cases: prob close to 0.5
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    
    # Score = abs(prob_pos - 0.5). Smaller is more uncertain.
    scores = np.abs(probs[:, 1] - 0.5)
    
    # Sort by score ascending
    sorted_indices = np.argsort(scores)
    selected_indices = sorted_indices[:k]
    
    return [pool[i] for i in selected_indices], "neutral"

def strategy_verify_pos(k: int) -> Tuple[List[int], str]:
    # Easy cases: prob close to 1.0 (Positive)
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    
    # Score = prob_pos. Larger is better.
    scores = probs[:, 1]
    
    sorted_indices = np.argsort(scores)[::-1] # Descending
    selected_indices = sorted_indices[:k]
    
    return [pool[i] for i in selected_indices], "positive"

def strategy_verify_neg(k: int) -> Tuple[List[int], str]:
    # Easy cases: prob close to 0.0 (Negative)
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    
    # Score = prob_pos. Smaller is better (more negative).
    scores = probs[:, 1]
    
    sorted_indices = np.argsort(scores) # Ascending
    selected_indices = sorted_indices[:k]
    
    return [pool[i] for i in selected_indices], "negative"


# --- Flask App ---

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/next_batch')
def next_batch():
    load_source_image()
    strategy = request.args.get('strategy', 'random')
    try:
        batch_size = int(request.args.get('batch_size', BATCH_SIZE))
    except ValueError:
        batch_size = BATCH_SIZE
    
    with app_state.lock:
        if not app_state.unlabeled:
            return jsonify({"items": [], "batch_type": "neutral"})

        # Fallback if model not ready for model-based strategies
        if app_state.model is None and strategy not in ['random', 'kmeans']:
            strategy = 'random'
            
        if strategy == 'random':
            ids, batch_type = strategy_random(batch_size)
        elif strategy == 'kmeans':
            ids, batch_type = strategy_kmeans(batch_size)
        elif strategy == 'borderline':
            ids, batch_type = strategy_uncertainty(batch_size)
            # For borderline, sorting by prob_pos helps group similar levels of uncertainty
        elif strategy == 'easy_pos':
            ids, batch_type = strategy_verify_pos(batch_size)
            # Already sorted by prob_pos descending in strategy
        elif strategy == 'easy_neg':
            ids, batch_type = strategy_verify_neg(batch_size)
            # Already sorted by prob_pos ascending in strategy
        else:
            ids, batch_type = strategy_random(batch_size)
        
        app_state.current_batch_type = batch_type
        
        # Get predictions for display
        probs, preds = predict_batch(ids)
        
        response_data = []
        for i, idx in enumerate(ids):
            img = get_image_crop(idx)
            img_b64 = image_to_base64(img)
            
            prob_pos = float(probs[i][1])
            pred_label = int(preds[i]) if app_state.model else None
            
            response_data.append({
                "id": idx,
                "image": img_b64,
                "prediction": pred_label,
                "prob_pos": prob_pos
            })
            
        # Optional: Extra sort for 'borderline' to make it easier to look at
        if strategy == 'borderline':
            response_data.sort(key=lambda x: x['prob_pos'])
            
    return jsonify({
        "items": response_data,
        "batch_type": app_state.current_batch_type
    })

@app.route('/api/submit_labels', methods=['POST'])
def submit_labels():
    data = request.json
    # data format: [{"id": 1, "label": 1}, ...]
    
    triggered_training = False
    
    with app_state.lock:
        for item in data:
            idx = item['id']
            label = int(item['label'])
            
            app_state.labeled[idx] = label
            if idx in app_state.unlabeled:
                app_state.unlabeled.remove(idx)
        
        if len(app_state.labeled) >= 2:
             train_step()
             triggered_training = True
    
    return jsonify({
        "success": True,
        "triggered_training": triggered_training,
        "labeled_count": len(app_state.labeled),
        "unlabeled_count": len(app_state.unlabeled)
    })

if __name__ == '__main__':
    load_source_image()
    # DO NOT CHANGE THE PORT - FIXED AT 8008
    app.run(debug=True, host='0.0.0.0', port=8008, use_reloader=False)