import os
import io
import json
import random
import threading
import base64
import numpy as np
import glob
import requests
import copy
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify, send_file

# For Audio Processing
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration & Constants ---
CELEBA_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
IMAGE_FILENAME = "celeba_48x48.png"
TILE_SIZE = 48
COLS = 200
ROWS = 150
TOTAL_IMAGES = COLS * ROWS

BATCH_SIZE_DEFAULT = 24
NUM_CLASSES = 2 
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PORT = 8008

# --- Model Definitions ---

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# --- Audio Processing ---

class AudioProcessor:
    @staticmethod
    def generate_mel_spectrogram(audio_path: str) -> Image.Image:
        """Converts audio file to a PIL Image of its Mel-Spectrogram."""
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

# --- DataSource Abstractions ---

class DataSource:
    def __init__(self, data_type: str, root_path: str):
        self.data_type = data_type # 'image' or 'audio'
        self.root_path = root_path
        self.file_list: List[str] = []
        self.scan_files()
        self.image_cache: Dict[int, Image.Image] = {}

    def scan_files(self):
        if not os.path.exists(self.root_path):
            self.file_list = []
            return
        extensions = ['*.png', '*.jpg', '*.jpeg'] if self.data_type == 'image' else ['*.wav', '*.mp3', '*.flac']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.root_path, ext)))
            files.extend(glob.glob(os.path.join(self.root_path, ext.upper())))
        self.file_list = sorted(files)

    def get_image(self, idx: int) -> Image.Image:
        if idx in self.image_cache: return self.image_cache[idx]
        path = self.file_list[idx]
        if self.data_type == 'image':
            img = Image.open(path).convert('RGB')
        else:
            img = AudioProcessor.generate_mel_spectrogram(path).resize((224, 224))
        self.image_cache[idx] = img
        return img

    def get_audio_path(self, idx: int) -> str:
        return self.file_list[idx]

class CelebADataSource:
    def __init__(self):
        self.data_type = 'image'
        self.source_image = None
        self.file_list = ["celeba_idx_" + str(i) for i in range(TOTAL_IMAGES)]
        self.image_cache: Dict[int, Image.Image] = {}

    def load(self):
        if not os.path.exists(IMAGE_FILENAME):
            print(f"Downloading CelebA from {CELEBA_URL}...")
            r = requests.get(CELEBA_URL)
            with open(IMAGE_FILENAME, 'wb') as f: f.write(r.content)
        self.source_image = Image.open(IMAGE_FILENAME).convert('RGB')

    def get_image(self, idx: int) -> Image.Image:
        if self.source_image is None: self.load()
        if idx in self.image_cache: return self.image_cache[idx]
        col, row = idx % COLS, idx // COLS
        left, top = col * TILE_SIZE, row * TILE_SIZE
        img = self.source_image.crop((left, top, left + TILE_SIZE, top + TILE_SIZE)).resize((224, 224))
        self.image_cache[idx] = img
        return img

    def get_audio_path(self, idx: int) -> str:
        raise ValueError("No audio in CelebA")

# --- State Management ---

@dataclass
class Snapshot:
    labeled: Dict[int, int]
    unlabeled: List[int]
    model_state: Optional[Dict[str, Any]]
    optimizer_state: Optional[Dict[str, Any]]

@dataclass
class State:
    data_source: Any = field(default_factory=CelebADataSource)
    labeled: Dict[int, int] = field(default_factory=dict) # Index -> Label
    unlabeled: List[int] = field(default_factory=list) # List of indices
    embed_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    model: Optional[MLPHead] = None
    optimizer: Optional[optim.Optimizer] = None
    text_query: Optional[str] = None
    text_embedding: Optional[torch.Tensor] = None
    current_batch_type: str = "neutral"
    
    # Undo History
    history: List[Snapshot] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def save_snapshot(self):
        """Save current state to history (limit 3)."""
        model_state = self.model.state_dict() if self.model else None
        opt_state = self.optimizer.state_dict() if self.optimizer else None
        
        # Deepcopy mutable structures
        snapshot = Snapshot(
            labeled=copy.deepcopy(self.labeled),
            unlabeled=copy.deepcopy(self.unlabeled),
            model_state=copy.deepcopy(model_state) if model_state else None,
            optimizer_state=copy.deepcopy(opt_state) if opt_state else None
        )
        self.history.append(snapshot)
        if len(self.history) > 3:
            self.history.pop(0)

    def restore_snapshot(self):
        """Restore last state from history."""
        if not self.history:
            return False
        
        snapshot = self.history.pop()
        self.labeled = snapshot.labeled
        self.unlabeled = snapshot.unlabeled
        
        if snapshot.model_state:
            # Re-init model if it was None but snapshot has it (unlikely in this flow but possible)
            if self.model is None:
                # Need input dim to init. Can guess from embed cache or just re-init on fly
                # Assuming embedding dim 512 for ViT-B-32
                self.model = MLPHead(512, NUM_CLASSES).to(DEVICE)
                self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
            self.model.load_state_dict(snapshot.model_state)
            if snapshot.optimizer_state and self.optimizer:
                self.optimizer.load_state_dict(snapshot.optimizer_state)
        else:
            self.model = None
            self.optimizer = None
            
        return True

app_state = State()
app_state.unlabeled = list(range(TOTAL_IMAGES))
random.shuffle(app_state.unlabeled)

# --- CLIP Initialization ---

print(f"Initializing CLIP model on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(DEVICE)
clip_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("CLIP initialized.")

# --- Helper Functions ---

def get_embedding(idx: int) -> np.ndarray:
    if idx in app_state.embed_cache:
        return app_state.embed_cache[idx]
    
    if app_state.data_source is None:
        raise RuntimeError("No data source configured")
        
    img = app_state.data_source.get_image(idx)
    img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    
    embedding = features.cpu().numpy()[0]
    app_state.embed_cache[idx] = embedding
    return embedding

def init_mlp_if_needed():
    if app_state.model is None:
        # Infer input dim from first available embedding
        first_id = app_state.unlabeled[0] if app_state.unlabeled else (list(app_state.labeled.keys())[0] if app_state.labeled else 0)
        sample_embed = get_embedding(first_id)
        input_dim = sample_embed.shape[0]
        app_state.model = MLPHead(input_dim, NUM_CLASSES).to(DEVICE)
        app_state.optimizer = optim.Adam(app_state.model.parameters(), lr=1e-3)

def train_step():
    if not app_state.labeled: return
    init_mlp_if_needed()
    if app_state.model is None or app_state.optimizer is None: return
    app_state.model.train()
    ids = list(app_state.labeled.keys())
    labels = list(app_state.labeled.values())
    embeddings = [get_embedding(i) for i in ids]
    X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    for _ in range(10):
        app_state.optimizer.zero_grad()
        outputs = app_state.model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        app_state.optimizer.step()

def predict_batch(ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if app_state.model is not None:
        embeddings = [get_embedding(i) for i in ids]
        X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
        app_state.model.eval()
        with torch.no_grad():
            outputs = app_state.model(X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
        return probs.cpu().numpy(), preds.cpu().numpy()

    if app_state.text_embedding is not None:
        embeddings_arr = np.array([get_embedding(i) for i in ids])
        img_feats = torch.tensor(embeddings_arr).to(DEVICE)
        text_feat = app_state.text_embedding.to(DEVICE)
        with torch.no_grad():
            sim = (img_feats @ text_feat.T).squeeze(1)
            logits = sim * 100.0
            probs_pos = torch.sigmoid(logits)
            probs = torch.zeros((len(ids), 2), device=DEVICE)
            probs[:, 1] = probs_pos
            probs[:, 0] = 1 - probs_pos
            preds = (probs_pos > 0.5).long()
        return probs.cpu().numpy(), preds.cpu().numpy()

    return np.ones((len(ids), NUM_CLASSES)) / NUM_CLASSES, np.zeros(len(ids))

# --- Strategies ---

def strategy_random(k: int) -> List[int]:
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 1000)]
    return random.sample(pool, min(k, len(pool)))

def strategy_kmeans(k: int) -> List[int]:
    pool_size = min(len(app_state.unlabeled), 1000)
    pool = app_state.unlabeled[:pool_size]
    embeddings = np.array([get_embedding(i) for i in pool])
    n_clusters = min(k, pool_size)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100)
    kmeans.fit(embeddings)
    selected = []
    for center in kmeans.cluster_centers_:
        dists = np.linalg.norm(embeddings - center, axis=1)
        selected.append(pool[np.argmin(dists)])
    return selected

def strategy_uncertainty(k: int) -> List[int]:
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    scores = np.abs(probs[:, 1] - 0.5)
    return [pool[i] for i in np.argsort(scores)[:k]]

def strategy_verify_pos(k: int) -> List[int]:
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    return [pool[i] for i in np.argsort(probs[:, 1])[::-1][:k]]

def strategy_verify_neg(k: int) -> List[int]:
    pool = app_state.unlabeled[:min(len(app_state.unlabeled), 500)]
    probs, _ = predict_batch(pool)
    return [pool[i] for i in np.argsort(probs[:, 1])[:k]]

# --- Flask App ---

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['POST'])
def set_config():
    data = request.json
    data_type = data.get('type', 'image')
    root_path = data.get('path', '')
    if not root_path or not os.path.exists(root_path):
        return jsonify({"success": False, "message": "Invalid path"}), 400
    with app_state.lock:
        app_state.data_source = DataSource(data_type, root_path)
        app_state.labeled = {}
        app_state.unlabeled = list(range(len(app_state.data_source.file_list)))
        random.shuffle(app_state.unlabeled)
        app_state.embed_cache = {}
        app_state.model = None
        app_state.optimizer = None
        app_state.history = [] # Reset history on config change
    return jsonify({"success": True, "count": len(app_state.unlabeled), "type": data_type})

@app.route('/api/set_query', methods=['POST'])
def set_query():
    query = request.json.get('query', '').strip()
    with app_state.lock:
        if not query:
            app_state.text_query = None
            app_state.text_embedding = None
        else:
            app_state.text_query = query
            with torch.no_grad():
                text_tokens = tokenizer([query]).to(DEVICE)
                text_features = clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                app_state.text_embedding = text_features
    return jsonify({"success": True})

@app.route('/api/next_batch')
def next_batch():
    if app_state.data_source is None:
        return jsonify({"items": [], "batch_type": "neutral", "message": "Not configured"})
    strategy = request.args.get('strategy', 'random')
    batch_size = int(request.args.get('batch_size', BATCH_SIZE_DEFAULT))
    with app_state.lock:
        if not app_state.unlabeled: return jsonify({"items": [], "batch_type": "neutral"})
        batch_type = "neutral"
        if strategy == 'easy_pos': batch_type = "positive"
        elif strategy == 'easy_neg': batch_type = "negative"
        app_state.current_batch_type = batch_type
        if app_state.model is None and strategy not in ['random', 'kmeans']:
            if app_state.text_embedding is None: strategy = 'random'
        if strategy == 'random': ids = strategy_random(batch_size)
        elif strategy == 'kmeans': ids = strategy_kmeans(batch_size)
        elif strategy == 'borderline': ids = strategy_uncertainty(batch_size)
        elif strategy == 'easy_pos': ids = strategy_verify_pos(batch_size)
        elif strategy == 'easy_neg': ids = strategy_verify_neg(batch_size)
        else: ids = strategy_random(batch_size)
        probs, preds = predict_batch(ids)
        response_data = []
        for i, idx in enumerate(ids):
            img_pil = app_state.data_source.get_image(idx)
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            response_data.append({
                "id": idx, "image": img_b64,
                "filename": os.path.basename(str(app_state.data_source.file_list[idx])),
                "prediction": int(preds[i]), "prob_pos": float(probs[i][1])
            })
        if batch_type == "positive": response_data.sort(key=lambda x: x['prob_pos'], reverse=True)
        elif batch_type == "negative": response_data.sort(key=lambda x: x['prob_pos'])
        else: response_data.sort(key=lambda x: x['prob_pos'], reverse=True)
    return jsonify({"items": response_data, "batch_type": batch_type, "data_type": app_state.data_source.data_type, "undo_available": len(app_state.history) > 0})

@app.route('/api/submit_labels', methods=['POST'])
def submit_labels():
    data = request.json
    with app_state.lock:
        # Save snapshot BEFORE updating
        app_state.save_snapshot()
        
        for item in data:
            idx = item['id']; label = int(item['label'])
            app_state.labeled[idx] = label
            if idx in app_state.unlabeled: app_state.unlabeled.remove(idx)
        if len(app_state.labeled) >= 2: train_step()
        
    return jsonify({"success": True, "labeled_count": len(app_state.labeled), "unlabeled_count": len(app_state.unlabeled)})

@app.route('/api/undo', methods=['POST'])
def undo_last_step():
    with app_state.lock:
        success = app_state.restore_snapshot()
    return jsonify({"success": success, "labeled_count": len(app_state.labeled), "unlabeled_count": len(app_state.unlabeled)})

@app.route('/api/image/<int:idx>')
def get_image(idx):
    if app_state.data_source:
        img = app_state.data_source.get_image(idx)
        buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return "Not configured", 404

@app.route('/api/audio/<int:idx>')
def get_audio(idx):
    if app_state.data_source and app_state.data_source.data_type == 'audio':
        path = app_state.data_source.get_audio_path(idx)
        return send_file(path)
    return "Not in audio mode", 404

@app.route('/api/labeled_data')
def get_labeled_data():
    grouped = {}
    with app_state.lock:
        for idx, label in app_state.labeled.items():
            lbl_str = str(label)
            if lbl_str not in grouped: grouped[lbl_str] = []
            grouped[lbl_str].append(idx)
    return jsonify(grouped)

@app.route('/api/export')
def export_labels():
    results = []
    with app_state.lock:
        if app_state.data_source:
            for idx, label in app_state.labeled.items():
                results.append({"filename": os.path.basename(str(app_state.data_source.file_list[idx])), "label": label})
    buf = io.BytesIO(); buf.write(json.dumps(results, indent=2).encode('utf-8')); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='labels.json', mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT, use_reloader=False)
