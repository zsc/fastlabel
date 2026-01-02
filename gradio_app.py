import os
import random
import io
import copy
import json
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import gradio as gr
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# --- Configuration ---
CELEBA_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
IMAGE_FILENAME = "celeba_48x48.png"
TILE_SIZE = 48
COLS = 200
ROWS = 150
TOTAL_IMAGES = COLS * ROWS
BATCH_SIZE = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models & Logic ---

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

print(f"Initializing CLIP model on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(DEVICE)
clip_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("CLIP initialized.")

# --- GPU Functions ---

@spaces.GPU
def extract_feature_gpu(image: Image.Image) -> np.ndarray:
    image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

@spaces.GPU
def encode_text_gpu(text: str) -> torch.Tensor:
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu()

@spaces.GPU
def train_step_gpu(model_state, labeled_data, embed_cache):
    model = MLPHead(512, 2).to(DEVICE)
    if model_state:
        model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    ids = list(labeled_data.keys())
    labels = list(labeled_data.values())
    embeddings = [embed_cache[i] for i in ids]
    
    X = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optimizer.step()
        
    return model.state_dict(), loss.item()

@spaces.GPU
def predict_batch_gpu(model_state, text_embed, embeddings_list):
    # If we have a model, use it
    if model_state is not None:
        model = MLPHead(512, 2).to(DEVICE)
        model.load_state_dict(model_state)
        model.eval()
        
        X = torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
        return probs.cpu().numpy(), preds.cpu().numpy()
    
    # Fallback: Zero-shot text
    if text_embed is not None:
        X = torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(DEVICE)
        text_feat = text_embed.to(DEVICE)
        with torch.no_grad():
            sim = (X @ text_feat.T).squeeze(1)
            logits = sim * 100.0
            probs_pos = torch.sigmoid(logits)
            probs = torch.zeros((len(embeddings_list), 2), device=DEVICE)
            probs[:, 1] = probs_pos
            probs[:, 0] = 1 - probs_pos
            preds = (probs_pos > 0.5).long()
        return probs.cpu().numpy(), preds.cpu().numpy()

    # Fallback: Random
    n = len(embeddings_list)
    return np.ones((n, 2)) / 2, np.zeros(n)

# --- DataSource ---

class CelebADataSource:
    def __init__(self):
        self.source_image = None
        self.load()

    def load(self):
        if not os.path.exists(IMAGE_FILENAME):
            print("Downloading CelebA...")
            try:
                r = requests.get(CELEBA_URL)
                with open(IMAGE_FILENAME, 'wb') as f: f.write(r.content)
            except:
                print("Failed to download CelebA")
                return
        self.source_image = Image.open(IMAGE_FILENAME).convert('RGB')

    def get_image(self, idx: int) -> Image.Image:
        if self.source_image is None: self.load()
        col, row = idx % COLS, idx // COLS
        left, top = col * TILE_SIZE, row * TILE_SIZE
        return self.source_image.crop((left, top, left + TILE_SIZE, top + TILE_SIZE))

data_source = CelebADataSource()

# --- Session State ---

@dataclass
class Snapshot:
    labeled: Dict[int, int]
    unlabeled: List[int]
    model_state: Optional[Dict]

@dataclass
class SessionState:
    labeled: Dict[int, int] = field(default_factory=dict)
    unlabeled: List[int] = field(default_factory=list)
    embed_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    model_state: Optional[Dict] = None
    
    text_query: Optional[str] = None
    text_embedding: Optional[torch.Tensor] = None
    
    current_batch_ids: List[int] = field(default_factory=list)
    current_batch_mode: str = "neutral"
    
    history: List[Snapshot] = field(default_factory=list)

    def __init__(self):
        self.labeled = {}
        # Limit pool for demo speed in Gradio
        self.unlabeled = list(range(2000)) 
        random.shuffle(self.unlabeled)
        self.embed_cache = {}
        self.model_state = None
        self.current_batch_ids = []
        self.history = []

    def save_snapshot(self):
        snap = Snapshot(
            labeled=copy.deepcopy(self.labeled),
            unlabeled=copy.deepcopy(self.unlabeled),
            model_state=copy.deepcopy(self.model_state) if self.model_state else None
        )
        self.history.append(snap)
        if len(self.history) > 3:
            self.history.pop(0)

    def restore_snapshot(self):
        if not self.history: return False
        snap = self.history.pop()
        self.labeled = snap.labeled
        self.unlabeled = snap.unlabeled
        self.model_state = snap.model_state
        return True

# --- Logic Handlers ---

def init_app():
    return SessionState()

def get_embedding_safe(idx, session):
    if idx not in session.embed_cache:
        img = data_source.get_image(idx)
        session.embed_cache[idx] = extract_feature_gpu(img)
    return session.embed_cache[idx]

def on_set_query(session, query):
    if not query:
        session.text_query = None
        session.text_embedding = None
        return session, "Query cleared."
    
    session.text_query = query
    session.text_embedding = encode_text_gpu(query)
    return session, f"Query set: '{query}'. Use 'Verify Positives' now."

def load_next_batch(session: SessionState, strategy: str):
    # Sample pool for prediction
    pool_size = min(len(session.unlabeled), 500)
    pool = session.unlabeled[:pool_size]
    
    if not pool:
        return session, [], "No more data"

    pool_embeds = [get_embedding_safe(i, session) for i in pool]
    probs, _ = predict_batch_gpu(session.model_state, session.text_embedding, pool_embeds)
    
    if strategy == "Random":
        # Fallback to model if available, else true random
        if session.model_state or session.text_embedding:
             # If model exists, random strategy usually implies diversity or just random sampling
             # Let's stick to simple random for diversity
             selected_indices = random.sample(range(len(pool)), min(BATCH_SIZE, len(pool)))
        else:
             selected_indices = random.sample(range(len(pool)), min(BATCH_SIZE, len(pool)))
        session.current_batch_mode = "neutral"
        title = "Random Batch: Select Positive items"
        
    elif strategy == "Verify Positives":
        sort_idx = np.argsort(probs[:, 1])[::-1]
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "verify_pos"
        title = "Verify Positives: Select items that are NOT Positive"
        
    elif strategy == "Verify Negatives":
        sort_idx = np.argsort(probs[:, 1])
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "verify_neg"
        title = "Verify Negatives: Select items that are NOT Negative"
        
    elif strategy == "Borderline":
        scores = np.abs(probs[:, 1] - 0.5)
        sort_idx = np.argsort(scores)
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "neutral"
        title = "Uncertainty Batch: Select Positive items"
    else:
        selected_indices = []
        title = "Unknown Strategy"

    session.current_batch_ids = [pool[i] for i in selected_indices]
    
    # Prepare Gallery
    images = []
    for idx in session.current_batch_ids:
        img = data_source.get_image(idx)
        # Find prob
        p_idx = pool.index(idx)
        conf = probs[p_idx][1]
        images.append((img, f"#{idx} ({conf:.0%})"))
        
    return session, images, title

def on_submit_click(session, gallery_selected):
    if not session.current_batch_ids:
        return session, [], "Load batch first", 0, 0
    
    # Save Undo
    session.save_snapshot()
    
    selected_indices = []
    if gallery_selected:
        selected_indices = [int(x) for x in gallery_selected]
    
    ids_to_remove = []
    for i, global_id in enumerate(session.current_batch_ids):
        is_selected = i in selected_indices
        label = 0
        
        if session.current_batch_mode == 'verify_pos':
            label = 0 if is_selected else 1
        elif session.current_batch_mode == 'verify_neg':
            label = 1 if is_selected else 0
        else: # Neutral
            label = 1 if is_selected else 0
            
        session.labeled[global_id] = label
        ids_to_remove.append(global_id)
        
    for gid in ids_to_remove:
        if gid in session.unlabeled:
            session.unlabeled.remove(gid)
            
    # Train
    for gid in session.labeled:
        get_embedding_safe(gid, session)
        
    model_state, loss = train_step_gpu(session.model_state, session.labeled, session.embed_cache)
    session.model_state = model_state
    
    status = f"Trained! Loss: {loss:.4f}"
    
    return session, [], status, len(session.labeled), len(session.unlabeled)

def on_undo_click(session):
    success = session.restore_snapshot()
    msg = "Undo successful" if success else "Nothing to undo"
    return session, msg, len(session.labeled), len(session.unlabeled)

def render_review_tab(session):
    # Group by label
    pos_ids = [i for i, l in session.labeled.items() if l == 1]
    neg_ids = [i for i, l in session.labeled.items() if l == 0]
    
    pos_imgs = [(data_source.get_image(i), f"#{i}") for i in pos_ids]
    neg_imgs = [(data_source.get_image(i), f"#{i}") for i in neg_ids]
    
    return pos_imgs, neg_imgs

def render_autolabel_tab(session):
    # Predict on a chunk of unlabeled
    pool = session.unlabeled[:200] # Limit for display
    if not pool: return []
    
    embeds = [get_embedding_safe(i, session) for i in pool]
    probs, preds = predict_batch_gpu(session.model_state, session.text_embedding, embeds)
    
    # Sort by conf descending
    results = []
    for i, idx in enumerate(pool):
        results.append({
            "id": idx,
            "conf": probs[i][1],
            "pred": preds[i]
        })
    results.sort(key=lambda x: x["conf"], reverse=True)
    
    # Format for gallery
    out = []
    for item in results:
        img = data_source.get_image(item["id"])
        label_str = "Pos" if item["pred"] == 1 else "Neg"
        out.append((img, f"{label_str} ({item['conf']:.0%})"))
    return out

def export_json(session):
    data = [{"filename": f"celeba_{k}.png", "label": v} for k, v in session.labeled.items()]
    file_path = "/tmp/labels.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    return file_path

# --- Layout ---

with gr.Blocks(title="FastLabel ZeroGPU") as demo:
    session = gr.State(init_app)
    
    gr.Markdown("# FastLabel on ZeroGPU (Multi-modal Active Learning)")
    
    with gr.Tabs():
        # --- TAB 1: LABELING ---
        with gr.Tab("Labeling"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Zero-shot Init")
                    txt_query = gr.Textbox(placeholder="e.g. 'wearing hat'", label="Text Query")
                    btn_query = gr.Button("Set Query")
                    
                    gr.Markdown("### 2. Strategy")
                    strategy_drop = gr.Dropdown(
                        choices=["Random", "Verify Positives", "Verify Negatives", "Borderline"], 
                        value="Random", show_label=False
                    )
                    btn_load = gr.Button("Load Batch", variant="primary")
                    
                    gr.Markdown("### 3. Actions")
                    btn_undo = gr.Button("Undo Last", variant="secondary")
                    
                    gr.Markdown("### Stats")
                    lbl_count = gr.Number(value=0, label="Labeled")
                    unlbl_count = gr.Number(value=2000, label="Unlabeled")
                    
                with gr.Column(scale=3):
                    info_box = gr.Markdown("### Ready. Set a query or just Load Batch.")
                    gallery = gr.Gallery(
                        label="Batch", show_label=False, columns=6, height="auto", allow_preview=False
                    )
                    btn_submit = gr.Button("Confirm & Train", variant="stop")

        # --- TAB 2: REVIEW ---
        with gr.Tab("Review"):
            btn_refresh_review = gr.Button("Refresh Review")
            gr.Markdown("#### Positive")
            gallery_pos = gr.Gallery(show_label=False, columns=8, height="auto")
            gr.Markdown("#### Negative")
            gallery_neg = gr.Gallery(show_label=False, columns=8, height="auto")

        # --- TAB 3: AUTOLABEL ---
        with gr.Tab("Autolabel (AI)"):
            btn_refresh_auto = gr.Button("Run Inference on Unlabeled Pool")
            gr.Markdown("Showing top 200 predictions sorted by confidence.")
            gallery_auto = gr.Gallery(show_label=False, columns=8, height="auto")

        # --- TAB 4: EXPORT ---
        with gr.Tab("Export"):
            btn_export = gr.Button("Generate JSON")
            file_output = gr.File(label="Download Labels")

    # --- Wiring ---
    
    btn_query.click(on_set_query, [session, txt_query], [session, info_box])
    
    btn_load.click(
        on_load_click, 
        [session, strategy_drop], 
        [session, gallery, info_box]
    )
    
    btn_submit.click(
        on_submit_click,
        [session, gallery],
        [session, gallery, info_box, lbl_count, unlbl_count]
    )
    
    btn_undo.click(
        on_undo_click,
        [session],
        [session, info_box, lbl_count, unlbl_count]
    )
    
    btn_refresh_review.click(
        render_review_tab,
        [session],
        [gallery_pos, gallery_neg]
    )
    
    btn_refresh_auto.click(
        render_autolabel_tab,
        [session],
        [gallery_auto]
    )
    
    btn_export.click(
        export_json,
        [session],
        [file_output]
    )

if __name__ == "__main__":
    demo.launch()