import os
import random
import io
import copy
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import gradio as gr
try:
    import spaces
except ImportError:
    # Mock spaces if running locally without it
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

# Load CLIP globally to avoid reloading per session (ZeroGPU handles shared memory well)
print(f"Initializing CLIP model on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(DEVICE)
clip_model.eval()
print("CLIP initialized.")

# --- Helper Functions (ZeroGPU Decorated) ---

@spaces.GPU
def extract_feature_gpu(image: Image.Image) -> np.ndarray:
    image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

@spaces.GPU
def train_step_gpu(model_state, labeled_data, embed_cache):
    # Reconstruct model from state dict
    model = MLPHead(512, 2).to(DEVICE)
    if model_state:
        model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # Prepare data
    ids = list(labeled_data.keys())
    labels = list(labeled_data.values())
    
    # Fetch embeddings (Assuming they are already computed and passed in, or we'd need to re-compute)
    # In this simplified demo, we assume embed_cache is sufficient.
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
def predict_batch_gpu(model_state, embeddings_list):
    if model_state is None:
        # Random probs
        n = len(embeddings_list)
        return np.ones((n, 2)) / 2, np.zeros(n)
        
    model = MLPHead(512, 2).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()
    
    X = torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        
    return probs.cpu().numpy(), preds.cpu().numpy()

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

# Global Data Source (ReadOnly)
data_source = CelebADataSource()

# --- Session State ---

@dataclass
class SessionState:
    labeled: Dict[int, int] = field(default_factory=dict)
    unlabeled: List[int] = field(default_factory=list)
    embed_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    model_state: Optional[Dict] = None
    
    current_batch_ids: List[int] = field(default_factory=list)
    current_batch_mode: str = "neutral" # 'verify_pos', 'verify_neg', 'neutral'

    def __init__(self):
        self.labeled = {}
        self.unlabeled = list(range(2000)) # Limit to 2000 for demo speed
        random.shuffle(self.unlabeled)
        self.embed_cache = {}
        self.model_state = None
        self.current_batch_ids = []

# --- Logic Handlers ---

def get_embedding_safe(idx, session):
    if idx not in session.embed_cache:
        img = data_source.get_image(idx)
        # Call GPU function
        session.embed_cache[idx] = extract_feature_gpu(img)
    return session.embed_cache[idx]

def load_next_batch(session: SessionState, strategy: str):
    # Simple strategy logic
    # For ZeroGPU demo, we'll keep strategies simple to avoid latency
    
    pool_size = min(len(session.unlabeled), 200)
    pool = session.unlabeled[:pool_size]
    
    if not pool:
        return session, [], "No more data"

    # Compute/Get embeddings for pool
    pool_embeds = [get_embedding_safe(i, session) for i in pool]
    
    # Predict
    probs, preds = predict_batch_gpu(session.model_state, pool_embeds)
    
    # Select based on strategy
    selected_indices = []
    
    if strategy == "Random" or session.model_state is None:
        selected_indices = random.sample(range(len(pool)), min(BATCH_SIZE, len(pool)))
        session.current_batch_mode = "neutral"
        batch_title = "Random Batch: Select Positive items"
    elif strategy == "Verify Positives":
        # Sort by prob_pos descending
        sort_idx = np.argsort(probs[:, 1])[::-1]
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "verify_pos"
        batch_title = "Verify Positives: Select items that are NOT Positive"
    elif strategy == "Verify Negatives":
        # Sort by prob_pos ascending
        sort_idx = np.argsort(probs[:, 1])
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "verify_neg"
        batch_title = "Verify Negatives: Select items that are NOT Negative"
    else:
        # Borderline
        scores = np.abs(probs[:, 1] - 0.5)
        sort_idx = np.argsort(scores)
        selected_indices = sort_idx[:BATCH_SIZE]
        session.current_batch_mode = "neutral"
        batch_title = "Uncertainty Batch: Select Positive items"

    session.current_batch_ids = [pool[i] for i in selected_indices]
    
    # Prepare Gallery Images
    images = []
    for idx in session.current_batch_ids:
        img = data_source.get_image(idx)
        # Add caption? Gradio Gallery supports (img, caption) tuples
        conf = probs[pool.index(idx)][1] if session.model_state else 0.5
        caption = f"#{idx} ({conf:.2f})"
        images.append((img, caption))
        
    return session, images, batch_title

def submit_batch(session: SessionState, selected_data: List[Any]):
    # selected_data is list of indices (if type='index') or item data
    # Gradio Gallery returns list of selected items.
    # We need to map selection back to IDs.
    
    if not session.current_batch_ids:
        return session, "Error: No batch loaded"

    # Gradio Gallery selection returns list of (image, caption) or just indices if type='index'
    # We will configure Gallery to return indices.
    
    # Determine Labels logic
    # mode 'verify_pos': All displayed are proposed Positive. Selected = Negative. Unselected = Positive.
    # mode 'verify_neg': All displayed are proposed Negative. Selected = Positive. Unselected = Negative.
    # mode 'neutral': Default assumption? Let's say user selects Positives.
    
    selected_indices_in_batch = [item[1] for item in selected_data] if selected_data else [] 
    # Wait, gr.Gallery with type='index' returns list of indices directly? 
    # Actually standard is list of selected items. 
    # Let's assume we get a list of INDICES in the gallery [0, 2, 5...]
    
    batch_ids = session.current_batch_ids
    
    # Map batch index to global ID
    # selected_data comes as a list of `SelectData` objects in newer Gradio, or indices.
    # We'll use the event listener approach which is safer.
    pass 

# --- Gradio Callback Wrappers ---

def init_app():
    return SessionState()

def on_load_click(session, strategy):
    session, images, title = load_next_batch(session, strategy)
    return session, images, title

def on_select(evt: gr.SelectData, session):
    # We can track selection state manually if needed, 
    # or just let the Gallery component handle multi-selection and we read `.value` on submit.
    # For simplicity, we read the Gallery value on Submit.
    return

def on_submit_click(session, gallery_selected_items):
    # gallery_selected_items: list of selected (image, caption) tuples? 
    # Or if type="index", list of indices.
    
    if not session.current_batch_ids:
        return session, [], "Please load a batch first.", "0", "0"
    
    selected_indices = []
    # Handle different Gradio versions/configs. Assuming type='index' -> [0, 2, ...]
    if gallery_selected_items:
        # If it's a list of dicts/SelectData (unlikely for .value)
        # If type='index', it is just a list of ints.
        selected_indices = [int(x) for x in gallery_selected_items]
    
    ids_to_remove = []
    
    for i, global_id in enumerate(session.current_batch_ids):
        is_selected = i in selected_indices
        
        # Logic Mapping
        label = None
        if session.current_batch_mode == 'verify_pos':
            # Proposed: Pos (1). Selected = Wrong = Neg (0). Unselected = Pos (1).
            label = 0 if is_selected else 1
        elif session.current_batch_mode == 'verify_neg':
            # Proposed: Neg (0). Selected = Wrong = Pos (1). Unselected = Neg (0).
            label = 1 if is_selected else 0
        else:
            # Neutral: User selects Positives (1). Unselected = Neg (0).
            label = 1 if is_selected else 0
            
        session.labeled[global_id] = label
        ids_to_remove.append(global_id)
        
    for gid in ids_to_remove:
        if gid in session.unlabeled:
            session.unlabeled.remove(gid)
            
    # Train
    # Ensure embeds exist
    for gid in session.labeled:
        get_embedding_safe(gid, session)
        
    model_state, loss = train_step_gpu(
        session.model_state, 
        session.labeled, 
        session.embed_cache
    )
    session.model_state = model_state
    
    # Reload next batch automatically? Or just clear?
    # Let's clear and show status.
    status = f"Trained! Loss: {loss:.4f}. Labeled: {len(session.labeled)}"
    
    # Load next immediately to keep flow
    # session, images, title = load_next_batch(session, "Verify Positives" if len(session.labeled) > 10 else "Random")
    # For now just clear
    
    return session, [], "Batch Submitted. Click Load to continue.", str(len(session.labeled)), str(len(session.unlabeled))


# --- UI Construction ---

with gr.Blocks(title="FastLabel ZeroGPU") as demo:
    session = gr.State(init_app)
    
    with gr.Row():
        gr.Markdown("## FastLabel on ZeroGPU")
    
    with gr.Row():
        with gr.Column(scale=1):
            strategy_drop = gr.Dropdown(
                choices=["Random", "Verify Positives", "Verify Negatives", "Borderline"], 
                value="Random", 
                label="Strategy"
            )
            btn_load = gr.Button("Load Batch", variant="primary")
            
            gr.Markdown("### Stats")
            lbl_count = gr.Number(value=0, label="Labeled")
            unlbl_count = gr.Number(value=2000, label="Unlabeled")
            
        with gr.Column(scale=3):
            info_box = gr.Markdown("### Ready to start. Click 'Load Batch'.")
            # Gallery to show images
            # allow_preview=False to disable modal, enable multi-select
            gallery = gr.Gallery(
                label="Batch Data", 
                show_label=False, 
                elem_id="gallery", 
                columns=[6], 
                rows=[4], 
                height="auto",
                allow_preview=False,
                type="index" # Important: return indices on selection
            )
            
            btn_submit = gr.Button("Confirm & Train", variant="stop")

    # Events
    btn_load.click(
        fn=on_load_click,
        inputs=[session, strategy_drop],
        outputs=[session, gallery, info_box]
    )
    
    btn_submit.click(
        fn=on_submit_click,
        inputs=[session, gallery],
        outputs=[session, gallery, info_box, lbl_count, unlbl_count]
    )

if __name__ == "__main__":
    demo.launch()
