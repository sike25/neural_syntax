import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import random
import os

# --- Constants & Metadata ---
WORLD_SIZE = 5
OBJECT_FEATURE_DIMENSION = 6
NEURALESE_DIMENSION = 12

VOCABULARY = {
    '<blank>': 0, 'not': 1, 'and': 2, 'or': 3, 'red': 4, 'green': 5, 'purple': 6,
    'circle': 7, 'square': 8, 'triangle': 9, 'no-outline': 10, 'slim-outline': 11, 'thick-outline': 12
}
IDX_TO_TOKEN = {v: k for k, v in VOCABULARY.items()}

# Normalization constants from translator.ipynb
NEURALESE_MEAN = torch.tensor([-0.9473, 6.5567, -3.5357, 1.7435, -9.0126, 9.1548, -5.7529, 1.2613, 6.9551, -6.7343, -1.3457, 0.2404])
NEURALESE_STD = torch.tensor([1.6304, 1.6894, 1.9658, 1.5834, 2.0234, 2.4538, 1.8573, 1.6224, 2.2969, 1.7448, 1.2031, 1.1943])

# --- Model Definitions ---
class ObjectEncoder(nn.Module):
    def __init__(self, output_dimension=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2),
            nn.Flatten(),
            nn.Linear(16, output_dimension)
        )
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        return self.encoder(x)

class Speaker(nn.Module):
    def __init__(self, input_dimension=35, output_dimension=12):
        super().__init__()
        self.speaker_net = nn.Sequential(
            nn.Linear(input_dimension, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dimension)
        )
    def forward(self, x): return self.speaker_net(x)

class Listener(nn.Module):
    def __init__(self, input_dimension=18, output_dimension=1):
        super().__init__()
        self.listener_net = nn.Sequential(
            nn.Linear(input_dimension, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, output_dimension)
        )
    def forward(self, x): return self.listener_net(x)

class Translator(nn.Module):
    def __init__(self, neuralese_dimension=12, max_rule_length=3, vocab_size=13, hidden_dimension=128):
        super().__init__()
        self.translation_net = nn.Sequential(
            nn.Linear(neuralese_dimension, hidden_dimension), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dimension, max_rule_length * vocab_size)
        )
    def forward(self, V):
        logits = self.translation_net(V)
        return logits.view(-1, 3, 13)

# --- Data Logic ---
COLORS = ['RED', 'GREEN', 'PURPLE']
SHAPES = ['CIRCLE', 'SQUARE', 'TRIANGLE']
OUTLINES = ['NONE', 'SLIM', 'THICK']

class ObjectInstance:
    def __init__(self, color, shape, outline):
        self.color, self.shape, self.outline = color, shape, outline
    def to_array(self):
        arr = np.zeros((3, 3), dtype=np.float32)
        arr[0, COLORS.index(self.color)] = 1
        arr[1, SHAPES.index(self.shape)] = 1
        arr[2, OUTLINES.index(self.outline)] = 1
        return arr

def generate_random_world():
    return [ObjectInstance(random.choice(COLORS), random.choice(SHAPES), random.choice(OUTLINES)) for _ in range(5)]

def generate_random_rule():
    ops = ['SINGLE', 'NOT', 'AND', 'OR']
    op = random.choice(ops)
    attrs = ['color', 'shape', 'outline']
    attr1 = random.choice(attrs)
    v_map = {'color': COLORS, 'shape': SHAPES, 'outline': OUTLINES}
    val1 = random.choice(v_map[attr1])
    rule = {'op': op, 'att1': attr1, 'val1': val1}
    if op in ['AND', 'OR']:
        attr2 = random.choice([a for a in attrs if a != attr1])
        rule['att2'], rule['val2'] = attr2, random.choice(v_map[attr2])
    return rule

def rule_matches(rule, obj):
    v1 = getattr(obj, rule['att1'])
    if rule['op'] == 'SINGLE': return v1 == rule['val1']
    if rule['op'] == 'NOT': return v1 != rule['val1']
    v2 = getattr(obj, rule['att2'])
    if rule['op'] == 'AND': return v1 == rule['val1'] and v2 == rule['val2']
    if rule['op'] == 'OR': return v1 == rule['val1'] or v2 == rule['val2']
    return False

def rule_to_text(rule):
    t1 = f"{rule['val1'].lower()}"
    if rule['op'] == 'SINGLE': return t1
    if rule['op'] == 'NOT': return f"not {t1}"
    return f"{t1} {rule['op'].lower()} {rule['val2'].lower()}"

# --- Visualization ---
def draw_world(world, selection=None):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect('equal'); ax.axis('off')
    color_map = {'RED': '#E74C3C', 'GREEN': '#2ECC71', 'PURPLE': '#9B59B6'}
    for i, obj in enumerate(world):
        c = color_map[obj.color]
        lw = {'NONE': 0, 'SLIM': 2, 'THICK': 5}[obj.outline]
        ec = 'black' if lw > 0 else c
        if selection is not None and selection[i]:
            ax.add_patch(patches.Rectangle((i - 0.45, -0.45), 0.9, 0.9, fill=False, color='gold', lw=3, ls='--'))
        if obj.shape == 'CIRCLE': ax.add_patch(patches.Circle((i, 0), 0.35, fc=c, ec=ec, lw=lw))
        elif obj.shape == 'SQUARE': ax.add_patch(patches.Rectangle((i - 0.3, -0.3), 0.6, 0.6, fc=c, ec=ec, lw=lw))
        elif obj.shape == 'TRIANGLE': ax.add_patch(patches.RegularPolygon((i, 0), 3, radius=0.4, fc=c, ec=ec, lw=lw))
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', transparent=True); plt.close()
    return Image.open(buf)

def draw_heatmap(vector):
    fig, ax = plt.subplots(figsize=(6, 1))
    v = vector.numpy()[0]
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
    ax.imshow(v_norm.reshape(1, -1), cmap='magma', aspect='auto')
    ax.set_xticks(range(12)); ax.set_yticks([])
    for i in range(12): ax.text(i, 0, f"{v[i]:.1f}", ha='center', va='center', color='white' if v_norm[i] < 0.5 else 'black', fontsize=8)
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
    return Image.open(buf)

# --- State ---
class AppState:
    def __init__(self):
        self.world = None; self.rule = None; self.mask = None; self.neuralese = None
        self.encoder = ObjectEncoder(); self.speaker = Speaker(); self.listener = Listener(); self.translator = Translator()
        try:
            m_path = os.path.join(os.path.dirname(__file__), "..", "training", "models")
            ckpt = torch.load(os.path.join(m_path, "speaker_listener.pt"), map_location='cpu')
            self.speaker.load_state_dict(ckpt['speaker']); self.encoder.load_state_dict(ckpt['encoder']); self.listener.load_state_dict(ckpt['listener'])
            self.translator.load_state_dict(torch.load(os.path.join(m_path, "mlp_translator.pt"), map_location='cpu'))
            for m in [self.encoder, self.speaker, self.listener, self.translator]: m.eval()
        except Exception as e: print(f"Model Load Error: {e}")

state = AppState()

def generate_instance():
    state.world = generate_random_world()
    while True:
        state.rule = generate_random_rule()
        state.mask = [rule_matches(state.rule, obj) for obj in state.world]
        if 0 < sum(state.mask) < 5: break
    return draw_world(state.world), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""

def run_speaker():
    w_t = torch.stack([torch.from_numpy(o.to_array()) for o in state.world])
    with torch.no_grad():
        feats = state.encoder(w_t).view(1, -1)
        m_t = torch.tensor(state.mask, dtype=torch.float32).view(1, -1)
        state.neuralese = state.speaker(torch.cat([feats, m_t], dim=1))
    return draw_heatmap(state.neuralese), gr.update(visible=True), gr.update(visible=True)

def run_listener():
    w_t = torch.stack([torch.from_numpy(o.to_array()) for o in state.world])
    with torch.no_grad():
        feats = state.encoder(w_t)
        n_rep = state.neuralese.repeat(5, 1)
        preds = (torch.sigmoid(state.listener(torch.cat([n_rep, feats], dim=1)).view(-1)) > 0.5).numpy()
    return draw_world(state.world, preds), gr.update(visible=True)

def run_translator():
    V_n = (state.neuralese - NEURALESE_MEAN) / (NEURALESE_STD + 1e-8)
    with torch.no_grad():
        tokens = [IDX_TO_TOKEN[i.item()] for i in torch.argmax(state.translator(V_n), dim=2)[0] if i.item() != 0]
    return f"Translator says: \"{' '.join(tokens)}\"", gr.update(visible=True)

# --- UI ---
with gr.Blocks(title="Neural Babel") as demo:
    with gr.Column() as landing:
        gr.Markdown("# Neural Babel\n### What Do Neural Networks Talk About?")
        gr.Markdown("Two AIs were trained to play a game. One (the speaker) looks at a set of objects, picks out the ones that fit a secret rule, and sends a private message to the other (the listener). The listener uses only that message — no rule, no hints — to find the right objects. A third AI, the translator, tries to decode what the message actually said.")
        explore_btn = gr.Button("Explore Project", variant="primary")
        gr.Markdown("---\n[Research + Code at Github](https://github.com/sike25/neural_syntax)")
    with gr.Column(visible=False) as main:
        gr.Markdown("## Neural Babel Demo")
        reset_btn = gr.Button("Generate New World", variant="secondary", size="sm")
        world_img = gr.Image(label="The World (W)", interactive=False)
        encode_btn = gr.Button("Speaker: Encode World + Rule", variant="primary")
        neuralese_img = gr.Image(label="Neuralese Message", visible=False, interactive=False)
        with gr.Row(visible=False) as action_row:
            with gr.Column():
                gr.Markdown("⬇️"); listen_btn = gr.Button("Listener: Select Objects"); listener_img = gr.Image(label="Listener Result")
            with gr.Column():
                gr.Markdown("⬇️"); translate_btn = gr.Button("Translator: Code to English"); translator_txt = gr.Markdown()
        reveal_btn = gr.Button("Reveal the Rule", visible=False); rule_txt = gr.Markdown()

    explore_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing, main]).then(generate_instance, None, [world_img, encode_btn, neuralese_img, action_row, reveal_btn, rule_txt])
    reset_btn.click(generate_instance, None, [world_img, encode_btn, neuralese_img, action_row, reveal_btn, rule_txt])
    encode_btn.click(run_speaker, None, [neuralese_img, action_row, encode_btn])
    listen_btn.click(run_listener, None, [listener_img, reveal_btn])
    translate_btn.click(run_translator, None, [translator_txt, reveal_btn])
    reveal_btn.click(lambda: f"### The Secret Rule was: **{rule_to_text(state.rule)}**", None, rule_txt)

if __name__ == "__main__":
    demo.launch()
