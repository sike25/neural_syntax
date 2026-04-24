import streamlit as st
import torch
import numpy as np
import random

from huggingface_hub import hf_hub_download
from model_definitions import SpeakerListenerSystem, Translator


# ── PAGE CONFIG ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Babel Radio",
    page_icon="📡",
    layout="centered",
)


# ── CONSTANTS ──────────────────────────────────────────────────────────────────

WORLD_SIZE               = 5
OBJECT_FEATURE_DIMENSION = 6
NEURALESE_DIMENSION      = 12

COLORS   = ['RED', 'GREEN', 'PURPLE']
SHAPES   = ['CIRCLE', 'SQUARE', 'TRIANGLE']
OUTLINES = ['NONE', 'SLIM', 'THICK']

SHAPE_GLYPH   = {'CIRCLE': '●', 'SQUARE': '■', 'TRIANGLE': '▲'}
COLOR_HEX     = {'RED': '#ff4b4b', 'GREEN': '#21c55d', 'PURPLE': '#a855f7'}
OUTLINE_STYLE = {'NONE': 'none', 'SLIM': '2px solid', 'THICK': '5px solid'}

TOKEN_TO_INDEX = {
    '<blank>': 0, 'not': 1, 'and': 2, 'or': 3, 'red': 4, 'green': 5, 'purple': 6,
    'circle': 7, 'square': 8, 'triangle': 9, 'no-outline': 10, 'slim-outline': 11, 'thick-outline': 12
}
INDEX_TO_TOKEN = {v: k for k, v in TOKEN_TO_INDEX.items()}

ALL_OBJECTS = [
    (color, shape, outline)
    for color   in COLORS
    for shape   in SHAPES
    for outline in OUTLINES
]


# ── DATA CLASSES ───────────────────────────────────────────────────────────────

class ObjectInstance:
    def __init__(self, color, shape, outline):
        self.color   = color
        self.shape   = shape
        self.outline = outline

    def to_array(self):
        arr = np.zeros((3, 3), dtype=np.float32)
        arr[0, COLORS.index(self.color)]     = 1
        arr[1, SHAPES.index(self.shape)]     = 1
        arr[2, OUTLINES.index(self.outline)] = 1
        return arr


# ── GAME LOGIC ─────────────────────────────────────────────────────────────────

def generate_random_world():
    sample = random.sample(ALL_OBJECTS, WORLD_SIZE)
    return [ObjectInstance(c, s, o) for c, s, o in sample]

def generate_random_rule():
    values_map = {'color': COLORS, 'shape': SHAPES, 'outline': OUTLINES}
    attributes = ['color', 'shape', 'outline']
    operation  = random.choice(['SINGLE', 'NOT', 'AND', 'OR'])
    attribute_1 = random.choice(attributes)
    value_1     = random.choice(values_map[attribute_1])
    rule = {'operation': operation, 'attribute_1': attribute_1, 'value_1': value_1}
    if operation in ['AND', 'OR']:
        attribute_2         = random.choice([a for a in attributes if a != attribute_1])
        value_2             = random.choice(values_map[attribute_2])
        rule['attribute_2'] = attribute_2
        rule['value_2']     = value_2
        rule['string']      = f"{value_1} {operation} {value_2}"
    elif operation == 'NOT':
        rule['string'] = f"{operation} {value_1}"
    else:
        rule['string'] = value_1
    return rule

def object_matches_rule(rule, obj):
    operation = rule['operation']
    value_1   = getattr(obj, rule['attribute_1'])
    if operation == 'SINGLE': return value_1 == rule['value_1']
    if operation == 'NOT':    return value_1 != rule['value_1']
    value_2 = getattr(obj, rule['attribute_2'])
    if operation == 'AND':    return (value_1 == rule['value_1']) and (value_2 == rule['value_2'])
    if operation == 'OR':     return (value_1 == rule['value_1']) or  (value_2 == rule['value_2'])
    raise Exception(f"Invalid rule: {rule}")


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():

    sl_path = hf_hub_download(repo_id="sike25/neural-babel", filename="speaker_listener.pt")
    sl_ckpt = torch.load(sl_path)

    hp      = sl_ckpt['hyperparameters']
    speaker_listener = SpeakerListenerSystem(
        world_size          = hp['world_size'],
        feature_dimension   = hp['object_feature_dimension'],
        neuralese_dimension = hp['neuralese_dimension'],
    )
    speaker_listener.load_state_dict(sl_ckpt['model_state_dict'])
    speaker_listener.eval()
    for p in speaker_listener.parameters():
        p.requires_grad = False

    tr_path = hf_hub_download(repo_id="sike25/neural-babel", filename="lstm_translator.pt")
    tr_ckpt = torch.load(tr_path)

    hp2     = tr_ckpt['hyperparameters']
    translator = Translator(
        neuralese_dimension = hp2['neuralese_dimension'],
        max_rule_length     = hp2['max_rule_length'],
        vocab_size          = hp2['vocab_size'],
        hidden_dimension    = hp2['hidden_dimension'],
    )
    translator.load_state_dict(tr_ckpt['model_state_dict'])
    translator.eval()
    for p in translator.parameters():
        p.requires_grad = False

    return speaker_listener, translator, tr_ckpt['neuralese_mean'], tr_ckpt['neuralese_std']

SPEAKER_LISTENER, TRANSLATOR, NEURO_MEAN, NEURO_STD = load_models()


# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg:       #080c0f;
    --surface:  #0f1519;
    --border:   #1e2d38;
    --accent:   #00e5ff;
    --dim:      #4a6a7a;
    --text:     #c8dde8;
    --green:    #21c55d;
    --red:      #ff4b4b;
    --purple:   #a855f7;
    --amber:    #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main { padding-top: 2rem; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Headings */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: var(--accent) !important;
     letter-spacing: 0.15em; font-size: 1.6rem !important; }
h3 { font-family: 'IBM Plex Mono', monospace !important; color: var(--dim) !important;
     font-size: 0.75rem !important; letter-spacing: 0.2em; text-transform: uppercase;
     margin-bottom: 0.5rem !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em;
    border-radius: 2px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Object cards */
.obj-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--dim);
    transition: border-color 0.3s ease;
}
.obj-card.selected { border-color: var(--accent) !important; }
.obj-card.wrong    { border-color: var(--red)    !important; }
.obj-glyph {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.4rem;
}

/* Section panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.panel.success { border-left-color: var(--green); }
.panel.error   { border-left-color: var(--red);   }
.panel.warn    { border-left-color: var(--amber);  }

/* Neuralese bar */
.neuro-cells { display: flex; gap: 4px; margin: 0.6rem 0; }
.neuro-cell {
    flex: 1;
    height: 32px;
    border-radius: 3px;
}

/* Rule display */
.rule-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    color: var(--accent);
    letter-spacing: 0.05em;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Mask dots */
.mask-dots { display: flex; gap: 6px; margin-top: 0.4rem; }
.mask-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--border);
}
.mask-dot.on  { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.mask-dot.match { background: var(--green); }
.mask-dot.miss  { background: var(--red);   }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────

def object_svg(obj, selected=False, mismatch=False):
    color         = COLOR_HEX[obj.color]
    stroke_width  = {'NONE': 0, 'SLIM': 2, 'THICK': 5}[obj.outline]
    outline_color = 'var(--accent)' if selected else ('var(--red)' if mismatch else 'white')

    if obj.shape == 'CIRCLE':
        shape = f'<circle cx="40" cy="40" r="30" fill="{color}" stroke="{outline_color}" stroke-width="{stroke_width}"/>' 
    elif obj.shape == 'SQUARE':
        shape = f'<rect x="10" y="10" width="60" height="60" fill="{color}" stroke="{outline_color}" stroke-width="{stroke_width}"/>' 
    else:
        shape = f'<polygon points="40,8 72,72 8,72" fill="{color}" stroke="{outline_color}" stroke-width="{stroke_width}"/>' 

    ring = f'<circle cx="40" cy="40" r="38" fill="none" stroke="{outline_color}" stroke-width="2" opacity="0.6"/>' if selected or mismatch else ''
    return f'<svg width="80" height="80" xmlns="http://www.w3.org/2000/svg">{ring}{shape}</svg>'

def neuralese_colors_html(values):
    # Map each value onto a diverging color scale:
    # negative  →  deep indigo  (#1e1b4b)
    # zero      →  dark neutral (#0f1519)
    # positive  →  cyan accent  (#00e5ff)
    mn, mx = min(values), max(values)
    abs_max = max(abs(mn), abs(mx)) or 1

    def lerp(a, b, t):
        return int(a + (b - a) * t)

    def val_to_color(v):
        t = v / abs_max  # -1 .. +1
        if t >= 0:
            # neutral → cyan
            r = lerp(0x0f, 0x00, t)
            g = lerp(0x15, 0xe5, t)
            b = lerp(0x19, 0xff, t)
        else:
            # neutral → indigo
            t = -t
            r = lerp(0x0f, 0x1e, t)
            g = lerp(0x15, 0x1b, t)
            b = lerp(0x19, 0x4b, t)
        return f'#{r:02x}{g:02x}{b:02x}'

    cells = ''
    for v in values:
        color = val_to_color(v)
        cells += f'<div class="neuro-cell" style="background:{color}" title="{v:+.3f}"></div>'
    return f'<div class="neuro-cells">{cells}</div>'

def mask_dots_html(pred, truth=None):
    dots = ''
    for i, p in enumerate(pred):
        if truth is None:
            cls = 'on' if p else ''
        else:
            if truth[i] and p:     cls = 'match'
            elif truth[i] and not p: cls = 'miss'
            elif not truth[i] and p: cls = 'miss'
            else:                    cls = ''
        dots += f'<div class="mask-dot {cls}"></div>'
    return f'<div class="mask-dots">{dots}</div>'


# ── INFERENCE ──────────────────────────────────────────────────────────────────

def run_inference(world, rule):
    mask = [object_matches_rule(rule, obj) for obj in world]
    W    = torch.stack([torch.from_numpy(obj.to_array()) for obj in world]).unsqueeze(0)
    X    = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        Y_logits, _, neuralese = SPEAKER_LISTENER(W, X)

        norm_neuralese   = (neuralese - NEURO_MEAN) / (NEURO_STD + 1e-8)
        predicted_logits = TRANSLATOR(norm_neuralese)
        predicted_tokens = torch.argmax(predicted_logits, dim=2)
        decoded_tokens   = [INDEX_TO_TOKEN[t.item()].upper() for t in predicted_tokens[0] if t.item() != 0]
        decoded_rule     = ' '.join(decoded_tokens)

        predicted_mask = [int(p) for p in (torch.sigmoid(Y_logits) > 0.5).numpy().flatten()]

    return {
        'world':          world,
        'rule':           rule['string'],
        'truth_mask':     [int(m) for m in mask],
        'predicted_mask': predicted_mask,
        'neuralese':      norm_neuralese.squeeze().tolist(),
        'decoded_rule':   decoded_rule,
    }


# ── SESSION STATE ──────────────────────────────────────────────────────────────

for key in ['game', 'show_neuralese', 'show_listener', 'show_translator', 'show_reveal']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'game' else False


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("# 📡 NEURAL BABEL RADIO")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown("""
<section id="landing_hero">
    <p>
    Two neural nets were trained to play a game.
    The first, the speaker, looks at a world of objects,
    picks out the ones that fit a secret rule,
    and sends a private message to the other, the listener.
    The listener uses only that message — without having seen the rule —
    to find the right objects.
    </p>
    <p>
    A third network, the translator, tries to decode that message into English.
    </p>
</section>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── GENERATE ──────────────────────────────────────────────────────────────────

if st.button("⟳  Generate Game Instance"):
    world = generate_random_world()
    rule  = None
    mask  = None
    while True:
        rule = generate_random_rule()
        mask = [object_matches_rule(rule, obj) for obj in world]
        if 0 < sum(mask) < WORLD_SIZE:
            break
    st.session_state.game           = run_inference(world, rule)
    st.session_state.show_neuralese = False
    st.session_state.show_listener  = False
    st.session_state.show_translator = False
    st.session_state.show_reveal    = False

game = st.session_state.game

if game:
    # World display
    st.markdown("### ◈ World")
    cols = st.columns(WORLD_SIZE)
    for i, (col, obj) in enumerate(zip(cols, game['world'])):
        with col:
            st.markdown(object_svg(obj), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SPEAKER ───────────────────────────────────────────────────────────────

    st.markdown("### ◈ Speaker")
    if st.button("Encode World + Rule → Neuralese"):
        st.session_state.show_neuralese = True

    if st.session_state.show_neuralese:
        neuro_vals = game['neuralese']
        bars_html  = neuralese_colors_html(neuro_vals)
        vals_str   = "  ".join(f"{v:+.2f}" for v in neuro_vals)
        st.markdown(f"""
        <div class="panel">
            <div style="color:var(--dim); font-size:0.7rem; margin-bottom:0.3rem">NEURALESE VECTOR (dim=12)</div>
            {bars_html}
            <div style="font-size:0.65rem; color:var(--dim); word-break:break-all">{vals_str}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── LISTENER ─────────────────────────────────────────────────────────

        st.markdown("### ◈ Listener")
        if st.button("Select Objects from Neuralese"):
            st.session_state.show_listener = True

        if st.session_state.show_listener:
            pred = game['predicted_mask']
            cols = st.columns(WORLD_SIZE)
            for i, (col, obj) in enumerate(zip(cols, game['world'])):
                with col:
                    st.markdown(object_svg(obj, selected=bool(pred[i])), unsafe_allow_html=True)

            dots = mask_dots_html(pred)
            st.markdown(f"""
            <div class="panel">
                <div style="color:var(--dim); font-size:0.7rem">LISTENER SELECTION</div>
                {dots}
                <div style="margin-top:0.5rem; font-size:0.8rem">
                    {"  ".join(str(p) for p in pred)}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── TRANSLATOR ───────────────────────────────────────────────────

            st.markdown("### ◈ Translator")
            if st.button("Decode Neuralese → English"):
                st.session_state.show_translator = True

            if st.session_state.show_translator:
                decoded = game['decoded_rule']
                st.markdown(f"""
                <div class="panel">
                    <div style="color:var(--dim); font-size:0.7rem; margin-bottom:0.4rem">DECODED RULE</div>
                    <span class="rule-text">"{decoded}"</span>
                </div>""", unsafe_allow_html=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── REVEAL ───────────────────────────────────────────────────

                st.markdown("### ◈ Reveal")
                if st.button("Reveal True Rule"):
                    st.session_state.show_reveal = True

                if st.session_state.show_reveal:
                    truth_rule = game['rule']
                    truth_mask = game['truth_mask']
                    pred_mask  = game['predicted_mask']
                    correct    = truth_mask == pred_mask
                    translated_correct = decoded.lower() == truth_rule.lower()

                    panel_cls = 'panel success' if correct else 'panel error'
                    dots      = mask_dots_html(pred_mask, truth_mask)
                    icon      = '✓' if correct else '✗'

                    st.markdown(f"""
                    <div class="{panel_cls}">
                        <div style="color:var(--dim); font-size:0.7rem; margin-bottom:0.4rem">TRUE RULE</div>
                        <span class="rule-text">{truth_rule.upper()}</span>
                        <div style="margin-top:0.8rem; color:var(--dim); font-size:0.7rem">TRUE MASK vs PREDICTED</div>
                        {dots}
                        <div style="margin-top:0.6rem; font-size:0.8rem">
                            Truth:&nbsp;&nbsp; {"  ".join(str(m) for m in truth_mask)}<br>
                            Predicted: {"  ".join(str(m) for m in pred_mask)}
                        </div>
                        <div style="margin-top:0.8rem; font-size:0.9rem; 
                            color: {'var(--green)' if correct else 'var(--red)'}">
                            {icon} Listener was {"correct" if correct else "incorrect"}
                        </div>
                        <div style="margin-top:0.3rem; font-size:0.9rem;
                            color: {'var(--green)' if translated_correct else 'var(--amber)'}">
                            {"✓ Translation matched" if translated_correct else "~ Translation differed"}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("### ◈ True Subset")
                    truth_cols = st.columns(WORLD_SIZE)
                    for i, (col, obj) in enumerate(zip(truth_cols, game['world'])):
                        with col:
                            is_selected  = bool(truth_mask[i])
                            is_wrong     = is_selected != bool(pred_mask[i])
                            st.markdown(object_svg(obj, selected=is_selected, mismatch=is_wrong), unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<section id="landing_footer">
    <a href="https://github.com/sike25/neural_syntax"
       style="color:var(--accent); font-family:'IBM Plex Mono',monospace;
              font-size:0.85rem; text-decoration:none; letter-spacing:0.05em;">
        → See Code + Research
    </a>
</section>
""", unsafe_allow_html=True)