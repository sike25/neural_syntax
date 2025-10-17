import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def load_dataset(path='dataset.npz'):
    """Load the dataset from npz file"""
    data = np.load(path, allow_pickle=True)
    
    W = data['W']
    X_mask = data['X_mask']
    rule_encodings = data['rule_encodings']
    rule_texts = data['rule_texts']
    
    # Load metadata
    metadata_path = path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return W, X_mask, rule_encodings, rule_texts, metadata

def array_to_object_properties(arr, metadata):
    """Convert one-hot array back to object properties"""
    encoding = metadata['encoding_info']
    
    # Reverse the encoding dictionaries
    colors = {v: k for k, v in encoding['color_indices'].items()}
    shapes = {v: k for k, v in encoding['shape_indices'].items()}
    outlines = {v: k for k, v in encoding['outline_indices'].items()}
    
    color_idx = np.argmax(arr[0])
    shape_idx = np.argmax(arr[1])
    outline_idx = np.argmax(arr[2])
    
    return {
        'color': colors[color_idx],
        'shape': shapes[shape_idx],
        'outline': outlines[outline_idx]
    }

def draw_object(ax, obj_props, x, y, size=0.8):
    """Draw a single object"""
    color_map = {'RED': '#E74C3C', 'GREEN': '#2ECC71', 'PURPLE': '#9B59B6'}
    shape_map = {'CIRCLE': 'circle', 'SQUARE': 'square', 'TRIANGLE': 'triangle'}
    
    color = color_map[obj_props['color']]
    shape = shape_map[obj_props['shape']]
    outline = obj_props['outline']
    
    # Determine edge properties
    if outline == 'NONE':
        edgecolor = color
        linewidth = 0
    elif outline == 'SLIM':
        edgecolor = 'black'
        linewidth = 1
    else:  # THICK
        edgecolor = 'black'
        linewidth = 4
    
    # Draw shape
    if shape == 'circle':
        circle = patches.Circle((x, y), size/2, facecolor=color, 
                               edgecolor=edgecolor, linewidth=linewidth)
        ax.add_patch(circle)
    elif shape == 'square':
        square = patches.Rectangle((x - size/2, y - size/2), size, size,
                                   facecolor=color, edgecolor=edgecolor, 
                                   linewidth=linewidth)
        ax.add_patch(square)
    elif shape == 'triangle':
        triangle = patches.RegularPolygon((x, y), 3, radius=size/2,
                                         facecolor=color, edgecolor=edgecolor,
                                         linewidth=linewidth)
        ax.add_patch(triangle)

def visualize_entry(W, X_mask, rule_text, idx, metadata):
    """Visualize a single dataset entry"""
    world = W[idx]
    mask = X_mask[idx]
    world_size = len(world)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    
    # Set up both axes
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, world_size)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Draw W (all objects) on first line
    for i, obj_arr in enumerate(world):
        obj_props = array_to_object_properties(obj_arr, metadata)
        draw_object(ax1, obj_props, i + 0.5, 0.5)
    
    ax1.text(-0.3, 0.5, 'W:', fontsize=16, fontweight='bold', 
             verticalalignment='center')
    
    # Draw X (subset) on second line
    x_objects = [world[i] for i in range(world_size) if mask[i]]
    for i, obj_arr in enumerate(x_objects):
        obj_props = array_to_object_properties(obj_arr, metadata)
        draw_object(ax2, obj_props, i + 0.5, 0.5)
    
    ax2.text(-0.3, 0.5, 'X:', fontsize=16, fontweight='bold', 
             verticalalignment='center')
    
    # Add rule text at bottom
    rule_text_str = f"Rule: {rule_text}"
    fig.text(0.5, 0.02, rule_text_str, ha='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Dataset Entry #{idx}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.95)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    W, X_mask, rule_encodings, rule_texts, metadata = load_dataset('dataset.npz')
    
    print(f"Dataset loaded: {len(W)} entries")
    print(f"World size: {len(W[0])} objects per world")
    
    # Pick random entry
    random_idx = random.randint(0, len(W) - 1)
    print(f"\nVisualizing random entry #{random_idx}")
    print(f"Rule: {rule_texts[random_idx]}")
    
    # Visualize
    visualize_entry(W, X_mask, rule_texts[random_idx], random_idx, metadata)