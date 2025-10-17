import json
import numpy as np

from enum import Enum
from itertools import combinations, product

## OBJECT DEFINITIONS

class Color(Enum):
    RED    = 'Red'
    GREEN  = 'Green'
    PURPLE = 'Purple'

class Shape(Enum):
    CIRCLE   = 'Circle'
    SQUARE   = 'Square'
    TRIANGLE = 'Triangle'

class Outline(Enum):
    NONE  = 'None'
    SLIM  = 'Slim'
    THICK = 'Thick'

class Attribute(Enum):
    COLOR   = 'color'
    SHAPE   = 'shape'
    OUTLINE = 'outline'

class Operation(Enum):
    AND = 'AND'
    NOT = 'NOT'
    OR  = 'OR'
    SINGLE = 'SINGLE'

class Object():
    def __init__(self, color, shape, outline):
        self.color = color
        self.shape = shape
        self.outline = outline

    def __repr__(self):
        return f"OBJ({self.color.value}-{self.shape.value}-{self.outline.value})"
    
    def __eq__(self, other):
        return (self.color   == other.color and 
                self.shape   == other.shape and 
                self.outline == other.outline)
    
    def __hash__(self):
        return hash((self.color, self.shape, self.outline))

class Rule():
    def __init__(self, operation, att_1, val_1, att_2=None, val_2=None):
        self.operation = operation
        self.att_1     = att_1  # Attribute enum
        self.val_1     = val_1  # Color/Shape/Outline enum
        self.att_2     = att_2  # Attribute enum (optional)
        self.val_2     = val_2  # Color/Shape/Outline enum (optional)

    def __repr__(self):
        if self.operation == Operation.NOT:
            return f"NOT {self.att_1.value}={self.val_1.value}"
        elif self.operation == Operation.SINGLE:
            return f"{self.att_1.value}={self.val_1.value}"
        else:
            return f"{self.att_1.value}={self.val_1.value} {self.operation.value} {self.att_2.value}={self.val_2.value}"
    
    def matches(self, obj):
        """Check if an object satisfies this rule"""
        val_1 = getattr(obj, self.att_1.value)
        
        if self.operation == Operation.NOT:
            return val_1 != self.val_1
        elif self.operation == Operation.SINGLE:
            return val_1 == self.val_1
        elif self.operation == Operation.AND:
            val_2 = getattr(obj, self.att_2.value)
            return val_1 == self.val_1 and val_2 == self.val_2
        elif self.operation == Operation.OR:
            val_2 = getattr(obj, self.att_2.value)
            return val_1 == self.val_1 or val_2 == self.val_2
        
        return False
    


## DATACLASS FOR DATASET

class DatasetGenerator():
    def __init__(self):
        self.or_rules     = []
        self.and_rules    = []
        self.not_rules    = []
        self.single_rules = []
        self.worlds       = []

        self.id      = 0
        self.W       = {} # id -> List[Object]
        self.X       = {} # id -> List[Object]
        self.labels  = {} # id -> Rule



## COMPREHENSIVE RULE GENERATION

def generate_single_rules():
    """Generate rules for single attribute-value pairs"""
    rules = []
    
    for attr, values in [(Attribute.COLOR, Color), 
                         (Attribute.SHAPE, Shape), 
                         (Attribute.OUTLINE, Outline)]:
        for value in values:
            rules.append(Rule(Operation.SINGLE, attr, value))
    
    return rules

def generate_not_rules():
    """Generate NOT rules for each attribute-value pair"""
    rules = []
    
    for attr, values in [(Attribute.COLOR, Color), 
                         (Attribute.SHAPE, Shape), 
                         (Attribute.OUTLINE, Outline)]:
        for value in values:
            rules.append(Rule(Operation.NOT, attr, value))
    
    return rules

def generate_or_rules():
    """Generate OR rules combining two different attributes"""
    rules = []
    attrs = [Attribute.COLOR, Attribute.SHAPE, Attribute.OUTLINE]
    value_map = {Attribute.COLOR: Color, Attribute.SHAPE: Shape, Attribute.OUTLINE: Outline}
    
    # Pick pairs of different attributes
    for att_1, att_2 in combinations(attrs, 2):
        values1 = list(value_map[att_1])
        values2 = list(value_map[att_2])
        
        # All combinations of values from these two attributes
        for val_1, val_2 in product(values1, values2):
            rules.append(Rule(Operation.OR, att_1, val_1, att_2, val_2))
    
    return rules

def generate_and_rules():
    """Generate AND rules combining two different attributes"""
    rules = []
    attrs = [Attribute.COLOR, Attribute.SHAPE, Attribute.OUTLINE]
    value_map = {Attribute.COLOR: Color, Attribute.SHAPE: Shape, Attribute.OUTLINE: Outline}
    
    # Pick pairs of different attributes
    for att_1, att_2 in combinations(attrs, 2):
        values1 = list(value_map[att_1])
        values2 = list(value_map[att_2])
        
        # All combinations of values from these two attributes
        for val_1, val_2 in product(values1, values2):
            rules.append(Rule(Operation.AND, att_1, val_1, att_2, val_2))
    
    return rules



## DATASET GENERATION

def generate_worlds(world_size=5):
    """Generate all possible worlds of given size"""
    # Create all possible objects
    all_objects = []
    for color in Color:
        for shape in Shape:
            for outline in Outline:
                all_objects.append(Object(color, shape, outline))
    
    # Generate all combinations of world_size objects
    worlds = list(combinations(all_objects, world_size))
    return worlds

def generate_dataset(world_size=5):
    """
    Generate dataset of datagenerator objects (fields: worlds, subsets, rules).
    
    Args:
        world_size: Number of objects per world
    """
    dg = DatasetGenerator()
    
    # Generate all rules
    dg.single_rules = generate_single_rules()
    dg.not_rules    = generate_not_rules()
    dg.or_rules     = generate_or_rules()
    dg.and_rules    = generate_and_rules()
    
    all_rules = dg.single_rules + dg.not_rules + dg.or_rules + dg.and_rules
    
    print(f"Generated {len(dg.single_rules)} SINGLE rules")
    print(f"Generated {len(dg.not_rules)}  NOT rules")
    print(f"Generated {len(dg.or_rules)} OR rules")
    print(f"Generated {len(dg.and_rules)} AND rules")
    print(f"Generated {len(all_rules)} TOTAL rules")
    
    # Generate all worlds
    dg.worlds = generate_worlds(world_size)
    print(f"Generated {len(dg.worlds)} unique worlds")
    
    # Generate dataset
    for world in dg.worlds:
        seen_subsets = set()
        for rule in all_rules:

            # Find subset X that satisfies the rule
            X = [obj for obj in world if rule.matches(obj)]

            
            # Skip if X is empty or X equals W (not interesting)
            if len(X) == 0 or len(X) == len(world):
                continue

            # Skip if we've seen this subset for this world already
            X_str = X.__repr__()
            if X_str in seen_subsets:
                continue
            seen_subsets.add(X_str)
            
            # Store the sample
            dg.W[dg.id] = list(world)
            dg.X[dg.id] = X
            dg.labels[dg.id] = rule
            dg.id += 1            

    
    print(f"\nGenerated {dg.id} dataset entries")
    return dg



## CONVERSION TO FILEDUMP

def object_to_array(object):
    """
    Convert object to 2D one-hot array (3, 3)
    Each row is one-hot encoding for [color, shape, outline]
    Object (RED, CIRCLE, SLIM) becomes:
    [[1,0,0], # RED    
    [1,0,0],  # CIRCLE
    [0,1,0]]  # SLIM
    """
    # Create index mappings
    color_to_idx   = {Color.RED: 0, Color.GREEN: 1, Color.PURPLE: 2}
    shape_to_idx   = {Shape.CIRCLE: 0, Shape.SQUARE: 1, Shape.TRIANGLE: 2}
    outline_to_idx = {Outline.NONE: 0, Outline.SLIM: 1, Outline.THICK: 2}
    
    # Create one-hot encodings
    arr = np.zeros((3, 3), dtype=np.float32)
    arr[0, color_to_idx[object.color]]     = 1
    arr[1, shape_to_idx[object.shape]]     = 1
    arr[2, outline_to_idx[object.outline]] = 1
    
    return arr

def rule_to_natural_language(rule):
    """Convert rule to natural language string"""
    color_map   = {Color.RED: 'red', Color.GREEN: 'green', Color.PURPLE: 'purple'}
    shape_map   = {Shape.CIRCLE: 'circle', Shape.SQUARE: 'square', Shape.TRIANGLE: 'triangle'}
    outline_map = {Outline.NONE: 'no-outline', Outline.SLIM: 'slim-outline', Outline.THICK: 'thick-outline'}
    
    value_maps = {
        Attribute.COLOR   : color_map,
        Attribute.SHAPE   : shape_map,
        Attribute.OUTLINE : outline_map
    }
    
    val_1_str = value_maps[rule.att_1][rule.val_1]
    
    if rule.operation == Operation.NOT:
        return f"not {val_1_str}"
    elif rule.operation == Operation.SINGLE:
        return f"{val_1_str}"
    elif rule.operation == Operation.AND:
        val_2_str = value_maps[rule.att_2][rule.val_2]
        return f"{val_1_str} and {val_2_str}"
    elif rule.operation == Operation.OR:
        val_2_str = value_maps[rule.att_2][rule.val_2]
        return f"{val_1_str} or {val_2_str}"
    
    return ""


def rule_to_encoding(rule):
    """
    Encode rule as a structured array for neural network input
    Returns: dict with operation type and relevant indices
    """
    color_to_idx   = {Color.RED: 0, Color.GREEN: 1, Color.PURPLE: 2}
    shape_to_idx   = {Shape.CIRCLE: 0, Shape.SQUARE: 1, Shape.TRIANGLE: 2}
    outline_to_idx = {Outline.NONE: 0, Outline.SLIM: 1, Outline.THICK: 2}
    
    attr_to_idx = {Attribute.COLOR: 0, Attribute.SHAPE: 1, Attribute.OUTLINE: 2}
    op_to_idx   = {Operation.SINGLE: 0, Operation.NOT: 1, Operation.AND: 2, Operation.OR: 3}
    
    value_maps = {
        Attribute.COLOR: color_to_idx,
        Attribute.SHAPE: shape_to_idx,
        Attribute.OUTLINE: outline_to_idx
    }
    
    encoding = {
        'operation': op_to_idx[rule.operation],
        'att_1'    : attr_to_idx[rule.att_1],
        'val_1'    : value_maps[rule.att_1][rule.val_1],
        'att_2'    : attr_to_idx[rule.att_2] if rule.att_2 else -1,
        'val_2'    : value_maps[rule.att_2][rule.val_2] if rule.att_2 else -1
    }
    
    return encoding

def dataset_to_npy(dg, output_path='dataset.npz'):
    """
    Convert dataset to NPY format
    
    Saves a single .npz file containing:
    - W: (N, world_size, 3, 3) - all worlds as one-hot arrays
    - X_mask: (N, world_size)  - boolean mask indicating which objects are in subset X
    - rule_encodings: (N, 5)   - encoded rules [operation, att_1, val_1, att_2, val_2]
    - rule_texts: (N,)         - natural language rule strings
    
    Also saves metadata as JSON
    """
    N = dg.id
    world_size = len(dg.W[0]) if N > 0 else 0
    
    # Initialize arrays
    W = np.zeros((N, world_size, 3, 3), dtype=np.float32)
    X_mask = np.zeros((N, world_size), dtype=bool)
    rule_encodings = np.zeros((N, 5), dtype=np.int32)
    rule_texts = []
    
    print("Converting dataset to numpy arrays...")
    for i in range(N):
        # Convert world objects to arrays
        for j, obj in enumerate(dg.W[i]):
            W[i, j] = object_to_array(obj)
            # Check if this object is in subset X
            X_mask[i, j] = obj in dg.X[i]
        
        # Encode rule
        rule_enc = rule_to_encoding(dg.labels[i])
        rule_encodings[i] = [
            rule_enc['operation'],
            rule_enc['att_1'],
            rule_enc['val_1'],
            rule_enc['att_2'],
            rule_enc['val_2']
        ]
        
        # Natural language rule
        rule_texts.append(rule_to_natural_language(dg.labels[i]))
    
    # Save as compressed npz
    np.savez_compressed(
        output_path,
        W=W,
        X_mask=X_mask,
        rule_encodings=rule_encodings,
        rule_texts=np.array(rule_texts, dtype=object)
    )
    
    print(f"Saved dataset to {output_path}")
    print(f"  W shape: {W.shape}")
    print(f"  X_mask shape: {X_mask.shape}")
    print(f"  rule_encodings shape: {rule_encodings.shape}")
    
    # Save metadata
    metadata = {
        'num_entries': N,
        'world_size': world_size,
        'num_colors': 3,
        'num_shapes': 3,
        'num_outlines': 3,
        'total_unique_objects': 27,
        'encoding_info': {
            'operation_indices': {'SINGLE': 0, 'NOT': 1, 'AND': 2, 'OR': 3},
            'attribute_indices': {'COLOR': 0, 'SHAPE': 1, 'OUTLINE': 2},
            'color_indices': {'RED': 0, 'GREEN': 1, 'PURPLE': 2},
            'shape_indices': {'CIRCLE': 0, 'SQUARE': 1, 'TRIANGLE': 2},
            'outline_indices': {'NONE': 0, 'SLIM': 1, 'THICK': 2}
        }
    }
    
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    dg = generate_dataset(world_size=5)
    dataset_to_npy(dg, output_path='dataset.npz')
    
    # # Show a few examples
    # print("\n=== Sample entries ===")
    # for i in range(min(5, dg.id)):
    #     print(f"\nEntry {i}:")
    #     print(f"  Rule: {dg.labels[i]}")
    #     print(f"  World W ({len(dg.W[i])} objects): {dg.W[i]}")
    #     print(f"  Subset X ({len(dg.X[i])} objects): {dg.X[i]}")