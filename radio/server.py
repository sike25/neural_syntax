

import numpy as np
import random
import torch
from model_definitions import SpeakerListenerSystem, Translator


### CONSTANTS ----------------------------------------

WORLD_SIZE               = 5
OBJECT_FEATURE_DIMENSION = 6
NEURALESE_DIMENSION      = 12

COLORS   = ['RED', 'GREEN', 'PURPLE']
SHAPES   = ['CIRCLE', 'SQUARE', 'TRIANGLE']
OUTLINES = ['NONE', 'SLIM', 'THICK']

TOKEN_TO_INDEX = {
    '<blank>': 0, 'not': 1, 'and': 2, 'or': 3, 'red': 4, 'green': 5, 'purple': 6,
    'circle': 7, 'square': 8, 'triangle': 9, 'no-outline': 10, 'slim-outline': 11, 'thick-outline': 12
}
INDEX_TO_TOKEN = {v: k for k, v in TOKEN_TO_INDEX.items()}


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
    

class GameState:

    def __init__(self, world, mask, rule, neuralese, predicted_mask, predicted_rule):
        self.world          = world
        self.rule           = rule
        self.mask           = mask
        self.neuralese      = neuralese
        self.predicted_rule = predicted_rule
        self.predicted_mask = predicted_mask


### FUNCTIONALITY -------------------------------------

def generate_random_world():
    return [ObjectInstance(random.choice(COLORS), random.choice(SHAPES), random.choice(OUTLINES)) for _ in range(WORLD_SIZE)]


def generate_random_rule():
    values_map  = {'color': COLORS, 'shape': SHAPES, 'outline': OUTLINES}
    attributes  = ['color', 'shape', 'outline']

    operation   = random.choice(['SINGLE', 'NOT', 'AND', 'OR'])
    attribute_1 = random.choice(attributes)
    value_1     = random.choice(values_map[attribute_1])

    rule = {'operation': operation, 
            'attribute': attribute_1, 
            'value_1'  : value_1}
    
    if operation in ['AND', 'OR']:
        attribute_2         = random.choice([a for a in attributes if a != attribute_1])
        rule['attribute_2'] = attribute_2
        rule['value_2']     = attribute_2, random.choice(values_map[attribute_2])
        rule['string']      = f"{attribute_1} {operation} {attribute_2}"
    elif operation == 'NOT':
        rule['string']      = f"{operation} {attribute_1}"
    else: # operation == SINGLE
        rule['string']      = attribute_1

    return rule

def object_matches_rule(rule, object):
    operation = rule['operation']

    value_1 = getattr(object, rule['attribute_1'])
    if operation == 'SINGLE': return value_1 == rule['value_1']
    if operation == 'NOT':    return value_1 != rule['value_1']

    value_2 = getattr(object, rule['attribute_2'])
    if operation == 'AND':
        return (value_1 == rule['value_1']) and (value_2 == rule['value_2'])
    if operation == 'OR':
        return (value_1 == rule['value_1']) or (value_2 == rule['value_2'])
    
    raise Exception(f"Invalid rule: {rule}")


def load_speaker_listener():
    checkpoint = torch.load('../training/models/speaker_listener.pt')
    hp = checkpoint['hyperparameters']
    speaker_listener = SpeakerListenerSystem(
        world_size          = hp['world_size'],
        feature_dimension   = hp['object_feature_dimension'],
        neuralese_dimension = hp['neuralese_dimension'],
    )
    speaker_listener.load_state_dict(checkpoint['model_state_dict'])
    speaker_listener.eval()
    for param in speaker_listener.parameters():
        param.requires_grad = False
    print("Speaker-listener model loaded and frozen.")
    return speaker_listener

def load_translator():
    pass



# generate_game_instance: W, X
# encode_to_neuralese: W, X -> N
# select_objects: W, N -> X
# translate_neuralese: N -> R

def play_game(game):

    # generate game instance (W, X_truth)
    world = generate_random_world()
    rule, mask = None, None
    while True:
        rule = generate_random_rule()
        mask = [object_matches_rule(rule, object) for object in world]
        if sum(mask) > 0 and sum(mask) < 5:
            break

    # load the models
    speaker_listener      = load_speaker_listener()
    translator, MEAN, STD = load_translator()

    # encode to neuralese
    W = torch.stack([torch.from_numpy(obj.to_array()) for obj in world]) # (1, 5, 3, 3)
    X = mask # (1, 5)
    neuralese = speaker_listener(W, X, return_neuralese_only=True) # (1, 12)

    # translate neuralese
    norm_neuralese    = (neuralese - MEAN) / (STD + 1e-8)
    predicted_logits  = translator(norm_neuralese) # (1, 3, 13)
    predicted_tokens  = torch.argmax(predicted_logits, dim=2) # (1, 3)
    decoded_tokens    = [INDEX_TO_TOKEN[t.item()].upper() for t in predicted_tokens[0] if t.item() != 0]
    decoded_rule      = ""
    for token in decoded_tokens:
        decoded_rule += token + " "

    # select objects via listener
    Y_logits, Y_truth = speaker_listener(W, X)
    Y_truth_mask      = [int(l) for l in Y_truth.numpy().flatten()]
    Y_predicted_mask  = [int(p) for p in torch.sigmoid(Y_logits).numpy().flatten()]

    # save results
    game.world          = world
    game.rule           = rule['string']
    game.mask           = Y_truth_mask
    game.neuralese      = norm_neuralese
    game.predicted_rule = decoded_rule
    game.predicted_mask = Y_predicted_mask

    

    


game_state = GameState()
    

