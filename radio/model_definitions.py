
### MODEL DEFINITIONS

import torch
import torch.nn as nn

WORLD_SIZE               = 5
OBJECT_FEATURE_DIMENSION = 6
NEURALESE_DIMENSION      = 12

class ObjectEncoder(nn.Module):
    """
    Encodes a single (3x3) Object into a feature vector
    Input     : [[001][100][010]] or Purple-Cirle-No-Outline. See metadata.json
    InputSize : (3,3)
    OutputSize: (OBJECT_FEATURE_DIMENSION)
    """
    def __init__(self, output_dimension=OBJECT_FEATURE_DIMENSION):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2),  # → (batch, 4, 2, 2)
            nn.Flatten(),                    # → (batch, 16)
            nn.Linear(16, output_dimension)
        )

    def forward(self, x):
        return self.encoder(x)
    


class Speaker(nn.Module):
    """
    Transforms the encoded objects in world W + boolean inclusion mask for target subset X
    into a representative vector.

    InputSize : (WORLD_SIZE * OBJECT_FEATURE_DIMENSION) + WORLD_SIZE)
    OutputSize: (NEURALESE_DIMENSION)
    """
    def __init__(self, input_dimension=(WORLD_SIZE * OBJECT_FEATURE_DIMENSION) + WORLD_SIZE,
                 output_dimension=NEURALESE_DIMENSION):
        super().__init__()
        self.speaker_net = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dimension)
        )

    def forward(self, x):
        return self.speaker_net(x)
    


class Listener(nn.Module):
    """
    Takes the resultant vector from the speaker, along with W_i (an element of the world W)
    and predicts whether this element belongs to X, the target subset.

    InputSize : (NEURALESE_DIMENSION + OBJECT_FEATURE_DIMENSION)
    OutputSize: (1) -> logit for binary classification
    """
    def __init__(self, input_dimension=NEURALESE_DIMENSION + OBJECT_FEATURE_DIMENSION,
                 output_dimension=1):
        super().__init__()
        self.listener_net = nn.Sequential(
            nn.Linear(input_dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_dimension)
        )

    def forward(self, x):
        return self.listener_net(x)
    


class SpeakerListenerSystem(nn.Module):
    """
    The end-to-end system.
    Encoder -> Speaker -> Listener
    """
    def __init__(self, world_size, feature_dimension, neuralese_dimension):
        super().__init__()

        self.world_size          = world_size
        self.feature_dimension   = feature_dimension
        self.neuralese_dimension = neuralese_dimension

        speaker_input_size  = (self.world_size * self.feature_dimension) + self.world_size
        listener_input_size = self.neuralese_dimension + self.feature_dimension

        self.encoder  = ObjectEncoder(output_dimension=self.feature_dimension)
        self.speaker  = Speaker(input_dimension=speaker_input_size, output_dimension=neuralese_dimension)
        self.listener = Listener(input_dimension=listener_input_size, output_dimension=1)


    def forward(self, W, X_mask, return_neuralese_only=False):
        """
        W:      A batch of worlds.
                Each world is a set of (3x3) objects.
                Tensor of shape (batch_size, world_size, 3, 3).

        X_mask: A batch of boolean masks
                Each value indicated whether the object at that index in W
                is included in the target subset X
                Tensor of shape (batch_size, world_size)
        """
        batch_size = W.shape[0]

        ### STEP 1: Encode all objects in the world
        # Reshape for batch processing ny the encoder (B, 5, 3, 3) -> (B*5, 1, 3, 3)
        W_flat = W.view(-1, 1, 3, 3)
        # Get features for all objects: (B*5, feature_dim)
        object_features_flat = self.encoder(W_flat)
        # Reshape back to per-batch item:(B, 5, feature_dim)
        object_features = object_features_flat.view(batch_size, self.world_size, self.feature_dimension)


        ### STEP 2: Assemble the inputs to the speaker model
        # Flatten object features: (B, 5, feature_dim) -> (B, 5*feature_dim)
        V_W =  object_features.view(batch_size, -1)
        # Create the indicator mask
        M_X = X_mask
        # Concatenate features and mask
        speaker_input = torch.cat([V_W, M_X], dim=1)


        ### STEP 3: Speaker generates neuralese
        # representation has shape (B, neuralese_dimension)
        representation = self.speaker(speaker_input)

        if return_neuralese_only:
            return representation

        ### STEP 4: Prepare the listener's input
        # The listener needs to pair the speaker's representation with each object feature
        # Expand the representation to match the number of objects
        # (B, rep_dim) -> (B, 1, rep_dim) -> (B, world_size, rep_dim)
        r_expanded = representation.unsqueeze(1).repeat(1, self.world_size, 1)
        # Concatenate with object features: (B, 5, rep_dim) + (B, 5, feature_dim)
        listener_input = torch.cat([r_expanded, object_features], dim=2)


        ### STEP 5: Shuffle inputs to the listener
        # This is to avoid the speaker simply learning to tell the listener about X_mask
        # without learning anything about the objects in X themselves.
        # Create a random permutation for each item in the batch
        shuffled_indices = [torch.randperm(self.world_size) for _ in range(batch_size)]
        # Apply the shuffle
        shuffled_input   = torch.stack([features[p] for features, p in zip(listener_input, shuffled_indices)])
        shuffled_labels  = torch.stack([labels[p]   for labels,   p in zip(X_mask, shuffled_indices)])


        ### STEP 6: Listener makes an inclusion prediction for each object
        # Reshape for batch processing by the listener
        # (B, 5, rep_dim + feature_dim) -> (B*5, rep_dim + feature_dim)
        listener_input_flat = shuffled_input.view(-1, self.neuralese_dimension + self.feature_dimension)
        # Get predictions (logits) -> (B*5, 1)
        predictions_flat = self.listener(listener_input_flat)
        # Reshape back to (B, world_size)
        predictions = predictions_flat.view(batch_size, self.world_size)

        return predictions, shuffled_labels
    


class Translator(nn.Module):
    """
    Translates a neuralese vector V (produced by the speaker) into a sequence of rule tokens
    using an autoregressive LSTM decoder.

    The neuralese vector initializes the hidden state.
    Tokens are generated one at a time, each conditioned on the previous.
    """
    def __init__(self,
                 neuralese_dimension = NEURALESE_DIMENSION,
                 vocab_size          = 13,
                 embed_dimension     = 32,
                 hidden_dimension    = 128,
                 max_rule_length     = 3):
        super().__init__()

        self.vocab_size      = vocab_size
        self.max_rule_length = max_rule_length
        self.hidden_dim      = hidden_dimension

        # Project neuralese vector into LSTM initial hidden + cell states
        self.init_hidden = nn.Linear(neuralese_dimension, hidden_dimension)
        self.init_cell   = nn.Linear(neuralese_dimension, hidden_dimension)

        # Learned embedding for each token in the vocabulary
        # vocab_size + 1 to include a <START> token
        self.embedding = nn.Embedding(vocab_size + 1, embed_dimension)
        self.start_token_idx = vocab_size  # <START> sits just outside the vocabulary

        # The LSTM cell — takes an embedded token, outputs a new hidden state
        self.lstm_cell = nn.LSTMCell(embed_dimension, hidden_dimension)

        # Project hidden state to logits over the vocabulary
        self.output_projection = nn.Linear(hidden_dimension, vocab_size)


    def forward(self, V, target_tokens=None):
        """
        V:             Neuralese vectors.          (batch_size, neuralese_dimension)
        target_tokens: Ground truth token indices. (batch_size, max_rule_length)
                       If provided, uses teacher forcing (training mode).
                       If None, generates autoregressively (inference mode).

        Returns:
        logits:        (batch_size, max_rule_length, vocab_size)
        """
        batch_size = V.shape[0]

        # Initialize hidden and cell states from the neuralese vector
        h = torch.tanh(self.init_hidden(V))  # (batch_size, hidden_dim)
        c = torch.tanh(self.init_cell(V))    # (batch_size, hidden_dim)

        # Kick off the sequence with the <START> token
        start_tokens = torch.full((batch_size,), self.start_token_idx,
                                  dtype=torch.long, device=V.device)
        current_input = self.embedding(start_tokens)  # (batch_size, embed_dim)

        logits = []

        for t in range(self.max_rule_length):

            # One LSTM step
            h, c = self.lstm_cell(current_input, (h, c))

            # Project to vocabulary logits
            step_logits = self.output_projection(h)  # (batch_size, vocab_size)
            logits.append(step_logits)

            # Decide next input: teacher forcing during training, argmax during inference
            if target_tokens is not None:
                current_input = self.embedding(target_tokens[:, t])
            else:
                predicted = torch.argmax(step_logits, dim=1)
                current_input = self.embedding(predicted)

        return torch.stack(logits, dim=1)  # (batch_size, max_rule_length, vocab_size)