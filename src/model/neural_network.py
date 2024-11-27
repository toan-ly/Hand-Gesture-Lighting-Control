import torch
import torch.nn as nn
import yaml
from src.utils.label_dict import label_dict_from_config_file

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("./data/hand_gesture.yaml")
        self.output_dim = len(list_label)

        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Layer 4
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            # Output layer
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

    def predict(self, x, threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob, dim=1)
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)

    def predict_with_known_class(self, x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob, dim=1)

    def score(self, logits):
        return -torch.amax(logits, dim=1)
