import torch
import numpy as np
from torch.nn.functional import softmax
from .model_loader import LSTMService

def prepare_sequence_for_lstm(keypoints: np.ndarray):
    # keypoints shape: (frames, 33, 3)
    frames, n_points, n_coords = keypoints.shape
    sequence = keypoints.reshape(frames, n_points * n_coords)
    return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

def compute_similarity_score(pred_logits, target_label_idx):
    probs = softmax(pred_logits, dim=1)[0]
    return probs[target_label_idx].item()

def predict_action_from_keypoints(keypoints, expected_action):
    model, idx_to_label = LSTMService.load_model()

    x = prepare_sequence_for_lstm(keypoints)
    logits = model(x)
    probs = softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_label = idx_to_label[pred_idx]
    confidence = probs[pred_idx].item()

    # Similarity (accuracy) = probabilité de l’action attendue
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    if expected_action in label_to_idx:
        expected_idx = label_to_idx[expected_action]
        similarity = probs[expected_idx].item()
    else:
        similarity = 0.0

    return {
        "predicted_action": pred_label,
        "confidence": round(confidence, 4),
        "similarity_to_expected": round(similarity, 4),
        "probabilities": {lbl: float(probs[i]) for i, lbl in idx_to_label.items()}
    }
