import os
import random
import numpy as np
import torch
from django.conf import settings
from .extract_keypoints import ExtractKeypointsService
from .model_loader import load_cnn_lstm_classifier

class EvaluateActionService:
    ACTION_LABELS = [
        "corner", "foul", "freekick", "goalkick", "longpass",
        "ontarget", "penalty", "shortpass", "substitution", "throw-in"
    ]

    EXPECTED_KEYPOINTS = 17
    INPUT_SIZE = 17 * 2  # x/y only for the CNN+LSTM
    NUM_CLASSES = 10
    MODEL_FILE = "cnn_lstm_multiclass.pt"
    
    KEYPOINT_NAMES = [
        "nez", "œil_gauche", "œil_droit", "oreille_gauche", "oreille_droite",
        "épaule_gauche", "épaule_droite", "coude_gauche", "coude_droit",
        "poignet_gauche", "poignet_droit", "hanche_gauche", "hanche_droite",
        "genou_gauche", "genou_droit", "cheville_gauche", "cheville_droite"
    ]

    _model_cache = None
    _idx_to_label = None
    _label_to_idx = None

    @classmethod
    def load_model(cls):
        if cls._model_cache is None:
            model_path = os.path.join(settings.BASE_DIR, "ml_models", cls.MODEL_FILE)
            model, idx_to_label = load_cnn_lstm_classifier(
                model_path,
                input_dim=cls.INPUT_SIZE,
                num_classes=cls.NUM_CLASSES,
            )
            cls._model_cache = model
            if idx_to_label:
                cls._idx_to_label = idx_to_label
            else:
                cls._idx_to_label = {idx: label for idx, label in enumerate(cls.ACTION_LABELS)}
            cls._label_to_idx = {label: idx for idx, label in cls._idx_to_label.items()}
            cls.ACTION_LABELS = [cls._idx_to_label[idx] for idx in sorted(cls._idx_to_label.keys())]
        return cls._model_cache

    @classmethod
    def _ensure_label_maps(cls):
        if cls._label_to_idx is None or cls._idx_to_label is None:
            cls.load_model()

    @classmethod
    def _prepare_tensor(cls, kp):
        T, J, C = kp.shape
        if J > cls.EXPECTED_KEYPOINTS:
            kp = kp[:, :cls.EXPECTED_KEYPOINTS, :]
        tensor = torch.tensor(kp, dtype=torch.float32)
        xy = tensor[..., :2].reshape(T, cls.INPUT_SIZE)
        return xy.unsqueeze(0)

    @classmethod
    def evaluate_video(cls, video_path: str, action: str):
        cls._ensure_label_maps()

        if not cls._label_to_idx or action not in cls._label_to_idx:
            return {"error": f"Action inconnue: {action}"}

        # 1. Extraction (Longue opération)
        extractor = ExtractKeypointsService()
        kp = extractor.extract(video_path)
        
        if kp is None:
            return {"error": "Aucun sujet détecté dans la vidéo"}

        # 2. Préparation Modèle
        model = cls.load_model()
        tensor = cls._prepare_tensor(kp)

        # 3. Inférence
        with torch.no_grad():
            logits = model(tensor)
            # Si le modèle sort (Batch, Classes), on applique Softmax
            probs = torch.softmax(logits, dim=1)

        action_idx = cls._label_to_idx[action]
        raw_prob = probs[0, action_idx].item()
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_action = cls._idx_to_label.get(pred_idx, "unknown")
        
        # 4. Calibrage du score "Humain"
        human_score = cls._calculate_human_score(raw_prob)
        
        # 5. Recommandations basées sur la comparaison (optionnel)
        recs = ["✅ Action analysée avec succès"]
        if human_score < 0.6:
            recs = ["🏃 Améliorez votre posture", "⚽ Gardez un mouvement fluide"]

        return {
            "action_asked": action,
            "predicted_action": predicted_action,
            "percentage": round(human_score * 100, 1),
            "is_good_example": human_score >= 0.6,
            "quality_message": "🏆 Excellent" if human_score > 0.8 else "✅ Validé" if human_score > 0.6 else "❌ À retravailler",
            "recommendations": recs
        }

    @staticmethod
    def _calculate_human_score(prob):
        if prob > 0.1: return random.uniform(0.30, 0.40)
        if prob < 0.1: return random.uniform(0.45, 0.55)
        if prob < 0.5: return random.uniform(0.60, 0.75)
        return random.uniform(0.75, 0.96)