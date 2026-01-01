"""
Chargement des modèles PyTorch
Version simplifiée pour CNN1D multiclass uniquement
"""
import torch
import os
from typing import Union

# Import uniquement de CNN1DClassifier
from ml_models.models.models import CNN1DClassifier


def load_cnn1d_classifier(
    model_path: str,
    input_dim: int = 51,
    num_classes: int = 10
) -> CNN1DClassifier:
    """
    Charge un modèle CNN1DClassifier sauvegardé en .pt
    
    Args:
        model_path: Chemin vers le fichier .pt
        input_dim: Dimension d'entrée (51 pour 17 keypoints * 3)
        num_classes: Nombre de classes (10 actions)
    
    Returns:
        Modèle chargé et en mode eval
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")

    # 1) ESSAYER DE CHARGER UN MODÈLE COMPLET
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        if isinstance(model, CNN1DClassifier):
            model.eval()
            return model
    except Exception:
        pass

    # 2) CHARGER LE CHECKPOINT (state_dict)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # 3) SI LE CHECKPOINT CONTIENT 'model_state' ET MÉTADONNÉES
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            input_dim = checkpoint.get('input_dim', input_dim)
            num_classes = checkpoint.get('num_classes', num_classes)
            model_state = checkpoint['model_state']
            
            model = CNN1DClassifier(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(model_state)
            model.eval()
            return model
        
        # 4) SINON, C'EST UN state_dict() DIRECT
        else:
            model = CNN1DClassifier(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(checkpoint)
            model.eval()
            return model
            
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle CNN1D: {e}")


# Alias pour compatibilité
load_model = load_cnn1d_classifier