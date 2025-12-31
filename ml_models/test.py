#!/usr/bin/env python
"""
Script de test pour l'API d'évaluation d'actions avec CNN1D
Usage: python test_api.py
"""
import os
import requests
import json
from datetime import datetime


# -------------------------
# CONFIG
# -------------------------
API_URL = "http://127.0.0.1:8000/api/evaluate/"
VIDEO_FILE = "corner.mp4"  # Changez selon votre vidéo
ACTION_TYPE = "corner"      # Actions disponibles: penalty, corner, freekick, etc.

# Liste des actions disponibles pour référence
AVAILABLE_ACTIONS = [
    "corner", "foul", "freekick", "goalkick", "longpass",
    "ontarget", "penalty", "shortpass", "substitution", "throw-in"
]


def print_header(text):
    """Affiche un header formaté"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Affiche une section"""
    print(f"\n📋 {text}")
    print("-" * 70)


def test_api():
    """Teste l'API d'évaluation d'actions"""
    
    print_header("🧪 TEST API - Évaluation d'actions sportives avec CNN1D")
    
    # -------------------------
    # 1. VÉRIFICATION DU FICHIER
    # -------------------------
    print_section("1. Vérification du fichier vidéo")
    
    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Fichier vidéo introuvable : {VIDEO_FILE}")
        print(f"\n💡 Assurez-vous que le fichier existe dans le répertoire courant")
        return False
    
    file_size = os.path.getsize(VIDEO_FILE) / (1024 * 1024)  # en MB
    print(f"✅ Fichier trouvé : {VIDEO_FILE}")
    print(f"   Taille : {file_size:.2f} MB")
    
    # -------------------------
    # 2. VÉRIFICATION DE L'ACTION
    # -------------------------
    print_section("2. Vérification du type d'action")
    
    if ACTION_TYPE not in AVAILABLE_ACTIONS:
        print(f"⚠️  Action '{ACTION_TYPE}' non reconnue")
        print(f"   Actions disponibles : {', '.join(AVAILABLE_ACTIONS)}")
    else:
        print(f"✅ Action valide : {ACTION_TYPE}")
    
    # -------------------------
    # 3. PRÉPARATION DE LA REQUÊTE
    # -------------------------
    print_section("3. Préparation de la requête")
    
    print(f"   URL     : {API_URL}")
    print(f"   Action  : {ACTION_TYPE}")
    print(f"   Vidéo   : {VIDEO_FILE}")
    
    # -------------------------
    # 4. ENVOI DE LA REQUÊTE
    # -------------------------
    print_section("4. Envoi de la requête à l'API")
    
    start_time = datetime.now()
    
    try:
        with open(VIDEO_FILE, "rb") as video_file:
            files = {"video": video_file}
            data = {"action": ACTION_TYPE}
            
            print("⏳ Envoi en cours... (cela peut prendre quelques secondes)")
            
            response = requests.post(
                API_URL,
                data=data,
                files=files,
                timeout=180
            )
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERREUR DE CONNEXION")
        print("   Impossible de se connecter au serveur")
        print("   Vérifiez que Django tourne sur http://127.0.0.1:8000/")
        print("\n💡 Lancez Django avec: python manage.py runserver")
        return False
        
    except requests.exceptions.Timeout:
        print("\n❌ TIMEOUT")
        print("   La requête a expiré (>120s)")
        print("   Le serveur est peut-être surchargé ou la vidéo trop lourde")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE : {e}")
        return False
    
    # -------------------------
    # 5. ANALYSE DE LA RÉPONSE
    # -------------------------
    print_section("5. Analyse de la réponse")
    
    print(f"⏱️  Temps de réponse : {elapsed_time:.2f}s")
    print(f"📊 Statut HTTP : {response.status_code}")
    
    if response.status_code != 200:
        print(f"\n❌ Erreur HTTP {response.status_code}")
        print(f"Réponse brute :")
        print(response.text)
        return False
    
    # -------------------------
    # 6. AFFICHAGE DES RÉSULTATS
    # -------------------------
    try:
        result = response.json()
        
        if "error" in result:
            print(f"\n❌ ERREUR SERVEUR : {result['error']}")
            return False
        
        print_header("✅ RÉSULTATS DE L'ÉVALUATION")
        
        # Informations principales
        print(f"\n🎯 ACTION ANALYSÉE")
        print(f"   Action demandée    : {result.get('action', 'N/A')}")
        print(f"   Action détectée    : {result.get('detected_action', 'N/A')}")
        print(f"   Résultat           : {'✅ CORRECT' if result.get('is_correct') else '❌ INCORRECT'}")
        print(f"   Confiance          : {result.get('percentage', 0):.2f}%")
        
        # Recommandations
        if 'recommendations' in result and result['recommendations']:
            print(f"\n💡 RECOMMANDATIONS POUR AMÉLIORER")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Tous les scores
        if 'all_predictions' in result:
            print(f"\n📊 SCORES POUR TOUTES LES ACTIONS")
            sorted_predictions = sorted(
                result['all_predictions'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )
            
            for action, scores in sorted_predictions:
                percentage = scores['percentage']
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                
                # Emoji selon le rang
                if action == result.get('detected_action'):
                    emoji = "🏆"
                elif percentage > 10:
                    emoji = "📈"
                else:
                    emoji = "  "
                
                print(f"   {emoji} {action:15s} {percentage:6.2f}% {bar}")
        
        # Statistiques supplémentaires
        print(f"\n📈 STATISTIQUES")
        print(f"   Temps total        : {elapsed_time:.2f}s")
        print(f"   Taille vidéo       : {file_size:.2f} MB")
        
        # Sauvegarde du résultat
        output_file = f"result_{ACTION_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"   Résultat sauvegardé: {output_file}")
        
        print_header("✅ TEST TERMINÉ AVEC SUCCÈS")
        return True
        
    except json.JSONDecodeError:
        print("\n❌ ERREUR DE PARSING JSON")
        print("Réponse brute :")
        print(response.text[:500])
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'analyse : {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_test():
    """Mode interactif pour tester plusieurs vidéos"""
    print_header("🎮 MODE INTERACTIF - Test de l'API")
    
    while True:
        print("\n" + "=" * 70)
        video = input("📁 Chemin de la vidéo (ou 'q' pour quitter) : ").strip()
        
        if video.lower() == 'q':
            print("👋 Au revoir !")
            break
        
        if not os.path.exists(video):
            print(f"❌ Fichier non trouvé : {video}")
            continue
        
        print("\n🎯 Actions disponibles :")
        for i, action in enumerate(AVAILABLE_ACTIONS, 1):
            print(f"   {i}. {action}")
        
        action_input = input("\n   Choisissez une action (nom ou numéro) : ").strip()
        
        # Gérer input par numéro
        if action_input.isdigit():
            idx = int(action_input) - 1
            if 0 <= idx < len(AVAILABLE_ACTIONS):
                action = AVAILABLE_ACTIONS[idx]
            else:
                print("❌ Numéro invalide")
                continue
        else:
            action = action_input.lower()
            if action not in AVAILABLE_ACTIONS:
                print(f"❌ Action inconnue : {action}")
                continue
        
        # Mettre à jour les variables globales
        global VIDEO_FILE, ACTION_TYPE
        VIDEO_FILE = video
        ACTION_TYPE = action
        
        # Lancer le test
        test_api()
        
        input("\n⏸️  Appuyez sur ENTRÉE pour continuer...")


if __name__ == "__main__":
    import sys
    
    # Mode interactif si aucun argument
    if len(sys.argv) == 1:
        # Test simple avec les valeurs par défaut
        success = test_api()
        sys.exit(0 if success else 1)
    
    # Mode avec arguments
    elif sys.argv[1] == "--interactive":
        interactive_test()
    
    # Mode avec vidéo et action en arguments
    elif len(sys.argv) >= 3:
        VIDEO_FILE = sys.argv[1]
        ACTION_TYPE = sys.argv[2]
        success = test_api()
        sys.exit(0 if success else 1)
    
    else:
        print("Usage:")
        print("  python test_api.py                    # Test avec config par défaut")
        print("  python test_api.py --interactive      # Mode interactif")
        print("  python test_api.py video.mp4 penalty  # Test avec vidéo et action")