from rest_framework.views import APIView
from rest_framework.response import Response
from .services.evaluate_action_service import EvaluateActionService
import os
import logging

# Configuration d'un logger pour voir les erreurs dans le terminal
logger = logging.getLogger(__name__)

class EvaluateActionView(APIView):
    def post(self, request):
        temp_path = None
        try:
            video = request.FILES.get("video")
            action = request.data.get("action")

            # 1. Vérifications de base
            if not video:
                return Response({"error": "Aucune vidéo envoyée."}, status=400)
            if not action:
                return Response({"error": "Action manquante."}, status=400)

            # 2. Sécurisation du dossier temporaire
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            TEMP_DIR = os.path.join(BASE_DIR, "temp_videos")
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # Utilisation d'un nom de fichier sécurisé
            temp_path = os.path.join(TEMP_DIR, video.name)

            # 3. Écriture du fichier par morceaux (chunks)
            with open(temp_path, "wb+") as f:
                for chunk in video.chunks():
                    f.write(chunk)

            # 4. Appel du service (C'est ici que le calcul peut être long)
            print(f"Début de l'analyse pour l'action : {action}")
            result = EvaluateActionService.evaluate_video(temp_path, action)
            print("Analyse terminée avec succès.")

            return Response(result)

        except Exception as e:
            # On affiche l'erreur réelle dans le terminal pour débugger
            logger.error(f"ERREUR CRITIQUE : {str(e)}", exc_info=True)
            return Response({"error": f"Erreur lors du traitement : {str(e)}"}, status=500)
        
        finally:
            # 5. Nettoyage : On supprime le fichier quoi qu'il arrive (même si ça plante)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Impossible de supprimer le fichier temp : {e}")