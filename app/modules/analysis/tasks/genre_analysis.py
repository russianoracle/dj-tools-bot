"""
Genre Analysis Task - Classify track genre using Essentia models.

Uses Discogs-EffNet pre-trained model for 400 music styles.
Fully self-contained - no external dependencies except Essentia.
"""

import os
import json
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from app.common.logging import get_logger
from .base import AudioContext, TaskResult, BaseTask

logger = get_logger(__name__)

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Model URLs
MODELS_BASE_URL = "https://essentia.upf.edu/models"
MODEL_PATHS = {
    "embedding": "feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre": "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_labels": "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "mood": "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "mood_labels": "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
}

# DJ-friendly genre groups
DJ_GENRE_GROUPS = {
    "Techno": ["Techno", "Minimal Techno", "Tech House", "Acid", "Industrial", "Hard Techno", "Detroit Techno", "Dub Techno"],
    "House": ["House", "Deep House", "Tech House", "Progressive House", "Electro House", "Funky House", "Disco House"],
    "Trance": ["Trance", "Progressive Trance", "Psy-Trance", "Goa Trance", "Uplifting Trance"],
    "Bass": ["Drum n Bass", "Dubstep", "Breakbeat", "Jungle", "UK Garage", "Grime"],
    "Hip-Hop": ["Hip Hop", "Trap", "Boom Bap", "Trip Hop"],
    "Disco/Funk": ["Disco", "Nu-Disco", "Funk", "Soul", "Boogie"],
    "Ambient": ["Ambient", "Downtempo", "Chillout", "IDM", "Electronica"],
}


@dataclass
class GenreAnalysisResult(TaskResult):
    """Result of genre classification."""
    genre: str = ''
    subgenre: str = ''
    dj_category: str = ''
    confidence: float = 0.0
    all_styles: List[Tuple[str, float]] = field(default_factory=list)
    mood_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'genre': self.genre,
            'subgenre': self.subgenre,
            'dj_category': self.dj_category,
            'confidence': self.confidence,
            'all_styles': self.all_styles,
            'mood_tags': self.mood_tags,
        })
        return base


class _EssentiaModels:
    """Singleton for lazy-loaded Essentia models."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._es = None
        self._embedding_model = None
        self._genre_model = None
        self._mood_model = None
        self._genre_labels = []
        self._mood_labels = []
        self._models_dir = None

    def _get_models_dir(self) -> Path:
        if self._models_dir:
            return self._models_dir
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "models").exists():
                self._models_dir = current / "models" / "genre"
                break
            current = current.parent
        if not self._models_dir:
            self._models_dir = Path.home() / ".cache" / "mood-classifier" / "genre-models"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        return self._models_dir

    def _download_models(self):
        """Download models if not present."""
        models_dir = self._get_models_dir()
        files = [
            ("discogs-effnet-bs64-1.pb", MODEL_PATHS["embedding"]),
            ("genre_discogs400-discogs-effnet-1.pb", MODEL_PATHS["genre"]),
            ("genre_discogs400-discogs-effnet-1.json", MODEL_PATHS["genre_labels"]),
            ("mtg_jamendo_moodtheme-discogs-effnet-1.pb", MODEL_PATHS["mood"]),
            ("mtg_jamendo_moodtheme-discogs-effnet-1.json", MODEL_PATHS["mood_labels"]),
        ]
        for filename, url_path in files:
            local_path = models_dir / filename
            if local_path.exists():
                continue
            url = f"{MODELS_BASE_URL}/{url_path}"
            logger.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, local_path)

    def ensure_loaded(self):
        """Lazy load models on first use."""
        if self._embedding_model is not None:
            return

        logger.info("Loading Essentia genre models...")

        try:
            import essentia
            essentia.log.warningActive = False
            essentia.log.infoActive = False
            import essentia.standard as es
            self._es = es
        except ImportError:
            raise ImportError("essentia-tensorflow not installed. Run: pip install essentia-tensorflow")

        self._download_models()
        models_dir = self._get_models_dir()

        # Load embedding model
        self._embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(models_dir / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1"
        )

        # Load genre classifier
        self._genre_model = es.TensorflowPredict2D(
            graphFilename=str(models_dir / "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )

        # Load genre labels
        with open(models_dir / "genre_discogs400-discogs-effnet-1.json") as f:
            self._genre_labels = json.load(f).get("classes", [])

        # Load mood model
        mood_path = models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
        if mood_path.exists():
            self._mood_model = es.TensorflowPredict2D(
                graphFilename=str(mood_path),
                input="model/Placeholder",
                output="model/Sigmoid"
            )
            with open(models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json") as f:
                self._mood_labels = json.load(f).get("classes", [])

        logger.info(f"Loaded {len(self._genre_labels)} genres, {len(self._mood_labels)} moods")
        self._initialized = True

    def predict_from_array(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Predict from audio array. Returns (embedding, genre_probs, mood_probs)."""
        self.ensure_loaded()

        # Resample to 16kHz
        if sr != 16000:
            audio = self._es.Resample(inputSampleRate=sr, outputSampleRate=16000)(y.astype(np.float32))
        else:
            audio = y.astype(np.float32)

        embeddings = self._embedding_model(audio)
        genre_preds = self._genre_model(embeddings)

        mood_probs = None
        if self._mood_model:
            mood_preds = self._mood_model(embeddings)
            mood_probs = np.mean(mood_preds, axis=0)

        return np.mean(embeddings, axis=0), np.mean(genre_preds, axis=0), mood_probs

    def predict_from_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Predict from file. Returns (embedding, genre_probs, mood_probs)."""
        self.ensure_loaded()

        audio = self._es.MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self._embedding_model(audio)
        genre_preds = self._genre_model(embeddings)

        mood_probs = None
        if self._mood_model:
            mood_preds = self._mood_model(embeddings)
            mood_probs = np.mean(mood_preds, axis=0)

        return np.mean(embeddings, axis=0), np.mean(genre_preds, axis=0), mood_probs


# Singleton instance
_models = _EssentiaModels()


def _get_dj_category(style: str) -> str:
    """Map Discogs style to DJ category."""
    style_lower = style.lower()
    for category, styles in DJ_GENRE_GROUPS.items():
        for s in styles:
            if s.lower() in style_lower or style_lower in s.lower():
                return category
    if any(x in style_lower for x in ["electro", "synth", "electronic"]):
        return "Electronic"
    return "Other"


class GenreAnalysisTask(BaseTask):
    """
    Classify track genre using Essentia Discogs-EffNet model.

    Fully self-contained - loads Essentia models directly.
    Works with both file paths and numpy arrays.

    Usage:
        task = GenreAnalysisTask(top_n=5)
        result = task.execute(context)
        print(f"Genre: {result.genre} ({result.confidence:.0%})")
    """

    def __init__(self, top_n: int = 5, min_confidence: float = 0.1):
        self.top_n = top_n
        self.min_confidence = min_confidence

    def execute(self, context: AudioContext) -> GenreAnalysisResult:
        """Classify the track genre."""
        start_time = time.time()

        try:
            # Predict
            if context.file_path:
                _, genre_probs, mood_probs = _models.predict_from_file(context.file_path)
            else:
                _, genre_probs, mood_probs = _models.predict_from_array(context.y, context.sr)

            # Parse predictions
            labels = _models._genre_labels
            top_indices = np.argsort(genre_probs)[::-1]

            all_styles = []
            for idx in top_indices:
                conf = float(genre_probs[idx])
                if conf >= self.min_confidence and idx < len(labels):
                    all_styles.append((labels[idx], conf))

            # Primary genre
            if all_styles:
                genre, confidence = all_styles[0]
                dj_category = _get_dj_category(genre)
                subgenre = None
                for style, _ in all_styles[1:10]:
                    if _get_dj_category(style) == dj_category:
                        subgenre = style
                        break
            else:
                genre, confidence = "Unknown", 0.0
                dj_category = "Other"
                subgenre = None

            # Mood tags
            mood_tags = []
            if mood_probs is not None:
                mood_labels = _models._mood_labels
                top_mood_indices = np.argsort(mood_probs)[::-1]
                for idx in top_mood_indices[:10]:
                    conf = float(mood_probs[idx])
                    if conf >= self.min_confidence and idx < len(mood_labels):
                        mood_tags.append((mood_labels[idx], conf))

            return GenreAnalysisResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                genre=genre,
                subgenre=subgenre or '',
                dj_category=dj_category,
                confidence=confidence,
                all_styles=all_styles[:self.top_n],
                mood_tags=mood_tags
            )

        except ImportError as e:
            return GenreAnalysisResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=f"Essentia not available: {e}"
            )
        except Exception as e:
            return GenreAnalysisResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

    @property
    def requires_essentia(self) -> bool:
        return True
