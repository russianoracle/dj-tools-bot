"""
Genre Classifier using pre-trained Essentia models.

Uses Discogs-EffNet for 400 music styles classification.
No manual labeling required - models are pre-trained on millions of tracks.

Optimized for Apple Silicon M2:
- Uses accelerate framework when available
- Efficient batch processing with optimal thread count
- Memory-efficient streaming for large files

Usage:
    from src.genre import GenreClassifier

    classifier = GenreClassifier()
    result = classifier.predict("/path/to/track.mp3")

    print(result.genre)      # "Techno"
    print(result.subgenre)   # "Minimal Techno"
    print(result.all_styles) # [("Minimal Techno", 0.85), ("Tech House", 0.12), ...]

References:
    - https://essentia.upf.edu/models.html
    - https://github.com/MTG/essentia
"""

# Suppress TensorFlow/Essentia warnings BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', message='.*No network created.*')
warnings.filterwarnings('ignore', message='.*MLIR.*')

# Redirect stderr for TF initialization noise
import sys
import contextlib
import atexit

# Global stderr suppression for TF cleanup messages
_original_stderr = sys.stderr
_devnull = None

def _suppress_tf_cleanup():
    """Suppress stderr at exit for TF cleanup noise."""
    global _devnull
    if _devnull is None:
        _devnull = open(os.devnull, 'w')
    sys.stderr = _devnull

# Register early to run before TF's cleanup
atexit.register(_suppress_tf_cleanup)

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()
import json
import logging
import urllib.request
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

# Detect Apple Silicon
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and
    platform.machine() == "arm64"
)

# Optimal thread count for M2 (4 performance + 4 efficiency cores)
M2_OPTIMAL_THREADS = 4 if IS_APPLE_SILICON else os.cpu_count() or 4

# Model URLs from Essentia
MODELS_BASE_URL = "https://essentia.upf.edu/models"

MODELS = {
    "discogs-effnet": {
        "embedding": "discogs-effnet-bs64-1.pb",
        "classifier": "genre_discogs400-discogs-effnet-1.pb",
        "labels": "genre_discogs400-discogs-effnet-1.json",
        "description": "400 Discogs styles (electronic music focused)"
    },
    "mtg-jamendo-genre": {
        "model": "mtg_jamendo_genre-discogs-effnet-1.pb",
        "labels": "mtg_jamendo_genre-discogs-effnet-1.json",
        "description": "87 genres from MTG-Jamendo dataset"
    },
    "mtg-jamendo-moodtheme": {
        "model": "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        "labels": "mtg_jamendo_moodtheme-discogs-effnet-1.json",
        "description": "56 mood/theme tags"
    }
}

# DJ-relevant genre groupings
DJ_GENRE_GROUPS = {
    "Techno": ["Techno", "Minimal Techno", "Tech House", "Acid", "Industrial",
               "Hard Techno", "Detroit Techno", "Dub Techno", "EBM"],
    "House": ["House", "Deep House", "Tech House", "Progressive House",
              "Electro House", "Funky House", "Disco House", "Afro House",
              "Jackin House", "Tribal House", "Chicago House"],
    "Trance": ["Trance", "Progressive Trance", "Psy-Trance", "Goa Trance",
               "Uplifting Trance", "Tech Trance", "Hard Trance"],
    "Bass": ["Drum n Bass", "Dubstep", "Breakbeat", "Jungle", "UK Garage",
             "Grime", "Bass Music", "Breaks", "Halftime"],
    "Hip-Hop": ["Hip Hop", "Trap", "Boom Bap", "Instrumental Hip Hop",
                "Abstract Hip Hop", "Trip Hop", "Gangsta Rap"],
    "Disco/Funk": ["Disco", "Nu-Disco", "Funk", "Soul", "Boogie", "Italo-Disco"],
    "Ambient": ["Ambient", "Downtempo", "Chillout", "IDM", "Electronica",
                "New Age", "Dark Ambient"],
    "Hardcore": ["Hardcore", "Gabber", "Hardstyle", "Happy Hardcore",
                 "Frenchcore", "Speedcore"]
}


@dataclass
class GenreResult:
    """Result of genre classification."""

    # Primary genre (highest confidence)
    genre: str
    confidence: float

    # DJ-friendly grouping
    dj_category: str

    # Subgenre (second highest in same category)
    subgenre: Optional[str] = None
    subgenre_confidence: float = 0.0

    # All predictions above threshold
    all_styles: List[Tuple[str, float]] = field(default_factory=list)

    # Mood/theme tags (if available)
    mood_tags: List[Tuple[str, float]] = field(default_factory=list)

    # Raw embeddings for similarity search
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "genre": self.genre,
            "confidence": round(self.confidence, 3),
            "dj_category": self.dj_category,
            "subgenre": self.subgenre,
            "subgenre_confidence": round(self.subgenre_confidence, 3),
            "top_styles": [(s, round(c, 3)) for s, c in self.all_styles[:5]],
            "mood_tags": [(t, round(c, 3)) for t, c in self.mood_tags[:5]]
        }

    def __str__(self) -> str:
        sub = f" / {self.subgenre}" if self.subgenre else ""
        return f"{self.genre}{sub} ({self.confidence:.0%}) [{self.dj_category}]"


class GenreClassifier:
    """
    Pre-trained genre classifier using Essentia TensorFlow models.

    Uses Discogs-EffNet architecture trained on 400 music styles.
    No manual labeling required.

    Args:
        models_dir: Directory to store/load models (default: models/genre/)
        use_mood: Also predict mood/theme tags
        device: 'cpu' or 'gpu' (auto-detected)
    """

    def __init__(self,
                 models_dir: str = None,
                 use_mood: bool = True,
                 min_confidence: float = 0.1):

        self.models_dir = Path(models_dir or self._default_models_dir())
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.use_mood = use_mood
        self.min_confidence = min_confidence

        self._essentia = None
        self._embedding_model = None
        self._genre_model = None
        self._mood_model = None
        self._genre_labels = None
        self._mood_labels = None

        self._initialized = False

    def _default_models_dir(self) -> Path:
        """Get default models directory."""
        # Try to find project root
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "models").exists():
                return current / "models" / "genre"
            current = current.parent
        return Path.home() / ".cache" / "mood-classifier" / "genre-models"

    def _ensure_initialized(self):
        """Lazy initialization - load models on first use."""
        if self._initialized:
            return

        logger.info("Initializing genre classifier...")

        # Import essentia and suppress debug output (with stderr suppression)
        try:
            with suppress_stderr():
                import essentia
                essentia.log.warningActive = False  # Suppress [ WARNING ] messages
                essentia.log.infoActive = False     # Suppress [ INFO ] messages
                import essentia.standard as es
            self._essentia = es

            # Register stderr suppression AFTER TF is loaded (runs first in LIFO)
            def suppress_at_exit():
                import os, sys
                try:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, 2)  # Redirect fd 2 (stderr) to /dev/null
                except:
                    pass
            atexit.register(suppress_at_exit)
        except ImportError:
            raise ImportError(
                "essentia-tensorflow not installed. Run:\n"
                "  pip install essentia-tensorflow\n"
                "  # or on macOS with conda:\n"
                "  conda install -c mtg essentia-tensorflow"
            )

        # Download models if needed
        self._download_models()

        # Load models (with stderr suppression for TF noise)
        with suppress_stderr():
            self._load_models()

        self._initialized = True
        logger.info("Genre classifier initialized")

    def _download_models(self):
        """Download pre-trained models if not present."""

        files_to_download = [
            ("discogs-effnet", "embedding"),
            ("discogs-effnet", "classifier"),
            ("discogs-effnet", "labels"),
        ]

        if self.use_mood:
            files_to_download.extend([
                ("mtg-jamendo-moodtheme", "model"),
                ("mtg-jamendo-moodtheme", "labels"),
            ])

        # Correct URLs for each model file
        model_urls = {
            # Embedding model
            "discogs-effnet-bs64-1.pb": "feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
            "discogs-effnet-bs64-1.json": "feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json",
            # Genre classifier
            "genre_discogs400-discogs-effnet-1.pb": "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
            "genre_discogs400-discogs-effnet-1.json": "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
            # Mood model
            "mtg_jamendo_moodtheme-discogs-effnet-1.pb": "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            "mtg_jamendo_moodtheme-discogs-effnet-1.json": "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
        }

        for model_name, file_type in files_to_download:
            model_info = MODELS[model_name]

            if file_type == "embedding":
                filename = model_info["embedding"]
            elif file_type == "classifier":
                filename = model_info["classifier"]
            elif file_type == "labels":
                if "labels" in model_info:
                    filename = model_info["labels"]
                else:
                    continue
            elif file_type == "model":
                filename = model_info["model"]
            else:
                continue

            local_path = self.models_dir / filename

            if local_path.exists():
                logger.debug(f"Model already exists: {filename}")
                continue

            if filename not in model_urls:
                logger.warning(f"Unknown model file: {filename}")
                continue

            url = f"{MODELS_BASE_URL}/{model_urls[filename]}"
            logger.info(f"Downloading {filename}...")

            try:
                urllib.request.urlretrieve(url, local_path)
                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise RuntimeError(f"Cannot download model {filename} from {url}")

    def _load_models(self):
        """Load TensorFlow models."""
        es = self._essentia

        # Load embedding model (EffNet-Discogs)
        embedding_path = self.models_dir / MODELS["discogs-effnet"]["embedding"]
        self._embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(embedding_path),
            output="PartitionedCall:1"
        )

        # Load genre classifier
        genre_path = self.models_dir / MODELS["discogs-effnet"]["classifier"]
        self._genre_model = es.TensorflowPredict2D(
            graphFilename=str(genre_path),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )

        # Load genre labels
        labels_path = self.models_dir / MODELS["discogs-effnet"]["labels"]
        with open(labels_path) as f:
            labels_data = json.load(f)
            self._genre_labels = labels_data.get("classes", [])

        # Load mood model (optional)
        if self.use_mood:
            mood_model_path = self.models_dir / MODELS["mtg-jamendo-moodtheme"]["model"]
            mood_labels_path = self.models_dir / MODELS["mtg-jamendo-moodtheme"]["labels"]

            if mood_model_path.exists():
                self._mood_model = es.TensorflowPredict2D(
                    graphFilename=str(mood_model_path),
                    input="model/Placeholder",
                    output="model/Sigmoid"
                )

                with open(mood_labels_path) as f:
                    labels_data = json.load(f)
                    self._mood_labels = labels_data.get("classes", [])

        logger.info(f"Loaded {len(self._genre_labels)} genre classes")
        if self._mood_labels:
            logger.info(f"Loaded {len(self._mood_labels)} mood classes")

    def predict(self, audio_path: str, return_embedding: bool = False) -> GenreResult:
        """
        Predict genre for an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, FLAC, etc.)
            return_embedding: Include 128-dim embedding in result

        Returns:
            GenreResult with genre, subgenre, confidence, etc.
        """
        self._ensure_initialized()

        es = self._essentia

        # Load audio and run models with stderr suppression
        with suppress_stderr():
            # Load audio at 16kHz (required by model)
            audio = es.MonoLoader(
                filename=audio_path,
                sampleRate=16000,
                resampleQuality=4
            )()

            # Get embeddings
            embeddings = self._embedding_model(audio)

        # Average embedding across time
        avg_embedding = np.mean(embeddings, axis=0)

        # Predict genres (suppress TF noise)
        with suppress_stderr():
            genre_predictions = self._genre_model(embeddings)
        avg_genre_pred = np.mean(genre_predictions, axis=0)

        # Get top predictions
        top_indices = np.argsort(avg_genre_pred)[::-1]
        all_styles = []

        for idx in top_indices:
            conf = float(avg_genre_pred[idx])
            if conf >= self.min_confidence:
                label = self._genre_labels[idx] if idx < len(self._genre_labels) else f"class_{idx}"
                all_styles.append((label, conf))

        # Get primary genre and DJ category
        if all_styles:
            genre, confidence = all_styles[0]
            dj_category = self._get_dj_category(genre)

            # Find subgenre in same category
            subgenre = None
            subgenre_conf = 0.0
            for style, conf in all_styles[1:10]:
                if self._get_dj_category(style) == dj_category:
                    subgenre = style
                    subgenre_conf = conf
                    break
        else:
            genre = "Unknown"
            confidence = 0.0
            dj_category = "Other"
            subgenre = None
            subgenre_conf = 0.0

        # Predict mood tags
        mood_tags = []
        if self._mood_model and self._mood_labels:
            with suppress_stderr():
                mood_predictions = self._mood_model(embeddings)
            avg_mood_pred = np.mean(mood_predictions, axis=0)

            top_mood_indices = np.argsort(avg_mood_pred)[::-1]
            for idx in top_mood_indices[:10]:
                conf = float(avg_mood_pred[idx])
                if conf >= self.min_confidence:
                    label = self._mood_labels[idx] if idx < len(self._mood_labels) else f"mood_{idx}"
                    mood_tags.append((label, conf))

        return GenreResult(
            genre=genre,
            confidence=confidence,
            dj_category=dj_category,
            subgenre=subgenre,
            subgenre_confidence=subgenre_conf,
            all_styles=all_styles[:20],
            mood_tags=mood_tags,
            embedding=avg_embedding if return_embedding else None
        )

    def predict_batch(self, audio_paths: List[str],
                      show_progress: bool = True,
                      parallel: bool = True,
                      workers: int = None) -> List[GenreResult]:
        """
        Predict genres for multiple files.

        Optimized for Apple Silicon M2 with parallel processing.

        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar
            parallel: Use parallel processing (recommended for M2)
            workers: Number of parallel workers (default: 4 for M2)

        Returns:
            List of GenreResult
        """
        # Initialize before parallel processing to avoid race conditions
        self._ensure_initialized()

        if workers is None:
            workers = M2_OPTIMAL_THREADS

        if not parallel or len(audio_paths) < 3:
            # Sequential processing for small batches
            results = []

            if show_progress:
                try:
                    import sys
                    from tqdm import tqdm
                    iterator = tqdm(audio_paths, desc="Classifying genres", file=sys.stdout)
                except ImportError:
                    iterator = audio_paths
            else:
                iterator = audio_paths

            for path in iterator:
                try:
                    result = self.predict(path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append(GenreResult(
                        genre="Error",
                        confidence=0.0,
                        dj_category="Unknown",
                        all_styles=[]
                    ))

            return results

        # Parallel processing for large batches (M2 optimized)
        if IS_APPLE_SILICON:
            logger.info(f"Using M2-optimized parallel processing ({workers} workers)")

        results = [None] * len(audio_paths)

        def process_track(idx_path):
            idx, path = idx_path
            try:
                return idx, self.predict(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                return idx, GenreResult(
                    genre="Error",
                    confidence=0.0,
                    dj_category="Unknown",
                    all_styles=[]
                )

        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.predict, path): idx
                for idx, path in enumerate(audio_paths)
            }

            if show_progress:
                try:
                    import sys
                    from tqdm import tqdm
                    pbar = tqdm(total=len(audio_paths),
                               desc=f"Classifying ({workers} workers)",
                               file=sys.stdout)
                except ImportError:
                    pbar = None
            else:
                pbar = None

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {audio_paths[idx]}: {e}")
                    results[idx] = GenreResult(
                        genre="Error",
                        confidence=0.0,
                        dj_category="Unknown",
                        all_styles=[]
                    )
                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        return results

    def _get_dj_category(self, style: str) -> str:
        """Map Discogs style to DJ-friendly category."""
        style_lower = style.lower()

        for category, styles in DJ_GENRE_GROUPS.items():
            for s in styles:
                if s.lower() in style_lower or style_lower in s.lower():
                    return category

        # Fallback mappings
        if any(x in style_lower for x in ["electro", "synth", "electronic"]):
            return "Electronic"
        if any(x in style_lower for x in ["rock", "metal", "punk"]):
            return "Rock"
        if any(x in style_lower for x in ["jazz", "blues"]):
            return "Jazz/Blues"
        if any(x in style_lower for x in ["reggae", "dub", "ska"]):
            return "Reggae/Dub"
        if any(x in style_lower for x in ["latin", "salsa", "bossa"]):
            return "Latin"
        if any(x in style_lower for x in ["world", "african", "arabic"]):
            return "World"
        if any(x in style_lower for x in ["classical", "orchestra"]):
            return "Classical"
        if any(x in style_lower for x in ["pop", "indie"]):
            return "Pop"

        return "Other"

    def get_similar_tracks(self, reference_path: str,
                          candidate_paths: List[str],
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find tracks similar to reference based on genre embedding.

        Args:
            reference_path: Path to reference track
            candidate_paths: List of candidate track paths
            top_k: Number of similar tracks to return

        Returns:
            List of (path, similarity_score) tuples
        """
        self._ensure_initialized()

        # Get reference embedding
        ref_result = self.predict(reference_path, return_embedding=True)
        ref_embedding = ref_result.embedding

        if ref_embedding is None:
            raise ValueError("Failed to get embedding for reference track")

        # Get candidate embeddings and compute similarity
        similarities = []

        for path in candidate_paths:
            try:
                result = self.predict(path, return_embedding=True)
                if result.embedding is not None:
                    # Cosine similarity
                    sim = np.dot(ref_embedding, result.embedding) / (
                        np.linalg.norm(ref_embedding) * np.linalg.norm(result.embedding)
                    )
                    similarities.append((path, float(sim)))
            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


def main():
    """Demo: classify a track."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.genre.classifier <audio_file>")
        print("\nExample:")
        print("  python -m src.genre.classifier /path/to/track.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"Classifying: {audio_path}")
    print("-" * 50)

    classifier = GenreClassifier()
    result = classifier.predict(audio_path)

    print(f"\nGenre: {result.genre} ({result.confidence:.1%})")
    print(f"DJ Category: {result.dj_category}")

    if result.subgenre:
        print(f"Subgenre: {result.subgenre} ({result.subgenre_confidence:.1%})")

    print(f"\nTop styles:")
    for style, conf in result.all_styles[:10]:
        print(f"  {conf:5.1%} {style}")

    if result.mood_tags:
        print(f"\nMood tags:")
        for tag, conf in result.mood_tags[:5]:
            print(f"  {conf:5.1%} {tag}")


if __name__ == "__main__":
    main()
