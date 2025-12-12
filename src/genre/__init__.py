"""
Genre classification module using pre-trained Essentia models.

Supports:
- Discogs 400 styles (electronic music focused)
- MTG-Jamendo genre/mood/instrument
- MusicNN activations
"""

from .classifier import GenreClassifier, GenreResult

__all__ = ['GenreClassifier', 'GenreResult']
