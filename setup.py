"""Setup configuration for Mood Classifier."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="mood-classifier",
    version="0.1.0",
    author="Mood Classifier Team",
    description="DJ Track Energy Zone Classification System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mood-classifier",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "audioread>=3.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "mutagen>=1.47.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "performance": [
            "numba>=0.57.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mood-classifier=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="audio music dj classification machine-learning librosa",
)
