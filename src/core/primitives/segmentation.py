"""
Segmentation primitives for music structure analysis.

Pure functions for building blocks of Laplacian Segmentation
(McFee & Ellis, 2014): https://zenodo.org/record/1415778

These primitives accept numpy arrays and return numpy arrays.
The orchestration logic belongs in Tasks, not here.
"""

import numpy as np
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scipy.ndimage
import librosa


def compute_recurrence_matrix(
    features: np.ndarray,
    width: int = 3,
    mode: str = 'affinity',
    sym: bool = True
) -> np.ndarray:
    """
    Compute recurrence matrix from features.

    Pure wrapper around librosa.segment.recurrence_matrix.

    Args:
        features: Feature matrix (n_features, n_frames)
        width: Width of the exclusion band around the diagonal
        mode: 'connectivity', 'affinity', or 'distance'
        sym: Symmetrize the matrix

    Returns:
        R: Recurrence matrix (n_frames, n_frames)
    """
    return librosa.segment.recurrence_matrix(
        features, width=width, mode=mode, sym=sym
    )


def compute_path_similarity(mfcc: np.ndarray) -> np.ndarray:
    """
    Compute path similarity matrix from MFCC sequence.

    Measures local (frame-to-frame) similarity for sequential structure.
    Pure function: array in -> array out.

    Args:
        mfcc: MFCC features (n_mfcc, n_frames)

    Returns:
        R_path: Tridiagonal path similarity matrix (n_frames, n_frames)
    """
    # Squared Euclidean distance between consecutive frames
    path_distance = np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)

    # Convert to similarity using Gaussian kernel
    sigma = np.median(path_distance) + 1e-10
    path_sim = np.exp(-path_distance / sigma)

    # Build tridiagonal matrix
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    return R_path


def compute_laplacian_eigenvectors(
    A: np.ndarray,
    k: int = 10,
    median_filter_size: int = 9,
    use_sparse_solver: bool = True,
) -> np.ndarray:
    """
    Compute smoothed eigenvectors of normalized Laplacian.

    Pure function for spectral decomposition.
    Uses sparse eigensolver for large matrices (10x faster).

    Args:
        A: Affinity matrix (n_frames, n_frames)
        k: Number of eigenvectors to return
        median_filter_size: Size of median filter for smoothing
        use_sparse_solver: Use sparse eigsh (much faster for k << n)

    Returns:
        X: Normalized eigenvector matrix (n_frames, k)
    """
    n = A.shape[0]
    k = min(k, n - 2)  # eigsh requires k < n

    # Normalized Laplacian (returns sparse if possible)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # Eigendecomposition - use sparse solver for large matrices
    if use_sparse_solver and n > 100:
        # Sparse solver: only compute k+1 smallest eigenvalues
        # Much faster: O(n²k) vs O(n³)
        try:
            from scipy.sparse import csr_matrix
            L_sparse = csr_matrix(L) if not scipy.sparse.issparse(L) else L
            evals, evecs = scipy.sparse.linalg.eigsh(
                L_sparse,
                k=k + 1,
                which='SM',  # Smallest magnitude
                tol=1e-4,
                maxiter=300,
            )
            # Sort by eigenvalue (eigsh doesn't guarantee order)
            idx = np.argsort(evals)
            evecs = evecs[:, idx]
        except Exception:
            # Fallback to dense solver
            evals, evecs = scipy.linalg.eigh(L if isinstance(L, np.ndarray) else L.toarray())
    else:
        # Dense solver for small matrices
        L_dense = L if isinstance(L, np.ndarray) else L.toarray()
        evals, evecs = scipy.linalg.eigh(L_dense)

    # Smooth eigenvectors
    evecs = scipy.ndimage.median_filter(evecs, size=(median_filter_size, 1))

    # Cumulative normalization
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5 + 1e-10

    # Extract first k eigenvectors, normalized
    k = min(k, evecs.shape[1])
    X = evecs[:, :k] / Cnorm[:, k - 1:k]

    return X


def detect_boundaries_from_labels(segment_ids: np.ndarray) -> np.ndarray:
    """
    Detect segment boundaries from label sequence.

    Pure function: finds indices where labels change.

    Args:
        segment_ids: Cluster label for each frame (n_frames,)

    Returns:
        bound_indices: Indices of boundary frames (includes 0)
    """
    if len(segment_ids) <= 1:
        return np.array([0])

    # Find where labels change
    changes = np.flatnonzero(segment_ids[:-1] != segment_ids[1:])
    bound_indices = np.concatenate([[0], changes + 1])

    return bound_indices


def enhance_recurrence_diagonals(
    R: np.ndarray,
    filter_size: int = 7
) -> np.ndarray:
    """
    Enhance diagonal structures in recurrence matrix.

    Applies median filter along time-lag representation.

    Args:
        R: Recurrence matrix (n_frames, n_frames)
        filter_size: Size of median filter

    Returns:
        R_filtered: Enhanced recurrence matrix
    """
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    return df(R, size=(1, filter_size))


def combine_recurrence_and_path(
    R_rec: np.ndarray,
    R_path: np.ndarray
) -> np.ndarray:
    """
    Combine recurrence and path similarity matrices.

    Uses balanced weighting from McFee & Ellis (2014).

    Args:
        R_rec: Recurrence matrix (self-similarity)
        R_path: Path similarity matrix (sequential)

    Returns:
        A: Combined affinity matrix
    """
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(R_rec, axis=1)

    # Balanced combination weight
    denom = np.sum((deg_path + deg_rec) ** 2) + 1e-10
    mu = deg_path.dot(deg_path + deg_rec) / denom

    return mu * R_rec + (1 - mu) * R_path
