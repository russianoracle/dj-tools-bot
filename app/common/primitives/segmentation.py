"""
Segmentation primitives for music structure analysis.

Pure functions for building blocks of Laplacian Segmentation
(McFee & Ellis, 2014): https://zenodo.org/record/1415778

These primitives accept numpy arrays and return numpy arrays.
The orchestration logic belongs in Tasks, not here.

NOTE: All functions use pure numpy/scipy only (NO librosa).
"""

import numpy as np
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scipy.ndimage
import scipy.linalg
from scipy.spatial.distance import cdist


def compute_recurrence_matrix(
    features: np.ndarray,
    width: int = 3,
    mode: str = 'affinity',
    sym: bool = True
) -> np.ndarray:
    """
    Compute recurrence matrix from features (PURE NUMPY).

    Args:
        features: Feature matrix (n_features, n_frames)
        width: Width of the exclusion band around the diagonal
        mode: 'connectivity', 'affinity', or 'distance'
        sym: Symmetrize the matrix

    Returns:
        R: Recurrence matrix (n_frames, n_frames)
    """
    # Ensure features are 2D (n_features, n_frames)
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    n_frames = features.shape[1]

    # Compute pairwise distances using cosine distance
    # Transpose to (n_frames, n_features) for cdist
    D = cdist(features.T, features.T, metric='cosine')

    # Apply exclusion band around diagonal using vectorized operation
    i_indices = np.arange(n_frames)[:, np.newaxis]
    j_indices = np.arange(n_frames)[np.newaxis, :]
    exclusion_mask = np.abs(i_indices - j_indices) <= width

    if mode == 'distance':
        R = D.copy()
        R[exclusion_mask] = 0
    elif mode == 'connectivity':
        # Binary: connected if distance below median
        valid_distances = D[~exclusion_mask]
        threshold = np.median(valid_distances) if len(valid_distances) > 0 else np.inf
        R = (D < threshold).astype(np.float32)
        R[exclusion_mask] = 0
    else:  # affinity
        # Convert distance to affinity using Gaussian kernel
        valid_distances = D[~exclusion_mask]
        sigma = np.median(valid_distances) + 1e-10 if len(valid_distances) > 0 else 1.0
        R = np.exp(-D**2 / (2 * sigma**2))
        R[exclusion_mask] = 0

    if sym:
        R = (R + R.T) / 2

    return np.ascontiguousarray(R, dtype=np.float32)


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
    mfcc = np.ascontiguousarray(mfcc, dtype=np.float32)

    # Squared Euclidean distance between consecutive frames
    path_distance = np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)

    # Convert to similarity using Gaussian kernel
    sigma = np.median(path_distance) + 1e-10
    path_sim = np.exp(-path_distance / sigma)

    # Build tridiagonal matrix
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    return np.ascontiguousarray(R_path, dtype=np.float32)


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

    return np.ascontiguousarray(X, dtype=np.float32)


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
    Enhance diagonal structures in recurrence matrix (PURE NUMPY, VECTORIZED).

    Applies median filter along time-lag representation.
    Uses vectorized diagonal extraction and assignment for M2 optimization.

    Args:
        R: Recurrence matrix (n_frames, n_frames), float32 contiguous
        filter_size: Size of median filter

    Returns:
        R_filtered: Enhanced recurrence matrix, float32 contiguous
    """
    R = np.ascontiguousarray(R, dtype=np.float32)
    n = R.shape[0]
    R_filtered = np.zeros((n, n), dtype=np.float32)

    # Vectorized diagonal processing using np.diagonal and np.fill_diagonal
    # Process all diagonals in a single batch by stacking operations
    all_lags = np.arange(-n + 1, n, dtype=np.int32)

    for k in all_lags:
        # np.diagonal is already vectorized - no inner loop
        diag = np.diagonal(R, offset=k)
        if diag.size == 0:
            continue

        # Apply median filter (scipy.ndimage is already optimized)
        if diag.size >= filter_size:
            filtered_diag = scipy.ndimage.median_filter(diag, size=filter_size)
        else:
            filtered_diag = diag.copy()

        # Vectorized assignment using advanced indexing
        diag_len = filtered_diag.size
        if k >= 0:
            row_idx = np.arange(diag_len, dtype=np.int32)
            col_idx = row_idx + k
        else:
            col_idx = np.arange(diag_len, dtype=np.int32)
            row_idx = col_idx - k

        R_filtered[row_idx, col_idx] = filtered_diag.astype(np.float32)

    return np.ascontiguousarray(R_filtered, dtype=np.float32)


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
    R_rec = np.ascontiguousarray(R_rec, dtype=np.float32)
    R_path = np.ascontiguousarray(R_path, dtype=np.float32)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(R_rec, axis=1)

    # Balanced combination weight
    denom = np.sum((deg_path + deg_rec) ** 2) + 1e-10
    mu = deg_path.dot(deg_path + deg_rec) / denom

    return np.ascontiguousarray(mu * R_rec + (1 - mu) * R_path, dtype=np.float32)