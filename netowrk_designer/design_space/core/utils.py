import itertools
import numpy as np

import hashlib


def parent_combinations(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]

def graph_hash_np(adjacency_matrix, ops):
    def hash_module(matrix, labelling):
        """Computes a graph-invariance MD5 hash of the matrix and label pair.
        Args:
            matrix: np.ndarray square upper-triangular adjacency matrix.
            labelling: list of int labels of length equal to both dimensions of
                matrix.
        Returns:
            MD5 hash of the matrix and labelling.
        """
        vertices = np.shape(matrix)[0]
        in_edges = np.sum(matrix, axis=0).tolist()
        out_edges = np.sum(matrix, axis=1).tolist()
        assert len(in_edges) == len(out_edges) == len(labelling), f'{labelling} {matrix}'
        hashes = list(zip(out_edges, in_edges, labelling))
        hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
        # Computing this up to the diameter is probably sufficient but since the
        # operation is fast, it is okay to repeat more times.
        for _ in range(vertices):
            new_hashes = []
            for v in range(vertices):
                in_neighbours = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbours = [hashes[w] for w in range(vertices) if matrix[v, w]]
                new_hashes.append(hashlib.md5(
                        (''.join(sorted(in_neighbours)) + '|' +
                        ''.join(sorted(out_neighbours)) + '|' +
                        hashes[v]).encode('utf-8')).hexdigest())
            hashes = new_hashes
        fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
        return fingerprint

    return hash_module(adjacency_matrix, ops)

def remove_zero_rows_cols(array):
    n = array.shape[0]  # Get the size of the square array
    indices = np.arange(n)
    zero_indices = indices[np.all(array == 0, axis=0) & np.all(array == 0, axis=1)]

    # Remove the zero rows and columns
    trimmed_array = np.delete(array, zero_indices, axis=0)
    trimmed_array = np.delete(trimmed_array, zero_indices, axis=1)

    return trimmed_array
