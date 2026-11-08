import numpy as np


def normalized(embedding):
    x = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(x)
    if not np.isfinite(x).all() or norm < 1e-8:
        raise ValueError("Embedding nul ou non fini")
    return x / norm


def cosine(a, b) -> float:
    return float(np.clip(np.dot(normalized(a), normalized(b)), -1, 1))
