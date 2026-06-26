from functools import lru_cache

import numpy as np
import pandas as pd

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_embedder(model_name: str = DEFAULT_MODEL_NAME):
    """
    Return a cached SentenceTransformer for the given model name.

    Parameters:
    - model_name: str, the SentenceTransformer model to load

    Returns:
    - SentenceTransformer instance
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Return the output dimension of the embedding model."""
    return get_embedder(model_name).get_sentence_embedding_dimension()


def embed_texts(texts, model_name: str = DEFAULT_MODEL_NAME) -> list:
    """
    Encode a sequence of texts into a list of embedding vectors (lists of floats).

    Parameters:
    - texts: iterable of str, the texts to encode
    - model_name: str, the SentenceTransformer model to use

    Returns:
    - list of list[float], one embedding per input text
    """
    model = get_embedder(model_name)
    return model.encode(list(texts), show_progress_bar=False).tolist()


def flatten_embedding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand list/array-valued columns into per-dimension scalar columns.

    Column order is preserved and each embedding column ``col`` becomes
    ``col_0 ... col_{d-1}``, so tabular models such as XGBoost can consume it.
    Scalar columns are passed through unchanged.

    Parameters:
    - df: pandas DataFrame, may contain embedding (array) columns

    Returns:
    - pandas DataFrame with embedding columns flattened into scalars
    """
    pieces = []
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            mat = np.vstack(df[col].apply(lambda v: np.asarray(v, dtype=np.float64)).values)
            pieces.append(pd.DataFrame(
                mat,
                columns=[f"{col}_{i}" for i in range(mat.shape[1])],
                index=df.index,
            ))
        else:
            pieces.append(df[[col]])
    return pd.concat(pieces, axis=1)
