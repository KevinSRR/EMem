import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal, TypeVar, Generic, Callable
import logging
from copy import deepcopy
import pandas as pd
import pickle
import json
from dataclasses import asdict, is_dataclass

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

# Type variable for generic content types
T = TypeVar('T')

class ContentStore(Generic[T]):
    def __init__(
        self, 
        db_filename: str, 
        namespace: str,
        batch_size: int = 32,
        embedding_model=None, 
        text_extraction_fn: Optional[Callable[[T], str]] = None,
        enable_embeddings: bool = True
    ):
        """
        Initializes the ContentStore with necessary configurations and sets up the working directory.

        Note: self.hash_id_to_text and self.text_to_hash_id are not maintained any more for string types. 

        Parameters:
        db_filename: The directory path where data will be stored or retrieved.
        namespace: A unique identifier for data segregation.
        batch_size: The batch size used for processing embeddings.
        embedding_model: The model used for embeddings (required if enable_embeddings=True).
        text_extraction_fn: Function to convert content objects to strings for embedding (required if enable_embeddings=True).
        enable_embeddings: Whether to compute and store embeddings for the content.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs filenames for storing data.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.batch_size = batch_size
        self.namespace = namespace
        self.enable_embeddings = enable_embeddings
        
        # Validate embedding requirements
        if self.enable_embeddings:
            if embedding_model is None:
                raise ValueError("embedding_model is required when enable_embeddings=True")
            if text_extraction_fn is None:
                raise ValueError("text_extraction_fn is required when enable_embeddings=True")
        
        self.embedding_model = embedding_model
        self.text_extraction_fn = text_extraction_fn

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        # Separate files for content and embeddings
        self.content_filename = os.path.join(db_filename, f"content_{self.namespace}.pkl")
        if self.enable_embeddings:
            self.embedding_filename = os.path.join(db_filename, f"embeddings_{self.namespace}.parquet")
            
        self._load_data()

    def _compute_content_hash(self, content: T) -> str:
        """Compute a hash ID for content object."""
        if isinstance(content, str):
            content_str = content
        else:
            # For structured objects, use JSON serialization for hashing
            if is_dataclass(content):
                content_dict = asdict(content)
            else:
                content_dict = content.__dict__ if hasattr(content, '__dict__') else str(content)
            content_str = json.dumps(content_dict, sort_keys=True, default=str) # for dataclass, the field will be in order when instantiated
        
        return compute_mdhash_id(content_str, prefix=self.namespace + "-")

    def get_missing_content_hash_ids(self, contents: List[T]) -> Dict[str, Dict[str, Any]]:
        """Get hash IDs for content that doesn't exist in the store."""
        content_dict = {}

        for content in contents:
            hash_id = self._compute_content_hash(content)
            content_dict[hash_id] = {'content': content}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(content_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        contents_to_add = [content_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": c} for h, c in zip(missing_ids, contents_to_add)}

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Backward compatibility method for string content."""
        return self.get_missing_content_hash_ids(texts)

    def insert_content(self, contents: List[T], encoding_instruction: Optional[str] = None) -> List[str]:
        """Insert content objects into the store."""
        content_dict = {}

        for content in contents:
            hash_id = self._compute_content_hash(content)
            content_dict[hash_id] = {'content': content}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(content_dict.keys())
        if not all_hash_ids:
            return []

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return []

        # Prepare the contents to store
        contents_to_store = [content_dict[hash_id]["content"] for hash_id in missing_ids]

        # Compute embeddings if enabled
        missing_embeddings = None
        if self.enable_embeddings:
            texts_to_encode = [self.text_extraction_fn(content) for content in contents_to_store]
            
            if encoding_instruction is None:
                logger.info(f"Encoding {len(texts_to_encode)} contents without instruction prefix")
                missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
            else:
                missing_embeddings = self.embedding_model.batch_encode(texts_to_encode, instruction=encoding_instruction)

        self._upsert(missing_ids, contents_to_store, missing_embeddings)

        return missing_ids

    def insert_strings(self, texts: List[str], encoding_instruction: Optional[str] = None) -> List[str]:
        """Backward compatibility method for string content."""
        return self.insert_content(texts, encoding_instruction)
        
    def _load_data(self):
        """Load content and embeddings from storage."""
        # Load content data
        if os.path.exists(self.content_filename):
            with open(self.content_filename, 'rb') as f:
                content_data = pickle.load(f)
                self.hash_ids = content_data['hash_ids']
                self.contents = content_data['contents']
                logger.info(f"Loaded {len(self.hash_ids)} content records from {self.content_filename}")
        else:
            self.hash_ids, self.contents = [], []

        # Load embedding data if enabled
        if self.enable_embeddings and os.path.exists(self.embedding_filename):
            df = pd.read_parquet(self.embedding_filename)
            embedding_hash_ids = df["hash_id"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            # Ensure embedding order matches content order
            if set(embedding_hash_ids) != set(self.hash_ids):
                logger.warning("Embedding hash IDs don't match content hash IDs, rebuilding embeddings...")
                self.embeddings = []
            else:
                # Reorder embeddings to match content order
                embedding_dict = {h: e for h, e in zip(embedding_hash_ids, self.embeddings)}
                self.embeddings = [embedding_dict[h] for h in self.hash_ids]
                logger.info(f"Loaded {len(self.embeddings)} embedding records from {self.embedding_filename}")
        else:
            self.embeddings = []

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for efficient access."""
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": c}
            for h, c in zip(self.hash_ids, self.contents)
        }
        
        # Backward compatibility indices for string content
        if self.contents and isinstance(self.contents[0], str):
            self.hash_id_to_text = {h: self.contents[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.contents[idx]: h for idx, h in enumerate(self.hash_ids)}
        else:
            self.hash_id_to_text = None
            self.text_to_hash_id = None

    def _save_data(self):
        """Save content and embeddings to storage."""
        # Save content data
        content_data = {
            'hash_ids': self.hash_ids,
            'contents': self.contents
        }
        with open(self.content_filename, 'wb') as f:
            pickle.dump(content_data, f)
        logger.info(f"Saved {len(self.hash_ids)} content records to {self.content_filename}")
        
        # Save embedding data if enabled
        if self.enable_embeddings and self.embeddings:
            embedding_data = pd.DataFrame({
                "hash_id": self.hash_ids,
                "embedding": self.embeddings
            })
            embedding_data.to_parquet(self.embedding_filename, index=False)
            logger.info(f"Saved {len(self.embeddings)} embedding records to {self.embedding_filename}")
        
        # Rebuild indices
        self._build_indices()

    def _upsert(self, hash_ids: List[str], contents: List[T], embeddings: Optional[List] = None):
        """Insert or update content and embeddings."""
        self.hash_ids.extend(hash_ids)
        self.contents.extend(contents)
        
        if self.enable_embeddings and embeddings is not None:
            self.embeddings.extend(embeddings)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids: List[str]):
        """Delete content by hash IDs."""
        indices = []

        for hash_id in hash_ids:
            if hash_id in self.hash_id_to_idx:
                indices.append(self.hash_id_to_idx[hash_id])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.contents.pop(idx)
            if self.enable_embeddings and self.embeddings:
                self.embeddings.pop(idx)

        logger.info(f"Saving records after deletion.")
        self._save_data()

    def get_row(self, hash_id: str) -> Dict[str, Any]:
        """Get content row by hash ID."""
        return self.hash_id_to_row[hash_id]

    def get_content(self, hash_id: str) -> T:
        """Get content object by hash ID."""
        idx = self.hash_id_to_idx[hash_id]
        return self.contents[idx]

    def get_hash_id(self, content: Union[str, T]) -> str:
        """Get hash ID for content."""
        if isinstance(content, str) and content in self.text_to_hash_id:
            return self.text_to_hash_id[content]
        else:
            # Compute hash for the content
            return self._compute_content_hash(content)

    def get_rows(self, hash_ids: List[str], dtype=np.float32) -> Dict[str, Dict[str, Any]]:
        """Get multiple content rows by hash IDs."""
        if not hash_ids:
            return {}

        results = {hash_id: self.hash_id_to_row[hash_id] for hash_id in hash_ids if hash_id in self.hash_id_to_row}
        return results

    def get_all_ids(self) -> List[str]:
        """Get all hash IDs."""
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        """Get all hash ID to row mappings."""
        return deepcopy(self.hash_id_to_row)

    def get_all_contents(self) -> List[T]:
        """Get all content objects."""
        return deepcopy(self.contents)

    def get_all_texts(self) -> Set[str]:
        """Get all text content (backward compatibility)."""
        if self.contents and isinstance(self.contents[0], str):
            return set(self.contents)
        else:
            return set(row['content'] for row in self.hash_id_to_row.values() if isinstance(row['content'], str))

    def get_embedding(self, hash_id: str, dtype=np.float32) -> Optional[np.ndarray]:
        """Get embedding by hash ID."""
        if not self.enable_embeddings or not self.embeddings:
            return None
        
        if hash_id not in self.hash_id_to_idx:
            return None
            
        idx = self.hash_id_to_idx[hash_id]
        return np.array(self.embeddings[idx], dtype=dtype)
    
    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> List[np.ndarray]:
        """Get multiple embeddings by hash IDs."""
        if not self.enable_embeddings or not self.embeddings or not hash_ids:
            return []

        valid_indices = []
        for h in hash_ids:
            if h in self.hash_id_to_idx:
                valid_indices.append(self.hash_id_to_idx[h])

        if not valid_indices:
            return []

        indices = np.array(valid_indices, dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]
        return embeddings


# Backward compatibility class
class EmbeddingStore(ContentStore[str]):
    """Backward compatibility wrapper for the original EmbeddingStore interface."""
    
    def __init__(self, embedding_model, db_filename: str, batch_size: int, namespace: str):
        # Use string identity function for text extraction
        super().__init__(
            db_filename=db_filename,
            namespace=namespace,
            batch_size=batch_size,
            embedding_model=embedding_model,
            text_extraction_fn=lambda x: x,  # Identity function for strings
            enable_embeddings=True
        )
        
        # For backward compatibility, maintain the old attribute names
        self.filename = os.path.join(db_filename, f"vdb_{namespace}.parquet")

    @property
    def texts(self):
        """Backward compatibility property."""
        return self.contents