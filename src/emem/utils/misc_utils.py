from argparse import ArgumentTypeError
import chunk
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
from datetime import datetime
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)


@dataclass
class QueryNerRawOutput:
    query: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class EDURawOutput:
    chunk_id: str
    response: str
    edus: List[str]
    metadata: Dict[str, Any]


@dataclass
class EDURERawOutput:
    chunk_id: str
    response: str
    edus: List[str]
    metadata: Dict[str, Any]
    edu2edu_relative_id: Optional[dict] = None
    edu_relative_id2edu: Optional[dict] = None
    edu_relation_triples: Optional[List[List[str]]] = None
    

    
@dataclass
class EventRawOutput:
    chunk_id: str
    edu_id: str
    edu_text: str
    response: Dict[str, str] # key is the subtask name and value is corresponding response str
    metadata: Dict[str, Dict[str, Any]] # key is the subtask name and value is the corresponding metadata
    event_type: Optional[str] = None
    event_triggers: Optional[List[str]] = None
    event_role_argument_pairs: Optional[List[Dict]] = None # {"role": role_argument_pair.role, "argument": role_argument_pair.argument}
    
    
@dataclass
class ContextualEventRawOutput:
    chunk_id: str
    edu_id: str
    edu_text: str
    context_text: str
    response: Dict[str, str] # key is the subtask name and value is corresponding response str
    metadata: Dict[str, Dict[str, Any]] # key is the subtask name and value is the corresponding metadata
    event_type: Optional[str] = None
    event_triggers: Optional[List[str]] = None
    event_role_argument_pairs: Optional[List[Dict]] = None # {"role": role_argument_pair.role, "argument": role_argument_pair.argument}


@dataclass
class SessionSummaryRawOutput:
    session_id: str
    summary: str
    response: str
    metadata: Dict[str, Any]


@dataclass
class ConversationEDURawOutput:
    session_id: str
    edu_id: str
    edu_text: str
    source_speakers: List[str]
    date: Union[str, datetime]  # Support both string and datetime for backward compatibility
    response: str
    metadata: Dict[str, Any]


@dataclass
class ConversationEDURawOutputWithTurnIds(ConversationEDURawOutput):
    """Extended version with turn ID attribution for more precise tracking."""
    source_turn_ids: Optional[List[int]] = None


@dataclass
class ConversationContextualEventRawOutput:
    session_id: str
    edu_id: str
    edu_text: str
    source_speakers: List[str]
    date: Union[str, datetime]  # Support both string and datetime for backward compatibility
    context_text: str
    response: Dict[str, str] # key is the subtask name and value is corresponding response str
    metadata: Dict[str, Dict[str, Any]] # key is the subtask name and value is the corresponding metadata
    event_type: Optional[str] = None
    event_triggers: Optional[List[str]] = None
    event_role_argument_pairs: Optional[List[Dict]] = None # {"role": role_argument_pair.role, "argument": role_argument_pair.argument}

    # def to_context_string(self, date_format_type: str = "locomo") -> str:
    #     """Convert the EDU to a formatted context string for QA."""
    #     from .date_utils import format_date_for_dataset, parse_date_string
        
    #     speakers_str = ', '.join(self.source_speakers) if self.source_speakers else 'Unknown'
        
    #     # Handle both string and datetime date formats
    #     if isinstance(self.date, datetime):
    #         date_str = format_date_for_dataset(self.date, date_format_type)
    #     else:
    #         # Try to parse string date for better formatting
    #         parsed_date = parse_date_string(self.date)
    #         date_str = format_date_for_dataset(parsed_date, date_format_type) if parsed_date else str(self.date)
        
    #     context_str = f"[Source conversation date: {date_str} - Source speakers: {speakers_str}] \"{self.edu_text}\""
    #     return context_str

    def to_context_string(self, date_format_type: str = "locomo") -> str:
        """Convert the EDU to a formatted context string for QA."""
        from .date_utils import format_date_for_dataset, parse_date_string
        
        speakers_str = ', '.join(self.source_speakers) if self.source_speakers else 'Unknown'
        
        # Handle both string and datetime date formats
        if isinstance(self.date, datetime):
            date_str = format_date_for_dataset(self.date, date_format_type)
        else:
            # Try to parse string date for better formatting
            parsed_date = parse_date_string(self.date)
            date_str = format_date_for_dataset(parsed_date, date_format_type) if parsed_date else str(self.date)
        
        if self.metadata.get('edu_type', '') == 'assistant_chunk':
            context_str = f"[Source conversation date: {date_str} - Source speakers: {speakers_str}] \"{self.metadata.get('chunk_content', '')}\""
        else:
            context_str = f"[Source conversation date: {date_str} - Source speakers: {speakers_str}] \"{self.edu_text}\""
        return context_str

    def get_text(self) -> str:
        """Get the EDU text content."""
        return self.edu_text


@dataclass
class ConversationContextualEventRawOutputWithTurnIds(ConversationContextualEventRawOutput):
    """
    Extended version with turn ID attribution for more precise tracking.
    
    Can represent three types via metadata['edu_type']:
    - 'user_edu': Simple EDU from User
    - 'assistant_edu': Simple EDU from Assistant
    - 'assistant_chunk': Structured chunk from Assistant
    
    For 'assistant_chunk' type:
    - edu_text: Contains chunk_summary (for retrieval)
    - metadata['chunk_content']: Contains full chunk_content (for event extraction/augmentation)
    """
    source_turn_ids: Optional[List[int]] = None

    


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[Any]
    doc_scores: Optional[np.ndarray] = None
    answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None
    edus: Optional[List[Union[str, 'ConversationContextualEventRawOutput']]] = None
    edu_scores: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
            "edus": self.edus,
            "edu_scores": [round(v, 4) for v in self.edu_scores.tolist()[:5]]  if self.edu_scores is not None else None,
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def reformat_openie_results(corpus_openie_results, openie_mode="online") -> Union[Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]], Dict[str, List[EventRawOutput]]]:

    # Process each chunk_item to remove role argument pairs with empty arguments
    for chunk_item in corpus_openie_results:
        for edu in chunk_item["edus"]:
            # Check if the edu has event_role_argument_pairs
            if edu.event_role_argument_pairs is not None:
                # Count empty arguments before filtering
                empty_count = sum(1 for pair in edu.event_role_argument_pairs if pair["argument"] == '')
                if empty_count > 0:
                    logger.warning(f"Removing {empty_count} empty argument pair(s) from edu: {edu.edu_id}")
                
                # # Filter out role argument pairs with empty arguments
                # edu.event_role_argument_pairs = [
                #     pair for pair in edu.event_role_argument_pairs 
                #     if pair["argument"] != ''
                # ]
                # Filter out role argument pairs with empty arguments
                edu.event_role_argument_pairs = [
                    pair for pair in edu.event_role_argument_pairs 
                    if pair["argument"] != ''
                ]
    
    return {
        chunk_item['idx']: chunk_item["edus"]
        for chunk_item in corpus_openie_results
    }, {chunk_item['idx']: chunk_item['session_summary'] for chunk_item in corpus_openie_results}


def reformat_openie_results_original(corpus_openie_results, openie_mode="online") -> Union[Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]], Dict[str, List[EventRawOutput]]]:
    return {
        chunk_item['idx']: chunk_item["edus"]
        for chunk_item in corpus_openie_results
    }, {chunk_item['idx']: chunk_item['session_summary'] for chunk_item in corpus_openie_results}
    if openie_mode in ["edu_based_ee_online", ]:
        event_output_dict = {
            chunk_item['idx']: [
                EventRawOutput(
                    chunk_id=chunk_item['idx'],
                    edu_id=edu_item['edu_idx'],
                    edu_text=edu_item['edu_text'],
                    response=edu_item['response'],
                    metadata=edu_item['metadata'],
                    event_type=edu_item.get('event_type'),
                    event_triggers=edu_item.get('event_triggers'),
                    event_role_argument_pairs=edu_item.get('event_role_argument_pairs')
                )
                for edu_item in chunk_item['edus']
            ]
            for chunk_item in corpus_openie_results
        }
        return event_output_dict
    elif openie_mode in ["edu_based_contextual_ee_online", ]:
        event_output_dict = {
            chunk_item['idx']: [
                ContextualEventRawOutput(
                    chunk_id=chunk_item['idx'],
                    edu_id=edu_item['edu_idx'],
                    edu_text=edu_item['edu_text'],
                    # edu_text=f"{edu_item['context_text']} {edu_item['edu_text']}",
                    context_text=edu_item['context_text'],
                    response=edu_item['response'],
                    metadata=edu_item['metadata'],
                    event_type=edu_item.get('event_type'),
                    event_triggers=edu_item.get('event_triggers'),
                    event_role_argument_pairs=edu_item.get('event_role_argument_pairs')
                )
                for edu_item in chunk_item['edus']
            ]
            for chunk_item in corpus_openie_results
        }
        return event_output_dict
    else:
        ner_output_dict = {
            chunk_item['idx']: NerRawOutput(
                chunk_id=chunk_item['idx'],
                response=None,
                metadata={},
                unique_entities=list(np.unique(chunk_item['extracted_entities']))
            )
            for chunk_item in corpus_openie_results
        }
        triple_output_dict = {
            chunk_item['idx']: TripleRawOutput(
                chunk_id=chunk_item['idx'],
                response=None,
                metadata={},
                triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
            )
            for chunk_item in corpus_openie_results
        }

        return ner_output_dict, triple_output_dict


def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> tuple[List[str], List[List[str]]]:
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


import json

def read_jsonl(file_path: str):
    """
    Reads a JSONL file and returns a list of JSON objects.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: A list of JSON objects, typically dictionaries.
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove any extraneous whitespace/newline characters.
            if line:  # Ensure that the line is not empty.
                results.append(json.loads(line))
    return results

def write_jsonl(file_path: str, data: list):
    """
    Writes a list of JSON objects to a JSONL file.
    
    Each JSON object in `data` is written to a separate line in the file.
    
    Args:
        file_path (str): Path to the JSONL file to write.
        data (list): A list of JSON-serializable Python objects (e.g., dictionaries).
    
    Returns:
        None
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')



def append_to_jsonl(data, filename: str, default=None) -> None:
    """Append to the end of a jsonl file.

    Args:
        data: Any JSON-serializable object.
        filename (str): Destination JSONL file.
        default (callable, optional): Custom JSON encoder for non-serializable types.
    """
    json_string = json.dumps(data, default=default)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json_string + "\n")