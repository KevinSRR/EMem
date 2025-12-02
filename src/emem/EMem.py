import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from pandas._config import _global_config
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
from copy import deepcopy
import pickle

from .llm import _get_llm_class, BaseLLM
from .llm.openai_gpt_batch import cache_json_encoder
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenStructuring
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .rerank_structuring import DSPyFilterStructuring
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput, ConversationContextualEventRawOutput, ConversationContextualEventRawOutputWithTurnIds
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

from .utils.conversation_data_utils import LoCoMoSample, Session, QA
from .content_store import ContentStore
from pydantic import BaseModel, Field
from typing import Literal

# Import response model from prompt template
from .prompts.templates.conversation_qa import ConversationQAResponse

logger = logging.getLogger(__name__)


class EMem:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific model instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (OpenStructuring): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"Model init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        self.openie = OpenStructuring(llm_model=self.llm_model) # max_workers not specified here as we will go with async

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        

        self.conversation_content_store = ContentStore[LoCoMoSample](
            db_filename=os.path.join(self.working_dir, "conversation_contents"),
            namespace="conversation",
            batch_size=self.global_config.embedding_batch_size,
            embedding_model=None,
            text_extraction_fn=None,
            enable_embeddings=False,
        )

        self.session_content_store = ContentStore[Session](
            db_filename=os.path.join(self.working_dir, "session_contents"),
            namespace="session",
            batch_size=self.global_config.embedding_batch_size,
            embedding_model=None,
            text_extraction_fn=None,
            enable_embeddings=False,
        )

        self.edu_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "edu_embeddings"),
                                                    self.global_config.embedding_batch_size, 'edu')
        self.argument_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "argument_embeddings"),
                                                    self.global_config.embedding_batch_size, 'argument')


        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir, f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.pkl')

        self.rerank_filter = DSPyFilter(self)
        self.rerank_filter_structuring = DSPyFilterStructuring(self)


        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph


    def index_conversation(self, conversation: LoCoMoSample):
        logger.info("Indexing Conversation")

        self.conversation_content_store.insert_content([conversation])
        
        self.speaker_a, self.speaker_b = conversation.conversation.speaker_a, conversation.conversation.speaker_b

        self.session_content_store.insert_content([session for session_id, session in conversation.conversation.sessions.items()])
        session_to_rows = self.session_content_store.get_all_id_to_rows()

        all_openie_info, session_keys_to_process = self.load_existing_openie(session_to_rows.keys())
        new_openie_rows = {k: session_to_rows[k] for k in session_keys_to_process}

        if len(session_keys_to_process) > 0:
            assert len(session_keys_to_process) == len(conversation.conversation.sessions), "We need to structure all sessions in sequential order in one shot in order to keep the cumulative summary functioning."

            logger.info(f"Conducting contextualized edu based ee online.")
            if self.global_config.date_format_type == "longmemeval":
                new_structuring_results_dict = self.openie.batch_conversation_openie_w_indep_summary_uniee_separate_edus_v2_wo_session_summary(sessions={k: v['content'] for k, v in new_openie_rows.items()}, speaker_names=[self.speaker_a, self.speaker_b], skip_edu_context_gen=self.global_config.skip_edu_context_gen) # from session hash key to a list of ConversationContextualEventRawOutputWithTurnIds in order for each extracted edus of each session
            else:
                new_structuring_results_dict = self.openie.batch_conversation_openie_w_indep_summary_uniee_v1_wo_session_summary(sessions={k: v['content'] for k, v in new_openie_rows.items()}, speaker_names=[self.speaker_a, self.speaker_b], skip_edu_context_gen=self.global_config.skip_edu_context_gen)

            self.merge_structuring_results(all_openie_info, new_openie_rows, new_structuring_results_dict, self.global_config.openie_mode)

        if self.global_config.save_openie and len(session_keys_to_process) > 0:
            self.save_structuring_results(all_openie_info, openie_mode=self.global_config.openie_mode)


        structuring_results_dict, session_id_to_summary = reformat_openie_results(all_openie_info, openie_mode=self.global_config.openie_mode) # from session hash id to a list of ConversationContextualEventRawOutput (for each edu), from session hash id to the summary str or None
        assert len(session_to_rows) == len(structuring_results_dict)
        
        # Store session_id_to_summary as instance attribute for use in add_new_nodes_structuring
        self.session_id_to_summary = session_id_to_summary

        session_ids = list(session_to_rows.keys())
        session_edus = [[edu_result.edu_text for edu_result in structuring_results_dict[session_id]] for session_id in session_ids]
        session_args = []
        for session_id in session_ids:
            session_args.append([])
            for edu_result in structuring_results_dict[session_id]:
                if edu_result.event_role_argument_pairs is None:
                    logger.warning(f"Empty role argument pairs extracted for edus: {edu_result}")
                    continue
                for role_arg_pair in edu_result.event_role_argument_pairs:
                    # if not role_arg_pair["argument"]: 
                    #     logger.warning(f"Empty argument: {edu_result.event_role_argument_pairs}")
                    #     raise
                    session_args[-1].append(role_arg_pair["argument"])

        # sessions are already inserted
        logger.info(f"Encoding EDUs")
        newly_inserted_session_ids = self.edu_embedding_store.insert_strings(
            texts=[edu for edu_list in session_edus for edu in edu_list],
            # encoding_instruction="passage" if "e5" in self.global_config.embedding_model_name.lower() else None
        ) # this function will be responsible to resolve repetitions

        logger.info(f"Encoding Arguments")
        self.argument_embedding_store.insert_strings(
            texts=[argument for arg_list in session_args for argument in arg_list],
            # encoding_instruction="query" if "e5" in self.global_config.embedding_model_name.lower() else None
        )

        logger.info(f"Constructing Graph")
        self.node_id_to_node_id_stats = {}
        self.edu_id_to_session_ids = {}
        self.arg_id_to_session_ids = {}
        # Track edge components for detailed edge metadata
        self.edge_components = {} # Maps (source, target) tuple to list of component dicts

        self.add_role_arg_edges(session_ids, structuring_results_dict)
        num_new_sessions = self.add_context_edges(session_ids, structuring_results_dict)

        if num_new_sessions > 0:
            logger.info(f"Found {num_new_sessions} new sessions to save into graph")
            self.add_synonymy_edges_structuring()
            self.augment_graph()
            self.save_igraph()
            
            # Update date_to_session_id mapping for new sessions
            self._update_date_to_session_mapping(session_ids, session_to_rows)
            
            self._save_conversation_graph_attributes()
            
        else:
            self._load_conversation_graph_attributes()

    def _update_date_to_session_mapping(self, session_ids, session_to_rows):
        """
        Updates the date_to_session_id mapping for new sessions.
        
        Args:
            session_ids: List of session IDs to process
            session_to_rows: Dictionary mapping session IDs to their row data
        """
        from .utils.date_utils import parse_date_string
        
        if not hasattr(self, 'date_to_session_id'):
            self.date_to_session_id = {}
        
        for session_id in session_ids:
            session_data = session_to_rows[session_id]['content']  # This is the Session object
            
            if hasattr(session_data, 'date_time') and session_data.date_time:
                date_parsed = parse_date_string(session_data.date_time)
                if date_parsed:
                    self.date_to_session_id[date_parsed] = session_id
                    logger.debug(f"Added date mapping: {date_parsed} -> {session_id}")
                else:
                    logger.warning(f"Failed to parse date '{session_data.date_time}' for session {session_id}")

    def _save_conversation_graph_attributes(self):
        """
        Saves conversation-specific graph attributes to a pickle file.
        
        Saves the following attributes:
        - node_id_to_node_id_stats: Edge statistics between nodes
        - edu_id_to_session_ids: Mapping from EDU IDs to session IDs they appear in
        - arg_id_to_session_ids: Mapping from argument IDs to session IDs they appear in
        - edge_components: Detailed edge component metadata
        - date_to_session_id: Mapping from parsed datetime objects to session IDs
        """
        conversation_graph_attrs = {
            'node_id_to_node_id_stats': self.node_id_to_node_id_stats,
            'edu_id_to_session_ids': self.edu_id_to_session_ids,
            'arg_id_to_session_ids': self.arg_id_to_session_ids,
            'edge_components': self.edge_components,
            'date_to_session_id': getattr(self, 'date_to_session_id', {})
        }
        
        conversation_attrs_path = os.path.join(self.working_dir, "conversation_graph_attributes.pkl")
        
        with open(conversation_attrs_path, 'wb') as f:
            pickle.dump(conversation_graph_attrs, f)
        
        logger.info(f"Conversation graph attributes saved to {conversation_attrs_path}")

    def _load_conversation_graph_attributes(self):
        """
        Loads conversation-specific graph attributes from a pickle file if it exists.
        
        Loads the following attributes:
        - node_id_to_node_id_stats: Edge statistics between nodes
        - edu_id_to_session_ids: Mapping from EDU IDs to session IDs they appear in
        - arg_id_to_session_ids: Mapping from argument IDs to session IDs they appear in
        - edge_components: Detailed edge component metadata
        - date_to_session_id: Mapping from parsed datetime objects to session IDs
        
        If the file doesn't exist, initializes these attributes as empty dictionaries.
        """
        conversation_attrs_path = os.path.join(self.working_dir, "conversation_graph_attributes.pkl")
        
        if os.path.exists(conversation_attrs_path):
            try:
                with open(conversation_attrs_path, 'rb') as f:
                    conversation_graph_attrs = pickle.load(f)
                
                self.node_id_to_node_id_stats = conversation_graph_attrs.get('node_id_to_node_id_stats', {})
                self.edu_id_to_session_ids = conversation_graph_attrs.get('edu_id_to_session_ids', {})
                self.arg_id_to_session_ids = conversation_graph_attrs.get('arg_id_to_session_ids', {})
                self.edge_components = conversation_graph_attrs.get('edge_components', {})
                self.date_to_session_id = conversation_graph_attrs.get('date_to_session_id', {})
                
                logger.info(f"Conversation graph attributes loaded from {conversation_attrs_path}")
            except Exception as e:
                logger.warning(f"Failed to load conversation graph attributes from {conversation_attrs_path}: {str(e)}")
                logger.info("Initializing conversation graph attributes as empty dictionaries")
                self._initialize_empty_conversation_graph_attributes()
        else:
            logger.info("No existing conversation graph attributes file found. Initializing as empty dictionaries")
            self._initialize_empty_conversation_graph_attributes()

    def _initialize_empty_conversation_graph_attributes(self):
        """
        Initializes conversation graph attributes as empty dictionaries.
        """
        self.node_id_to_node_id_stats = {}
        self.edu_id_to_session_ids = {}
        self.arg_id_to_session_ids = {}
        self.edge_components = {}
        self.date_to_session_id = {}
    

    def _enrich_edus_with_metadata(self, edu_ids: List[int], edu_scores: np.ndarray) -> List[Union['ConversationContextualEventRawOutput', 'ConversationContextualEventRawOutputWithTurnIds']]:
        """
        Enrich EDUs with speaker and temporal metadata from their source sessions.
        
        Args:
            edu_ids: List of EDU node indices
            edu_scores: Array of EDU scores
        
        Returns:
            List of ConversationContextualEventRawOutput objects (or ConversationContextualEventRawOutputWithTurnIds if turn attribution is used) with metadata
        """
        enriched_edus = []
        
        for i, edu_idx in enumerate(edu_ids):
            # Get EDU content
            edu_key = self.edu_node_keys[edu_idx]
            edu_row = self.edu_embedding_store.get_row(edu_key)
            edu_text = edu_row["content"]
            
            # Find the corresponding ConversationContextualEventRawOutput from structuring results
            found_edu = None
            for session_key, session_edus in self.structuring_results_dict.items():
                for edu_obj in session_edus:
                    if hasattr(edu_obj, 'edu_text') and edu_obj.edu_text == edu_text:
                        found_edu = edu_obj
                        break
                if found_edu:
                    break
            
            if found_edu and isinstance(found_edu, ConversationContextualEventRawOutput):
                # Use the existing enriched EDU object
                enriched_edus.append(found_edu)
            else:
                # Fallback: create a minimal ConversationContextualEventRawOutput
                logger.warning(f"Could not find enriched EDU for text: {edu_text[:50]}...")
                raise ValueError(f"Could not find enriched EDU for text: {edu_text[:50]}...")
        
        return enriched_edus


    def add_role_arg_edges(self, chunk_ids: List[str], structuring_results_dict: Dict[str, List[EventRawOutput]]):
        """
        Adds edges between EDU nodes and argument nodes to the graph based on event role-argument pairs extracted from structured results.

        For each chunk, this function processes its list of EDU extraction results. For each EDU, if event role-argument pairs are present, it computes unique hash IDs for the EDU text and each argument. It then updates the node-to-node statistics for both (EDU, argument) and (argument, EDU) edges, and tracks which chunk each EDU and argument node appears in. 
        
        Additionally, this function tracks detailed edge component information including the edge type ("EDU-Arg"), the role content, weight (1.0), and directional information for later inclusion in the graph's edge attributes.

        Parameters:
            chunk_ids (List[str]):
                A list of unique identifiers for the chunks being processed.
            structuring_results_dict (Dict[str, List[EventRawOutput]]):
                A dictionary mapping chunk IDs to lists of EventRawOutput objects, each representing an EDU and its extracted event information.
        
        Side Effects:
            - Updates self.node_id_to_node_id_stats with edge weights
            - Updates self.edge_components with component metadata for EDU-Arg edges
            - Updates self.edu_id_to_chunk_ids and self.arg_id_to_chunk_ids mappings
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding structuring role and arguments to graph.")

        # self.node_id_to_node_id_stats = {}
        # self.edu_id_to_chunk_ids = {}
        # self.arg_id_to_chunk_ids = {}
        for chunk_key in tqdm(chunk_ids):
            if chunk_key not in current_graph_nodes:
                edu_results = structuring_results_dict[chunk_key]
                arg_ids_in_chunk = set()
                edu_ids_in_chunk = set()

                for edu_result in edu_results:
                    if edu_result.event_role_argument_pairs is None:
                        logger.warning(f"Empty event role arg paris for edu: {edu_result.edu_text}")
                        continue
                    
                    edu_key = compute_mdhash_id(content=edu_result.edu_text, prefix=("edu-"))
                    for role_arg_pair in edu_result.event_role_argument_pairs:
                        arg_key = compute_mdhash_id(content=role_arg_pair["argument"], prefix=("argument-"))

                        self.node_id_to_node_id_stats[(edu_key, arg_key)] = self.node_id_to_node_id_stats.get(
                            (edu_key, arg_key), 0.0
                        ) + 1
                        self.node_id_to_node_id_stats[(arg_key, edu_key)] = self.node_id_to_node_id_stats.get(
                            (arg_key, edu_key), 0.0
                        ) + 1
                        
                        # Track edge components for EDU-Arg relationships
                        component = {
                            "type": "EDU-Arg",
                            "content": role_arg_pair["role"],
                            "weight": 1.0,
                            "direction": [edu_key, arg_key],
                            "metadata": {
                                "event_type": edu_result.event_type,
                                "event_triggers": edu_result.event_triggers,
                            },
                        }
                        if (edu_key, arg_key) not in self.edge_components:
                            self.edge_components[(edu_key, arg_key)] = []
                        self.edge_components[(edu_key, arg_key)].append(component)
                        
                        # Add reverse direction component
                        reverse_component = {
                            "type": "EDU-Arg",
                            "content": role_arg_pair["role"],
                            "weight": 1.0,
                            "direction": [arg_key, edu_key],
                            "metadata": {
                                "event_type": edu_result.event_type,
                                "event_triggers": edu_result.event_triggers,
                            },
                        }
                        if (arg_key, edu_key) not in self.edge_components:
                            self.edge_components[(arg_key, edu_key)] = []
                        self.edge_components[(arg_key, edu_key)].append(reverse_component)
                        
                        edu_ids_in_chunk.add(edu_key)
                        arg_ids_in_chunk.add(arg_key)

                for edu_id in edu_ids_in_chunk:
                    self.edu_id_to_session_ids[edu_id] = self.edu_id_to_session_ids.get(edu_id, set()).union(set([chunk_key]))
                for arg_id in arg_ids_in_chunk:
                    self.arg_id_to_session_ids[arg_id] = self.arg_id_to_session_ids.get(arg_id, set()).union(set([chunk_key]))


    def add_context_edges(self, chunk_ids: List[str], structuring_results_dict: Dict[str, List[EventRawOutput]]):
        """
        Adds edges connecting passage nodes to EDU nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding structuring results. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the EDU
        nodes (defined by the computed unique hash IDs of EDU texts). The method
        also updates the node-to-node statistics map (not incrementally, always assign the edge weight to 1.0) and keeps count of newly added
        passage nodes.
        
        Additionally, this function tracks detailed edge component information including the edge type ("EDU-Passage_context"), context text content (if available for contextual mode), weight (1.0), and directional information for later inclusion in the graph's edge attributes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            structuring_results_dict : Dict[str, List[EventRawOutput]]
                A dictionary mapping chunk IDs to lists of EventRawOutput objects,
                each containing EDU information extracted from the corresponding chunk.

        Returns:
            int
                The number of new passage nodes added to the graph.
                
        Side Effects:
            - Updates self.node_id_to_node_id_stats with edge weights
            - Updates self.edge_components with component metadata for EDU-Passage_context edges
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting session nodes to EDU nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                edu_results: List[ConversationContextualEventRawOutput] = structuring_results_dict[chunk_key]
                
                for edu_result in edu_results:
                    if not edu_result.edu_text:
                        logger.warning(f"Empty edu_text for session id: {edu_result.session_id}")
                        continue
                    
                    edu_key = compute_mdhash_id(content=edu_result.edu_text, prefix=("edu-"))
                    self.node_id_to_node_id_stats[(chunk_key, edu_key)] = 1.0
                    
                    # Track edge components for EDU-Session context relationships
                    # Get context text if available (for contextual mode)
                    context_text = None
                    if hasattr(edu_result, 'context_text') and edu_result.context_text:
                        context_text = edu_result.context_text
                    
                    # Parse date to datetime object for consistent handling
                    from .utils.date_utils import parse_date_string
                    parsed_date = parse_date_string(edu_result.date)
                    
                    component = {
                        "type": "EDU-Session_context",
                        "content": context_text,
                        "weight": 1.0,
                        "direction": [chunk_key, edu_key],
                        "metadata": {
                            "source_speakers": edu_result.source_speakers,
                            "date": parsed_date,  # Store as datetime object
                        },
                    }
                    if (chunk_key, edu_key) not in self.edge_components:
                        self.edge_components[(chunk_key, edu_key)] = []
                    self.edge_components[(chunk_key, edu_key)].append(component)

                num_new_chunks += 1

        return num_new_chunks


    def add_synonymy_edges_structuring(self):
        """
        Adds synonymy edges between similar arguments in the structuring mode to enhance connectivity.

        This method performs key operations to compute and add synonymy edges for arguments (which are equivalent to entities
        in the standard OpenIE mode). It first retrieves embeddings for all argument nodes, then conducts nearest neighbor 
        (KNN) search to find similar arguments. These similar arguments are identified based on a score threshold, and edges 
        are added to represent the synonym relationship for coreference resolution.
        
        Additionally, this function tracks detailed edge component information including the edge type ("Arg-Arg_synonymy"), no content (None), similarity score as weight, and no direction (None, indicating bidirectional) for later inclusion in the graph's edge attributes.

        Attributes:
            argument_id_to_row: dict (populated within the function). Maps each argument ID to its corresponding row data.
            argument_embedding_store: Manages retrieval of texts and embeddings for all rows related to arguments.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_id_to_node_id_stats: dict. Stores scores for edges between nodes representing their relationship.
            
        Side Effects:
            - Updates self.node_id_to_node_id_stats with synonymy edge weights
            - Updates self.edge_components with component metadata for Arg-Arg_synonymy edges

        """
        logger.info(f"Expanding graph with synonymy edges for structuring mode (arguments only)")

        # Process Arguments (equivalent to entities in standard mode)
        self.argument_id_to_row = self.argument_embedding_store.get_all_id_to_rows()
        argument_node_keys = list(self.argument_id_to_row.keys())

        if len(argument_node_keys) > 0:
            logger.info(f"Performing KNN retrieval for each argument nodes ({len(argument_node_keys)}).")

            argument_embs = self.argument_embedding_store.get_embeddings(argument_node_keys)

            # Build synonymy edges between argument nodes
            query_arg_key2knn_arg_keys = retrieve_knn(query_ids=argument_node_keys,
                                                      key_ids=argument_node_keys,
                                                      query_vecs=argument_embs,
                                                      key_vecs=argument_embs,
                                                      k=self.global_config.synonymy_edge_topk,
                                                      query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                      key_batch_size=self.global_config.synonymy_edge_key_batch_size)

            num_synonym_arg_edges = 0

            for arg_key in tqdm(query_arg_key2knn_arg_keys.keys(), total=len(query_arg_key2knn_arg_keys), desc="Processing argument synonymy"):
                arg_text = self.argument_id_to_row[arg_key]["content"]

                if len(re.sub('[^A-Za-z0-9]', '', arg_text)) > 2:
                    nns = query_arg_key2knn_arg_keys[arg_key]

                    num_nns = 0
                    for nn, score in zip(nns[0], nns[1]):
                        if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                            break

                        nn_arg_text = self.argument_id_to_row[nn]["content"]

                        if nn != arg_key and nn_arg_text != '':
                            sim_edge = (arg_key, nn)
                            num_synonym_arg_edges += 1

                            self.node_id_to_node_id_stats[sim_edge] = score
                            
                            # Track edge components for Arg-Arg synonymy relationships
                            component = {
                                "type": "Arg-Arg_synonymy",
                                "content": None,
                                "weight": score,
                                "direction": None,  # Bidirectional for synonymy
                                "metadata": {
                                    "sim_score": score,
                                },
                            }
                            if sim_edge not in self.edge_components:
                                self.edge_components[sim_edge] = []
                            self.edge_components[sim_edge].append(component)
                            
                            num_nns += 1

            logger.info(f"Added {num_synonym_arg_edges} argument synonymy edges")
        else:
            logger.info("No arguments found for synonymy edge creation")
            num_synonym_arg_edges = 0

        logger.info(f"Total synonymy edges added in structuring mode: {num_synonym_arg_edges}")

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Assumes the openie data contains the `docs` field which is a list of each chunk's openie results. For each chunk's
        openie result, there will be one `passage` field containing the content of the chunk, and there will be an added field
        `idx` when calling this function containing the hash of the chunk content. 

        
        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            
            with open(self.openie_results_path, 'rb') as f:
                openie_results = pickle.load(f)
            all_openie_info = openie_results.get('sessions', [])


            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:

                openie_info['idx'] = self.session_content_store._compute_content_hash(content=openie_info['session'])
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save


    def merge_structuring_results(
        self,
        all_openie_info: List[dict],
        chunks_to_save: Dict[str, dict],
        structuring_results_dict: Dict[str, Any],
        openie_mode: str
    ) -> List[dict]:
        """
        Merges structuring results with corresponding passage and metadata.

        This function integrates the structuring results, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged structuring
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge structuring results to dictionaries with `hash_id` and `content` keys.
           structuring_results_dict (Dict[str, Any]): A dictionary mapping chunk keys
                to their corresponding structuring results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            structuring results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage: Session = row['content']
            chunk_openie_info = {
                'idx': chunk_key, 
                'session': passage,
                'session_summary': structuring_results_dict[chunk_key]['summary'],
                'edus': structuring_results_dict[chunk_key]['edus']
            }
            all_openie_info.append(chunk_openie_info)

        return all_openie_info


    def save_structuring_results(self, all_openie_info: List[dict], openie_mode: str = None):
        """
        Computes statistics on event extraction results from conversation structuring and saves the aggregated data in a
        pickle file. The function calculates EDU coverage ratios, event extraction density, and role-argument
        statistics to monitor the health of event extraction.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from conversation structuring, including
                extracted EDUs with event types, triggers, and role-argument pairs from conversation sessions.
        """

        # Calculate session text length for conversation sessions
        total_session_words = 0
        for chunk in all_openie_info:
            session = chunk['session']
            # Calculate total words in session turns
            session_words = sum([len(turn.text.split()) for turn in session.turns])
            total_session_words += session_words
            
        # Calculate EDU coverage statistics (how much of original session is covered by EDUs)
        total_edu_words = sum([len(edu.edu_text.split()) for chunk in all_openie_info for edu in chunk['edus']])
        
        # Calculate event extraction statistics
        total_edus = sum([len(chunk['edus']) for chunk in all_openie_info])
        total_triggers = sum([len(edu.event_triggers) if edu.event_triggers else 0 for chunk in all_openie_info for edu in chunk['edus']])
        total_role_args = sum([len(edu.event_role_argument_pairs) if edu.event_role_argument_pairs else 0 for chunk in all_openie_info for edu in chunk['edus']])

        # Calculate context_text statistics
        context_word_counts = []
        context_to_session_ratios = []
        context_to_edu_ratios = []
        
        for chunk in all_openie_info:
            session = chunk['session']
            session_words = sum([len(turn.text.split()) for turn in session.turns])
            
            for edu in chunk['edus']:
                edu_words = len(edu.edu_text.split())
                context_text = edu.context_text if edu.context_text else ''
                if isinstance(context_text, str) and len(context_text) > 0:
                    c_words = len(context_text.split())
                    context_word_counts.append(c_words)
                    if session_words > 0:
                        context_to_session_ratios.append(c_words / session_words)
                    if edu_words > 0:
                        context_to_edu_ratios.append(c_words / edu_words)
        
        total_context_texts = len(context_word_counts)
        total_context_words = sum(context_word_counts) if context_word_counts else 0
        avg_context_words = round(total_context_words / total_context_texts, 4) if total_context_texts > 0 else 0
        avg_context_to_session_ratio = round(sum(context_to_session_ratios) / len(context_to_session_ratios), 4) if context_to_session_ratios else 0
        avg_context_to_edu_ratio = round(sum(context_to_edu_ratios) / len(context_to_edu_ratios), 4) if context_to_edu_ratios else 0
        
        # Calculate event extraction density per session
        sessions_with_events = sum([1 for chunk in all_openie_info if any(len(edu.event_triggers) > 0 if edu.event_triggers else False for edu in chunk['edus'])])
        sessions_with_role_args = sum([1 for chunk in all_openie_info if any(len(edu.event_role_argument_pairs) > 0 if edu.event_role_argument_pairs else False for edu in chunk['edus'])])
        
        # Calculate average event extraction per EDU
        edus_with_events = sum([1 for chunk in all_openie_info for edu in chunk['edus'] if edu.event_triggers and len(edu.event_triggers) > 0])
        edus_with_role_args = sum([1 for chunk in all_openie_info for edu in chunk['edus'] if edu.event_role_argument_pairs and len(edu.event_role_argument_pairs) > 0])
        
        # Calculate role-argument statistics
        all_roles = []
        all_arguments = []
        for chunk in all_openie_info:
            for edu in chunk['edus']:
                if edu.event_role_argument_pairs:
                    for role_arg_pair in edu.event_role_argument_pairs:
                        all_roles.append(role_arg_pair['role'])
                        all_arguments.append(role_arg_pair['argument'])
        
        # Calculate average argument length
        if total_role_args > 0:
            avg_arg_chars = round(sum([len(arg) for arg in all_arguments]) / total_role_args, 4)
            avg_arg_words = round(sum([len(arg.split()) for arg in all_arguments]) / total_role_args, 4)
        else:
            avg_arg_chars = 0
            avg_arg_words = 0

        if len(all_openie_info) > 0:
            # EDU coverage ratio
            if total_session_words > 0:
                edu_coverage_ratio = round(total_edu_words / total_session_words, 4)
            else:
                edu_coverage_ratio = 0
                
            # Event extraction density
            if total_edus > 0:
                avg_events_per_edu = round(total_triggers / total_edus, 4)
                avg_role_args_per_edu = round(total_role_args / total_edus, 4)
                event_edu_ratio = round(edus_with_events / total_edus, 4)
                role_arg_edu_ratio = round(edus_with_role_args / total_edus, 4)
            else:
                avg_events_per_edu = 0
                avg_role_args_per_edu = 0
                event_edu_ratio = 0
                role_arg_edu_ratio = 0
                
            # Session-level event extraction coverage
            if len(all_openie_info) > 0:
                session_event_coverage = round(sessions_with_events / len(all_openie_info), 4)
                session_role_arg_coverage = round(sessions_with_role_args / len(all_openie_info), 4)
            else:
                session_event_coverage = 0
                session_role_arg_coverage = 0
                
            structuring_dict = {
                'sessions': all_openie_info,
                # EDU coverage statistics
                'edu_coverage_ratio': edu_coverage_ratio,
                'total_session_words': total_session_words,
                'total_edu_words': total_edu_words,
                # Event extraction statistics
                'total_edus': total_edus,
                'total_triggers': total_triggers,
                'total_role_args': total_role_args,
                # Event extraction density
                'avg_events_per_edu': avg_events_per_edu,
                'avg_role_args_per_edu': avg_role_args_per_edu,
                'event_edu_ratio': event_edu_ratio,
                'role_arg_edu_ratio': role_arg_edu_ratio,
                # Session-level coverage
                'session_event_coverage': session_event_coverage,
                'session_role_arg_coverage': session_role_arg_coverage,
                # Role-argument quality
                'avg_arg_chars': avg_arg_chars,
                'avg_arg_words': avg_arg_words,
                'unique_roles': len(set(all_roles)) if all_roles else 0,
                # Contextual EDU statistics
                'total_context_texts': total_context_texts,
                'total_context_words': total_context_words,
                'avg_context_words': avg_context_words,
                'avg_context_to_session_ratio': avg_context_to_session_ratio,
                'avg_context_to_edu_ratio': avg_context_to_edu_ratio,
            }
            
            with open(self.openie_results_path, 'wb') as f:
                pickle.dump(structuring_dict, f)
            logger.info(f"Conversation structuring results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes_structuring()
        self.add_new_edges_structuring()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())


    def add_new_nodes_structuring(self):
        """
        Adds new nodes to the graph from EDU, argument, and passage embedding stores for structuring mode.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the EDU, argument, and passage embedding stores.
        The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        
        For Session nodes, adds metadata containing summary and parsed date information.
        For EDU and Argument nodes, adds empty metadata dict.
        """
        from datetime import datetime

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        edu_to_row = self.edu_embedding_store.get_all_id_to_rows()
        argument_to_row = self.argument_embedding_store.get_all_id_to_rows()
        session_to_row = self.session_content_store.get_all_id_to_rows()

        new_nodes = {}
        
        # Add Session nodes with type "Session" and metadata containing summary and parsed date
        for node_id, node in session_to_row.items():
            node['name'] = node_id
            node['type'] = 'Session'
            
            # Add metadata with summary and parsed date
            session_data = node['content']  # This is the Session object
            
            # Get session summary from session_id_to_summary if available
            session_summary = self.session_id_to_summary[node_id]
            
            # Parse the date string to datetime object using centralized utility
            from .utils.date_utils import parse_date_string
            date_parsed = None
            if hasattr(session_data, 'date_time') and session_data.date_time:
                date_parsed = parse_date_string(session_data.date_time)
                if not date_parsed:
                    logger.warning(f"Failed to parse date '{session_data.date_time}' for session {node_id}")
            
            node['metadata'] = {
                'summary': session_summary,
                'date': date_parsed
            }
            
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        # Add EDU nodes with type "EDU" and empty metadata
        for node_id, node in edu_to_row.items():
            node['name'] = node_id
            node['type'] = 'EDU'
            node['metadata'] = {}  # Empty metadata for EDU nodes
            
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        # Add Argument nodes with type "Argument" and empty metadata
        for node_id, node in argument_to_row.items():
            node['name'] = node_id
            node['type'] = 'Argument'
            node['metadata'] = {}  # Empty metadata for Argument nodes
            
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges_structuring(self):
        """
        Processes edges from `node_id_to_node_id_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases for structuring mode.
        
        This method creates edges in the igraph with both weight and components attributes. The components
        attribute contains detailed metadata about each edge including type (EDU-Arg, EDU-Passage_context, 
        Arg-Arg_synonymy), content (role names, context text, or None), individual weights, and directional
        information collected during the graph construction process.
        
        Side Effects:
            - Adds validated edges to self.graph with weight and components attributes
            - Logs warnings for invalid edges that cannot be added
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        
        # Process structuring mode edges (node_id_to_node_id_stats)
        for edge, weight in self.node_id_to_node_id_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": [], "components": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
                
                # Add components information
                edge_key = (source_node_id, target_node_id)
                components = self.edge_components.get(edge_key, [])
                valid_weights["components"].append(components)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # Structuring mode statistics
        edu_nodes_keys = self.edu_embedding_store.get_all_ids()
        graph_info["num_edu_nodes"] = len(set(edu_nodes_keys))
        
        argument_nodes_keys = self.argument_embedding_store.get_all_ids()
        graph_info["num_argument_nodes"] = len(set(argument_nodes_keys))
        
        session_nodes_keys = self.session_content_store.get_all_ids()
        graph_info["num_session_nodes"] = len(set(session_nodes_keys))
        
        graph_info["num_total_nodes"] = graph_info["num_edu_nodes"] + graph_info["num_argument_nodes"] + graph_info["num_session_nodes"]
        
        # Count different types of edges in structuring mode
        num_role_arg_edges = 0
        num_context_edges = 0
        num_synonymy_edges = 0
        session_nodes_set = set(session_nodes_keys)
        edu_nodes_set = set(edu_nodes_keys)
        argument_nodes_set = set(argument_nodes_keys)
        
        for node_pair in self.node_id_to_node_id_stats:
            if node_pair[0] in session_nodes_set or node_pair[1] in session_nodes_set:
                num_context_edges += 1
            elif (node_pair[0] in edu_nodes_set and node_pair[1] in argument_nodes_set) or \
                    (node_pair[0] in argument_nodes_set and node_pair[1] in edu_nodes_set):
                num_role_arg_edges += 1
            elif node_pair[0] in argument_nodes_set and node_pair[1] in argument_nodes_set:
                num_synonymy_edges += 1
        
        graph_info['num_role_arg_edges'] = num_role_arg_edges
        graph_info['num_context_edges'] = num_context_edges
        graph_info['num_synonymy_edges'] = num_synonymy_edges
        graph_info["num_total_edges"] = len(self.node_id_to_node_id_stats)


        return graph_info

    def dense_session_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            logger.info("Query embedding is None. Encoding the query before running dense session retrieval.")
            query_embedding = self.embedding_model.batch_encode(
                query,
                # instruction="query" if "e5" in self.global_config.embedding_model_name.lower() else get_query_instruction('query_to_passage'),
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )

        query_doc_scores = np.dot(self.session_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_arguments_weights(self,
                          link_top_k: int,
                          all_argument_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_argument_weights to retain only the weights for the
        top-ranked arguments in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected arguments in argument
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_argument_weights (np.ndarray): An array representing the argument weights, indexed
                by argument ID.
            linking_score_map (Dict[str, float]): A mapping of argument content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_argument_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` arguments.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k arguments in all_argument_weights
        top_k_arguments = set(linking_score_map.keys())
        top_k_arguments_keys = set(
            [compute_mdhash_id(content=top_k_argument, prefix="argument-") for top_k_argument in top_k_arguments])

        for argument_key in self.node_name_to_vertex_idx:
            if argument_key not in top_k_arguments_keys:
                argument_id = self.node_name_to_vertex_idx.get(argument_key, None)
                if argument_id is not None:
                    all_argument_weights[argument_id] = 0.0

        assert np.count_nonzero(all_argument_weights) == len(linking_score_map.keys())
        return all_argument_weights, linking_score_map

    
    def rerank_filter_structuring_v1(self,
                                     query: str,
                                     candidate_items: List[str],
                                     candidate_indices: List[int],
                                     len_after_rerank: int = None) -> Tuple[List[int], List[str], dict]:
        """
        Rerank and filter candidate EDUs using an LLM-based approach with structured outputs.
        This version is more inclusive and better suited for conversational memory retrieval.
        
        Args:
            query: The user's query string
            candidate_items: List of candidate EDU strings to filter
            candidate_indices: List of indices corresponding to candidate_items
            len_after_rerank: Maximum number of EDUs to return (optional, uses all selected if None)
            
        Returns:
            Tuple of (filtered_indices, filtered_items, metadata_dict)
            - filtered_indices: List of indices of selected EDUs
            - filtered_items: List of selected EDU strings
            - metadata_dict: Dictionary containing confidence and other metadata
        """
        from .prompts.templates.edu_filter_v1 import FilteredEDUs, prompt_template
        from .prompts import PromptTemplateManager
        from .utils.llm_utils import fix_broken_generated_json
        from .utils.config_utils import get_support_json_schema
        import difflib
        import json
        
        if len(candidate_items) == 0:
            return [], [], {'confidence': None, 'num_candidates': 0, 'num_selected': 0}
        
        try:
            # Get support for JSON schema
            support_json_schema = get_support_json_schema()
            
            # Initialize prompt template manager
            prompt_template_manager = PromptTemplateManager(
                role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
            )
            
            # # Register the template
            # prompt_template_manager.add_template(name='edu_filter_v1', template=prompt_template)
            
            # Render the prompt with query and candidate EDUs (use json.dumps for consistent formatting)
            if self.global_config.date_format_type == "longmemeval":
                messages = prompt_template_manager.render(
                    name='edu_filter_v1',
                    query=query,
                    candidate_edus=json.dumps(candidate_items, indent=2)
                )
            else:
                messages = prompt_template_manager.render(
                    name='edu_filter_locomo_v1',
                    query=query,
                    candidate_edus=json.dumps(candidate_items, indent=2)
                )

            # Call LLM with structured output if supported
            raw_response = ""
            metadata = {}
            
            if support_json_schema:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages,
                    response_format=FilteredEDUs
                )
            else:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages
                )
            
            metadata['cache_hit'] = cache_hit
            
            # Parse response
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            
            # Extract selected EDUs
            filtered_result = FilteredEDUs.model_validate_json(real_response)
            selected_edus = filtered_result.selected_edus
            
            # Match selected EDUs back to candidate items using fuzzy matching
            result_indices = []
            matched_edus = []
            
            for selected_edu in selected_edus:
                # Find the closest match among candidate items
                closest_matches = difflib.get_close_matches(
                    str(selected_edu), 
                    [str(item) for item in candidate_items],
                    n=1,
                    cutoff=0.85  # High threshold for accuracy
                )
                
                if closest_matches:
                    try:
                        matched_idx = candidate_items.index(closest_matches[0])
                        if matched_idx not in result_indices:  # Avoid duplicates
                            result_indices.append(matched_idx)
                            matched_edus.append(candidate_items[matched_idx])
                    except ValueError:
                        logger.warning(f"Could not find matched EDU in candidate list: {closest_matches[0]}")
            
            # Convert to actual indices from the candidate_indices list
            filtered_indices = [candidate_indices[i] for i in result_indices if i < len(candidate_indices)]
            filtered_items = matched_edus
            
            # Apply length limit if specified
            if len_after_rerank is not None:
                filtered_indices = filtered_indices[:len_after_rerank]
                filtered_items = filtered_items[:len_after_rerank]
            
            result_metadata = {
                'confidence': None,
                'num_candidates': len(candidate_items),
                'num_selected': len(filtered_items),
                'cache_hit': cache_hit,
                'llm_metadata': metadata
            }
            
            return filtered_indices, filtered_items, result_metadata
            
        except Exception as e:
            logger.error(f"Error in rerank_filter_structuring_v1: {str(e)}")
            # Fallback: return all candidates up to len_after_rerank
            if len_after_rerank is not None:
                fallback_indices = candidate_indices[:len_after_rerank]
                fallback_items = candidate_items[:len_after_rerank]
            else:
                fallback_indices = candidate_indices
                fallback_items = candidate_items
            
            return fallback_indices, fallback_items, {
                'confidence': None,
                'error': str(e),
                'num_candidates': len(candidate_items),
                'num_selected': len(fallback_items)
            }
    
    def rerank_filter_arguments_v1(self,
                                   query: str,
                                   candidate_arguments: List[str],
                                   candidate_arg_keys: List[str],
                                   candidate_arg_scores: List[float],
                                   len_after_rerank: int = None) -> Tuple[List[str], List[str], List[float], dict]:
        """
        Rerank and filter candidate argument nodes using an LLM-based approach with structured outputs.
        This method helps identify semantically relevant argument nodes that can lead to relevant EDUs and sessions.
        
        Args:
            query: The user's query string
            candidate_arguments: List of candidate argument strings (content) to filter
            candidate_arg_keys: List of argument node keys corresponding to candidate_arguments
            candidate_arg_scores: List of similarity scores corresponding to candidate_arguments
            len_after_rerank: Maximum number of arguments to return (optional, uses all selected if None)
            
        Returns:
            Tuple of (filtered_arg_keys, filtered_arguments, filtered_scores, metadata_dict)
            - filtered_arg_keys: List of argument node keys of selected arguments
            - filtered_arguments: List of selected argument strings
            - filtered_scores: List of scores corresponding to selected arguments
            - metadata_dict: Dictionary containing confidence and other metadata
        """
        from .prompts.templates.argument_filter_v1 import FilteredArguments, prompt_template
        from .prompts import PromptTemplateManager
        from .utils.llm_utils import fix_broken_generated_json
        from .utils.config_utils import get_support_json_schema
        import difflib
        import json
        
        if len(candidate_arguments) == 0:
            return [], [], [], {'confidence': None, 'num_candidates': 0, 'num_selected': 0}
        
        try:
            # Get support for JSON schema
            support_json_schema = get_support_json_schema()
            
            # Initialize prompt template manager
            prompt_template_manager = PromptTemplateManager(
                role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
            )
            
            # Render the prompt with query and candidate arguments
            if self.global_config.date_format_type == "longmemeval":
                messages = prompt_template_manager.render(
                    name='argument_filter_v1',
                    query=query,
                    candidate_arguments=json.dumps(candidate_arguments, indent=2)
                )
            else:
                messages = prompt_template_manager.render(
                    name='argument_filter_locomo_v1',
                    query=query,
                    candidate_arguments=json.dumps(candidate_arguments, indent=2)
                )

            # Call LLM with structured output if supported
            raw_response = ""
            metadata = {}
            
            if support_json_schema:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages,
                    response_format=FilteredArguments
                )
            else:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages
                )
            
            metadata['cache_hit'] = cache_hit
            
            # Parse response
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            
            # Extract selected arguments
            filtered_result = FilteredArguments.model_validate_json(real_response)
            selected_arguments = filtered_result.selected_arguments
            
            # Match selected arguments back to candidate items using fuzzy matching
            result_arg_keys = []
            result_arguments = []
            result_scores = []
            
            for selected_arg in selected_arguments:
                # Find the closest match among candidate arguments
                closest_matches = difflib.get_close_matches(
                    str(selected_arg), 
                    [str(arg) for arg in candidate_arguments],
                    n=1,
                    cutoff=0.85  # High threshold for accuracy
                )
                
                if closest_matches:
                    try:
                        matched_idx = candidate_arguments.index(closest_matches[0])
                        if matched_idx not in [candidate_arguments.index(arg) for arg in result_arguments]:  # Avoid duplicates
                            result_arg_keys.append(candidate_arg_keys[matched_idx])
                            result_arguments.append(candidate_arguments[matched_idx])
                            result_scores.append(candidate_arg_scores[matched_idx])
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not find matched argument in candidate list: {closest_matches[0]}, error: {e}")
            
            # Apply length limit if specified
            if len_after_rerank is not None:
                result_arg_keys = result_arg_keys[:len_after_rerank]
                result_arguments = result_arguments[:len_after_rerank]
                result_scores = result_scores[:len_after_rerank]
            
            result_metadata = {
                'confidence': None,
                'num_candidates': len(candidate_arguments),
                'num_selected': len(result_arguments),
                'cache_hit': cache_hit,
                'llm_metadata': metadata
            }
            
            return result_arg_keys, result_arguments, result_scores, result_metadata
            
        except Exception as e:
            logger.error(f"Error in rerank_filter_arguments_v1: {str(e)}")
            # Fallback: return all candidates up to len_after_rerank
            if len_after_rerank is not None:
                fallback_keys = candidate_arg_keys[:len_after_rerank]
                fallback_arguments = candidate_arguments[:len_after_rerank]
                fallback_scores = candidate_arg_scores[:len_after_rerank]
            else:
                fallback_keys = candidate_arg_keys
                fallback_arguments = candidate_arguments
                fallback_scores = candidate_arg_scores
            
            return fallback_keys, fallback_arguments, fallback_scores, {
                'confidence': None,
                'error': str(e),
                'num_candidates': len(candidate_arguments),
                'num_selected': len(fallback_arguments)
            }

    def run_ppr_structuring(self,
                reset_prob: np.ndarray,
                damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph in structuring mode and computes relevance scores for
        nodes corresponding to both conversation sessions and EDUs. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four numpy arrays:
                - sorted_session_ids: The sorted node IDs of conversation sessions based
                  on their relevance scores in descending order.
                - sorted_session_scores: The corresponding relevance scores of each session
                  in the same order.
                - sorted_edu_ids: The sorted node IDs of EDUs based on their relevance 
                  scores in descending order.
                - sorted_edu_scores: The corresponding relevance scores of each EDU
                  in the same order.
        """

        if damping is None: 
            damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        # Extract session scores and sort them
        session_scores = np.array([pagerank_scores[idx] for idx in self.session_node_idxs])
        sorted_session_ids = np.argsort(session_scores)[::-1]
        sorted_session_scores = session_scores[sorted_session_ids.tolist()]

        # Extract EDU scores and sort them
        edu_scores = np.array([pagerank_scores[idx] for idx in self.edu_node_idxs])
        sorted_edu_ids = np.argsort(edu_scores)[::-1]
        sorted_edu_scores = edu_scores[sorted_edu_ids.tolist()]

        return sorted_session_ids, sorted_session_scores, sorted_edu_ids, sorted_edu_scores

    def qa_conversation(self, queries: List[QuerySolution], save_qa_traces: bool = True) -> Tuple[List[QuerySolution], List[str], List[Dict], List[Dict]]:
        """
        Executes conversation-based question-answering (QA) inference using retrieved conversation sessions and EDUs.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved sessions, EDUs, and other related information.
            save_qa_traces: bool
                Whether to save detailed QA traces for analysis. Default is True.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
                - A list of QA trace dictionaries for each query (if save_qa_traces is True).
        """
        # Import response model from prompt template
        from .prompts.templates.conversation_qa import ConversationQAResponse, ConversationQAThoughtResponse
        
        # Initialize QA tracing
        all_qa_traces = [] if save_qa_traces else None
        
        # Running inference for conversation QA using batch API
        all_qa_messages = []

        # Calculate conversation date span for all queries
        date_span_info = ""
        if hasattr(self, 'date_to_session_id') and self.date_to_session_id:
            dates = sorted(self.date_to_session_id.keys())
            if dates:
                earliest_date = dates[0].strftime("%d %B, %Y")
                latest_date = dates[-1].strftime("%d %B, %Y")
                if earliest_date == latest_date:
                    date_span_info = f" spanning {earliest_date}"
                else:
                    date_span_info = f" spanning from {earliest_date} to {latest_date}"
        
        # Get speaker information for all queries
        speaker_info = ""
        if hasattr(self, 'speaker_a') and hasattr(self, 'speaker_b') and self.speaker_a and self.speaker_b:
            speaker_info = f" between {self.speaker_a} and {self.speaker_b}"
        
        for query_idx, query_solution in tqdm(enumerate(queries), desc="Preparing conversation QA prompts"):
            qa_trace = {} if save_qa_traces else None
            
            # Obtain the retrieved EDUs (primary context) and sessions (for reference)
            retrieved_edus = query_solution.edus[:self.global_config.qa_top_k] if query_solution.edus else []
            retrieved_sessions = query_solution.docs[:self.global_config.qa_top_k] if query_solution.docs else []
            
            # Log input information for tracing
            if save_qa_traces:
                qa_trace.update({
                    "query_idx": query_idx,
                    "question": query_solution.question,
                    "category": getattr(query_solution, 'category', None),
                    "expected_answer": getattr(query_solution, 'expected_answer', None),
                    "evaluation_answer": getattr(query_solution, 'evaluation_answer', None),
                    "num_retrieved_edus": len(retrieved_edus),
                    "num_retrieved_sessions": len(retrieved_sessions),
                    "qa_top_k": self.global_config.qa_top_k,
                    "date_span_info": date_span_info,
                    "speaker_info": speaker_info
                })
            
            # Format EDU-based context with speaker and temporal information
            context_parts = []
            
            if retrieved_edus:
                # Primary path: Use EDUs as context with enriched metadata
                for edu_idx, edu in enumerate(retrieved_edus):
                    # Use enriched EDU with metadata
                    context_string = edu.to_context_string(self.global_config.date_format_type)
                    context_parts.append(f"Memory {edu_idx + 1}: {context_string}")
                    
                    # Log individual EDU information for tracing
                    if save_qa_traces:
                        if "retrieved_edus" not in qa_trace:
                            qa_trace["retrieved_edus"] = []
                        qa_trace["retrieved_edus"].append({
                            "edu_idx": edu_idx,
                            "edu_text": edu.edu_text if hasattr(edu, 'edu_text') else str(edu),
                            "context_string": context_string,
                            "edu_score": query_solution.edu_scores[edu_idx] if (query_solution.edu_scores is not None and edu_idx < len
                            (query_solution.edu_scores)) else None,
                            "source_speakers": getattr(edu, 'source_speakers', []),
                            "date": getattr(edu, 'date', None),
                            "session_id": getattr(edu, 'session_id', None)
                        })

                context = '\n'.join(context_parts)
            else:
                # raise ValueError("No EDUs provided for QA")
                context = ""
                logger.warning(f"No EDUs provided for QA for query {query_idx}")
            
            # Log context information for tracing
            if save_qa_traces:
                qa_trace.update({
                    "formatted_context": context,
                    "context_length": len(context),
                    "num_context_parts": len(context_parts)
                })
            
            # Create QA message using prompt template manager
            qa_message = self.prompt_template_manager.render(
                name="conversation_qa",
                context=context,
                question=query_solution.question,
                date_span_info=date_span_info,
                speaker_info=speaker_info
            )
            
            # Log prompt information for tracing
            if save_qa_traces:
                qa_trace.update({
                    "input_message": qa_message,
                    "prompt_template": "conversation_qa",
                    "message_length": len(str(qa_message))
                })
            
            all_qa_messages.append(qa_message)
            
            if save_qa_traces:
                all_qa_traces.append(qa_trace)

        # Get structured responses from LLM using batch processing
        logger.info(f"Running batch conversation QA inference for {len(all_qa_messages)} queries...")
        
        qa_start_time = time.time()
        
        # Check if JSON schema is supported for response format
        try:
            from .utils.config_utils import get_support_json_schema
            support_json_schema = get_support_json_schema()
        except:
            support_json_schema = False
            
        if support_json_schema:
            batch_results = self.llm_model.batch_infer(
                messages=all_qa_messages,
                # response_format=ConversationQAResponse
                response_format=ConversationQAThoughtResponse
            )
        else:
            batch_results = self.llm_model.batch_infer(messages=all_qa_messages)
        
        qa_end_time = time.time()
        qa_inference_time = qa_end_time - qa_start_time
        
        logger.info(f"QA inference completed in {qa_inference_time:.2f}s")
        
        # Process batch results
        all_response_message = []
        all_metadata = []
        all_answers = []
        
        for batch_idx, batch_result in enumerate(batch_results):
            try:
                raw_response, metadata, cache_hit = batch_result
                metadata['cache_hit'] = cache_hit
                
                # Parse the string response into Pydantic object
                parse_error = None
                try:
                    # parsed_response = ConversationQAResponse.model_validate_json(raw_response)
                    parsed_response = ConversationQAThoughtResponse.model_validate_json(raw_response)
                    answer = parsed_response.final_answer
                except Exception as e:
                    parse_error = str(e)
                    logger.warning(f"Failed to parse response: {e}. Raw: {raw_response}")
                    answer = str(raw_response)
                
                # Log response information for tracing
                if save_qa_traces and batch_idx < len(all_qa_traces):
                    all_qa_traces[batch_idx].update({
                        "raw_response": raw_response,
                        "parsed_final_answer": answer,
                        "response_metadata": metadata,
                        "parse_error": parse_error,
                        "qa_inference_time": qa_inference_time / len(batch_results),  # Average time per query
                        "support_json_schema": support_json_schema
                    })
                
                all_response_message.append(raw_response)
                all_metadata.append(metadata)
                all_answers.append(answer)
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Error processing batch result {batch_idx}: {error_msg}")
                
                # Log error information for tracing
                if save_qa_traces and batch_idx < len(all_qa_traces):
                    all_qa_traces[batch_idx].update({
                        "raw_response": "",
                        "parsed_final_answer": "",
                        "response_metadata": {'error': error_msg},
                        "parse_error": error_msg,
                        "qa_inference_time": qa_inference_time / len(batch_results),
                        "support_json_schema": support_json_schema
                    })
                
                all_response_message.append("")
                all_metadata.append({'error': error_msg})
                all_answers.append("")

        # Process responses and extract predicted answers
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extracting Answers from LLM Response"):
            # Use the pre-parsed answer from batch processing
            pred_ans = all_answers[query_solution_idx]
            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)
            
            # Log final answer information for tracing
            if save_qa_traces and query_solution_idx < len(all_qa_traces):
                all_qa_traces[query_solution_idx].update({
                    "final_predicted_answer": pred_ans
                })

        # Save QA traces if requested
        if save_qa_traces:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            trace_filename = f"qa_conversation_traces_{timestamp}.pkl"
            trace_filepath = os.path.join(self.working_dir, trace_filename)
            try:
                with open(trace_filepath, 'wb') as f:
                    pickle.dump(all_qa_traces, f)
                logger.info(f"QA traces saved to {trace_filepath}")
            except Exception as e:
                logger.warning(f"Failed to save QA traces: {str(e)}")

        return queries_solutions, all_response_message, all_metadata, all_qa_traces or []
    
    def rag_qa_conversation(self, 
                           queries: Union[List[str], List[QA]], 
                           gold_docs: List[List[str]] = None,
                           gold_answers: List[List[str]] = None) -> Union[Tuple[List[QuerySolution], List[str], List[Dict], List[Dict]], Tuple[List[QuerySolution], List[str], List[Dict], List[Dict], Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using conversation sessions for the LoCoMo dataset.

        This method handles both string-based queries and QA objects from LoCoMo dataset. It performs
        retrieval of relevant conversation sessions and EDUs, then generates answers using the
        conversation-specific QA pipeline.

        Parameters:
            queries (Union[List[str], List[QA]]): A list of queries, which can be either strings or
                QA objects from LoCoMo dataset.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard sessions for
                each query. This is used if session-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], List[Dict], Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                - List of query retrieval trace dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).
        """
        
        # Set up evaluators if needed
        if gold_answers is not None:
            from .evaluation.qa_eval import QAExactMatch, QAF1Score
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Convert QA objects to query strings if needed
        query_strings = []
        qa_categories = []
        final_answers = []
        
        for query in queries:
            if isinstance(query, QA):
                query_strings.append(query.question)
                qa_categories.append(query.category)
                final_answers.append(query.final_answer)
            elif isinstance(query, str):
                query_strings.append(query)
                qa_categories.append(None)
                final_answers.append(None)
            else:
                query_strings.append(str(query))
                qa_categories.append(None)
                final_answers.append(None)

        # Retrieve relevant sessions and EDUs
        logger.info("Retrieving relevant conversation sessions and EDUs...")
        if self.global_config.date_format_type == "longmemeval":
            query_strings_for_retrieval = [q.split('\n')[1] for q in query_strings]
            retrieval_results, query_retrieval_traces = self.retrieve_structuring_conversation_enhanced_v2(queries=query_strings_for_retrieval, gold_docs=gold_docs, no_query_trace_saving=True)
            assert len(retrieval_results) == len(query_strings)
            for i, query_solution in enumerate(retrieval_results):
                query_solution.question = query_strings[i]
            for i, query_solution in enumerate(retrieval_results):
                assert '\n' in query_solution.question
        else:
            retrieval_results, query_retrieval_traces = self.retrieve_structuring_conversation_enhanced_v2(queries=query_strings, gold_docs=gold_docs, no_query_trace_saving=True)

        
        # Enhance QuerySolution objects with category information for better QA
        enhanced_queries = []
        for i, query_solution in enumerate(retrieval_results):
            # Add category information if available
            if i < len(qa_categories) and qa_categories[i] is not None:
                query_solution.category = qa_categories[i]
            if i < len(final_answers) and final_answers[i] is not None:
                query_solution.expected_answer = final_answers[i]
                
                # For adversarial questions (category 5), the correct evaluation answer is "Not mentioned in the conversation"
                # The final_answer contains the adversarial_answer which is a wrong tempting answer
                if qa_categories[i] == 5:
                    query_solution.evaluation_answer = "Not mentioned in the conversation"
                else:
                    query_solution.evaluation_answer = final_answers[i]
            enhanced_queries.append(query_solution)

        # Perform conversation-specific QA
        logger.info("Performing conversation QA...")
        queries_solutions, all_response_message, all_metadata, all_qa_traces = self.qa_conversation(enhanced_queries, save_qa_traces=True)
        
        # Evaluate QA if gold answers provided
        if gold_answers is not None:
            # Prepare gold answers, handling QA objects vs strings and adversarial questions
            if isinstance(queries[0], QA):
                eval_gold_answers = []
                for qa in queries:
                    if qa.category == 5:  # Adversarial questions
                        eval_gold_answers.append(["Not mentioned in the conversation"])
                    elif qa.final_answer:
                        eval_gold_answers.append([qa.final_answer])
                    else:
                        eval_gold_answers.append([""])
            else:
                eval_gold_answers = gold_answers
                
            # Calculate evaluation metrics
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=eval_gold_answers, 
                predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max
            )
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=eval_gold_answers, 
                predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max
            )

            # Combine and round off evaluation results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for conversation QA: {overall_qa_results}")

            # Save evaluation results to QuerySolution objects
            for idx, q in enumerate(queries_solutions):
                if idx < len(eval_gold_answers):
                    q.gold_answers = eval_gold_answers[idx]
                if gold_docs is not None and idx < len(gold_docs):
                    q.gold_docs = gold_docs[idx]

            # Add instance-level evaluation scores to QA traces
            if all_qa_traces:
                for idx, qa_trace in enumerate(all_qa_traces):
                    if idx < len(example_qa_em_results) and idx < len(example_qa_f1_results):
                        qa_trace.update({
                            "gold_answers": eval_gold_answers[idx] if idx < len(eval_gold_answers) else [],
                            "evaluation_scores": {
                                "exact_match": example_qa_em_results[idx],
                                "f1_score": example_qa_f1_results[idx]
                            },
                            "overall_qa_results": overall_qa_results
                        })
                
                # Save updated QA traces with evaluation scores
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
                trace_filename = f"qa_conversation_traces_with_eval_{timestamp}.pkl"
                trace_filepath = os.path.join(self.working_dir, trace_filename)
                try:
                    with open(trace_filepath, 'wb') as f:
                        pickle.dump(all_qa_traces, f)
                    logger.info(f"QA traces with evaluation scores saved to {trace_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to save QA traces with evaluation: {str(e)}")

            return queries_solutions, all_response_message, all_metadata, query_retrieval_traces, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata, query_retrieval_traces

    def retrieve_structuring_conversation_enhanced_v2(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None,
                 no_query_trace_saving: bool = False) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict] | Tuple[List[QuerySolution], List[Dict]]:
        retrieve_start_time = time.time()  # Record start time
        
        # ========== HELPER FUNCTION: Balance EDUs by speaker (Can be commented out for ablation) ==========
        def balance_edus_by_speaker(edu_indices, edu_scores, edu_node_keys, edu_node_idx_to_speaker, 
                                   node_name_to_vertex_idx, target_count):
            """
            Balance EDU indices to have equal representation from each speaker while maintaining score order.
            
            Args:
                edu_indices: List of EDU indices in the edu_node_keys list (not graph indices)
                edu_scores: Corresponding scores for EDUs
                edu_node_keys: List mapping EDU indices to node keys
                edu_node_idx_to_speaker: Mapping from graph node idx to speaker
                node_name_to_vertex_idx: Mapping from node key to graph idx
                target_count: Target number of EDUs to return
                
            Returns:
                Balanced list of EDU indices
            """
            # Categorize EDUs by speaker
            speaker_edus = {}  # speaker -> list of (edu_idx, score)
            for edu_idx in edu_indices:
                edu_node_key = edu_node_keys[edu_idx]
                graph_idx = node_name_to_vertex_idx.get(edu_node_key)
                if graph_idx is not None:
                    speaker = edu_node_idx_to_speaker.get(graph_idx, None)
                    if speaker is not None:
                        score = edu_scores[edu_idx]
                        if speaker not in speaker_edus:
                            speaker_edus[speaker] = []
                        speaker_edus[speaker].append((edu_idx, score))
            
            if len(speaker_edus) < 2:
                # Not enough speakers to balance, return original
                return edu_indices[:target_count]
            
            # Sort each speaker's EDUs by score (descending)
            for speaker in speaker_edus:
                speaker_edus[speaker] = sorted(speaker_edus[speaker], key=lambda x: x[1], reverse=True)
            
            # Get speakers
            speakers = list(speaker_edus.keys())
            half_count = target_count // 2
            
            # Take top half_count from each speaker
            balanced_edus = []
            for speaker in speakers:
                balanced_edus.extend([edu_idx for edu_idx, _ in speaker_edus[speaker][:half_count]])
            
            # If we need more EDUs to reach target_count (odd number case), take from highest scoring remaining
            if len(balanced_edus) < target_count:
                all_remaining = []
                for speaker in speakers:
                    all_remaining.extend(speaker_edus[speaker][half_count:])
                all_remaining = sorted(all_remaining, key=lambda x: x[1], reverse=True)
                for edu_idx, _ in all_remaining[:target_count - len(balanced_edus)]:
                    balanced_edus.append(edu_idx)
            
            # Sort balanced EDUs by their scores to maintain score order
            balanced_with_scores = [(edu_idx, edu_scores[edu_idx]) for edu_idx in balanced_edus]
            balanced_with_scores = sorted(balanced_with_scores, key=lambda x: x[1], reverse=True)
            
            return [edu_idx for edu_idx, _ in balanced_with_scores]
        
        def balance_ppr_edus_by_speaker(sorted_edu_ids, sorted_edu_scores, edu_node_keys, edu_node_idxs, 
                                       edu_node_idx_to_speaker, target_count):
            """
            Balance PPR-sorted EDU IDs to have equal representation from each speaker while maintaining PPR score order.
            
            Strategy: Iterate through sorted EDUs and collect top-scoring ones from each speaker until we have
            target_count EDUs with equal (or near-equal) representation from each speaker.
            
            Args:
                sorted_edu_ids: Array of edu indices sorted by PPR score (indices in edu_node_idxs)
                sorted_edu_scores: Corresponding PPR scores
                edu_node_keys: List mapping edu array index to node keys
                edu_node_idxs: List of graph node indices for all EDUs
                edu_node_idx_to_speaker: Mapping from graph node idx to speaker
                target_count: Target number of EDUs to return
                
            Returns:
                Tuple of (balanced_edu_ids, balanced_edu_scores)
            """
            # Collect EDUs by speaker while iterating through sorted order
            speaker_edus = {}  # speaker -> list of (position, array_idx, ppr_score)
            speaker_counts = {}  # speaker -> count collected
            half_count = target_count // 2
            
            balanced_edus = []  # list of (position, array_idx, ppr_score, speaker)
            
            # Iterate through sorted EDUs and collect balanced set
            for i, edu_idx in enumerate(sorted_edu_ids):
                if len(balanced_edus) >= target_count:
                    break
                    
                graph_idx = edu_node_idxs[edu_idx]
                speaker = edu_node_idx_to_speaker.get(graph_idx, None)
                
                if speaker is not None:
                    # Initialize speaker tracking if needed
                    if speaker not in speaker_counts:
                        speaker_counts[speaker] = 0
                        speaker_edus[speaker] = []
                    
                    # Add this EDU if this speaker hasn't reached half_count yet
                    if speaker_counts[speaker] < half_count:
                        ppr_score = sorted_edu_scores[i]
                        balanced_edus.append((i, edu_idx, ppr_score, speaker))
                        speaker_counts[speaker] += 1
                        speaker_edus[speaker].append((i, edu_idx, ppr_score))
            
            # Handle case where we don't have enough from both speakers
            if len(balanced_edus) < target_count:
                # Add more from any speaker until we reach target_count
                for i, edu_idx in enumerate(sorted_edu_ids):
                    if len(balanced_edus) >= target_count:
                        break
                    graph_idx = edu_node_idxs[edu_idx]
                    speaker = edu_node_idx_to_speaker.get(graph_idx, None)
                    if speaker is not None:
                        ppr_score = sorted_edu_scores[i]
                        # Check if this EDU is already included
                        if not any(item[1] == edu_idx for item in balanced_edus):
                            balanced_edus.append((i, edu_idx, ppr_score, speaker))
            
            if len(balanced_edus) == 0:
                # Fallback: return original top-k
                return sorted_edu_ids[:target_count], sorted_edu_scores[:target_count]
            
            # Sort by original position (which corresponds to PPR score order) to maintain score order
            balanced_edus = sorted(balanced_edus, key=lambda x: x[0])
            
            balanced_edu_ids = np.array([item[1] for item in balanced_edus])
            balanced_edu_scores = np.array([item[2] for item in balanced_edus])
            
            return balanced_edu_ids, balanced_edu_scores
        # ========== END HELPER FUNCTIONS ==========

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.query_to_embedding: Dict = {'edu': {}, 'passage': {}}

            self.argument_node_keys: list = list(self.argument_embedding_store.get_all_ids()) # a list of argument node keys
            self.edu_node_keys: list = list(self.edu_embedding_store.get_all_ids()) # a list of edu node keys
            self.session_node_keys: list = list(self.session_content_store.get_all_ids()) 

            # Load existing openie info to get EDU enrichment data
            all_openie_info, _ = self.load_existing_openie([])
            self.structuring_results_dict = reformat_openie_results(all_openie_info, openie_mode=self.global_config.openie_mode)[0]

            expected_node_count = len(self.argument_node_keys) + len(self.edu_node_keys) + len(self.session_node_keys)
            actual_node_count = self.graph.vcount()
            if expected_node_count != actual_node_count:
                error_msg = f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}"
                logger.warning(error_msg)
                raise ValueError(error_msg)

            try:
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
                self.node_name_to_vertex_idx = igraph_name_to_idx

                # Check if all argument, edu, and session nodes are in the graph
                missing_argument_nodes = [node_key for node_key in self.argument_node_keys if node_key not in igraph_name_to_idx]
                missing_edu_nodes = [node_key for node_key in self.edu_node_keys if node_key not in igraph_name_to_idx]
                missing_session_nodes = [node_key for node_key in self.session_node_keys if node_key not in igraph_name_to_idx]

                if missing_argument_nodes or missing_edu_nodes or missing_session_nodes:
                    error_msg = f"Missing nodes in graph - Arguments: {len(missing_argument_nodes)}, EDUs: {len(missing_edu_nodes)}, Sessions: {len(missing_session_nodes)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                self.argument_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.argument_node_keys] # a list of backbone graph node index
                self.edu_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.edu_node_keys] # a list of backbone graph node index
                self.session_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.session_node_keys] # a list of backbone graph node index

            except Exception as e:
                logger.error(f"Error creating node index mapping: {str(e)}")
                raise e
            

            logger.info("Loading embeddings.")
            self.argument_embeddings = np.array(self.argument_embedding_store.get_embeddings(self.argument_node_keys))
            self.edu_embeddings = np.array(self.edu_embedding_store.get_embeddings(self.edu_node_keys))

            if self.global_config.query_to_session_retrieval:
                self.session_embeddings = self.embedding_model.batch_encode([self.session_id_to_summary[node_key] for node_key in self.session_node_keys], norm=True)

            # build mapping from edu source speakers to the edu node ids
            self.source_utterance_role_to_edu_node_ids = {}
            for session_hash_id in self.structuring_results_dict:
                for edu_result in self.structuring_results_dict[session_hash_id]:
                    edu_hash_id = edu_result.edu_id
                    edu_source_speakers = edu_result.source_speakers if edu_result.source_speakers else []
                    edu_source_speakers = self.speaker_a if self.speaker_a in edu_source_speakers else self.speaker_b
                    # edu_source_speakers = tuple(sorted(list(set([role for role in edu_source_speakers if role in [self.speaker_a, self.speaker_b]]))))

                    if edu_hash_id in self.node_name_to_vertex_idx:
                        if edu_source_speakers not in self.source_utterance_role_to_edu_node_ids:
                            self.source_utterance_role_to_edu_node_ids[edu_source_speakers] = []
                        self.source_utterance_role_to_edu_node_ids[edu_source_speakers].append(self.node_name_to_vertex_idx[edu_hash_id])
                    else:
                        logger.warning(f"EDU hash id ({edu_hash_id}) from session (hash id: {session_hash_id}) not exists in graph")
            for source_utterance_role, edu_node_ids in self.source_utterance_role_to_edu_node_ids.items():
                logger.info(f"Source role={source_utterance_role}: {len(edu_node_ids)} EDUs")
            
            # Build reverse mapping: from edu node idx to speaker
            self.edu_node_idx_to_speaker = {}
            for speaker, edu_node_idxs in self.source_utterance_role_to_edu_node_ids.items():
                for edu_node_idx in edu_node_idxs:
                    self.edu_node_idx_to_speaker[edu_node_idx] = speaker


            self.ready_to_retrieve = True

        # get_query_embeddings
        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (query.question not in self.query_to_embedding['edu'] or query.question not in self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['edu'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_edu.")
            query_embeddings_for_edu = self.embedding_model.batch_encode(
                all_query_strings,
                # instruction="query" if "e5" in self.global_config.embedding_model_name.lower() else get_query_instruction('query_to_sentence'),
                instruction=get_query_instruction('query_to_sentence'),
                norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_edu):
                self.query_to_embedding['edu'][query] = embedding
            
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings,
                # instruction="query" if "e5" in self.global_config.embedding_model_name.lower() else get_query_instruction('query_to_passage'),
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding
        
        # Extract entities from queries using batch processing
        logger.info(f"Extracting entities from {len(queries)} queries.")
        query_ner_results = self.openie.batch_query_er(queries)
        # query_ner_results = self.openie.batch_query_ner(queries)
        query_to_entities = {result.query: result.unique_entities for result in query_ner_results}
        # query_to_entities = {q: [] for q in queries} # for ablation

        # Precompute entity embeddings for similarity matching
        all_unique_entities = list(set([entity for entities in query_to_entities.values() for entity in entities]))
        if len(all_unique_entities) > 0:
            logger.info(f"Encoding {len(all_unique_entities)} unique query entities for similarity matching.")
            entity_embeddings = self.embedding_model.batch_encode(
                all_unique_entities,
                # instruction=get_query_instruction('query_to_sentence'),
                norm=True
            )
            entity_to_embedding = {entity: embedding for entity, embedding in zip(all_unique_entities, entity_embeddings)}
        else:
            # raise
            logger.warning("No unique entities found, using empty entity_to_embedding.")
            entity_to_embedding = {}


        retrieval_results = []

        # Log the trace
        all_query_traces = [] # a list of {} corresponding to each input query

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            query_trace = {}
            rerank_start = time.time()

            # get edu scores
            query_embedding = self.query_to_embedding['edu'].get(query, None)
            if query_embedding is None:
                query_embedding = self.embedding_model.batch_encode(
                    query, 
                    # instruction="query" if "e5" in self.global_config.embedding_model_name.lower() else get_query_instruction('query_to_sentence'), 
                    instruction=get_query_instruction('query_to_sentence'), 
                    norm=True
                )

            if len(self.edu_embeddings) == 0:
                error_msg = "No EDU embeddings available for retrieval"
                logger.error(error_msg)
                raise ValueError(error_msg)

            try:
                query_edu_scores = np.dot(self.edu_embeddings, query_embedding.T) # shape: (#edus, )
                query_edu_scores = np.squeeze(query_edu_scores) if query_edu_scores.ndim == 2 else query_edu_scores
            except Exception as e:
                raise e


            # rerank edus
            link_top_k: int = self.global_config.linking_top_k
            if len(query_edu_scores) == 0 or len(self.edu_node_keys) == 0:
                error_msg = f"Cannot proceed with retrieval - query_edu_scores: {len(query_edu_scores)}, edu_node_keys: {len(self.edu_node_keys)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Log the trace
            query_trace["query_to_edu"] = None
            try:
                # Get the top k edus by scores
                if len(query_edu_scores) <= link_top_k:
                    # If we have fewer edus than requested, use all of them
                    candidate_edu_indices = np.argsort(query_edu_scores)[::-1].tolist()
                else:
                    # Otherwise get the top k
                    candidate_edu_indices = np.argsort(query_edu_scores)[-link_top_k:][::-1].tolist()
                
                # # ========== APPLY FEATURE 1: Balance candidate EDUs by speaker ==========
                # candidate_edu_indices = balance_edus_by_speaker(
                #     candidate_edu_indices, query_edu_scores, self.edu_node_keys, 
                #     self.edu_node_idx_to_speaker, self.node_name_to_vertex_idx, link_top_k
                # )
                # # ========== END APPLY FEATURE 1 ==========
                
                # Get the actual edu IDs
                real_candidate_edu_ids = [self.edu_node_keys[idx] for idx in candidate_edu_indices]
                edu_row_dict = self.edu_embedding_store.get_rows(real_candidate_edu_ids)
                candidate_edus = [edu_row_dict[idx]['content'] for idx in real_candidate_edu_ids]

                # Log the trace
                query_trace["query_to_edu"] = {
                    "pre_rerank": {
                        "real_candidate_edu_ids": real_candidate_edu_ids,
                        "edu_row_dict": edu_row_dict,
                        "candidate_edus": candidate_edus,
                    },
                    "post_rerank": None
                }

                # Rerank the edus
                top_k_edu_indices, top_k_edus, reranker_dict = self.rerank_filter_structuring_v1(
                    query,
                    candidate_edus,
                    candidate_edu_indices,
                    len_after_rerank=link_top_k
                )
                # top_k_edu_indices, top_k_edus = candidate_edu_indices, candidate_edus

                if self.global_config.skip_retrieval_ppr:
                    # Enrich EDUs with speaker and temporal metadata
                    enriched_edus = self._enrich_edus_with_metadata(top_k_edu_indices, [])
                    
                    retrieval_results.append(QuerySolution(
                        question=query, 
                        docs=['1','2'], 
                        doc_scores=[1,2], 
                        edus=enriched_edus, 
                        edu_scores=[1,2]
                    ))

                    # Log the trace
                    query_trace["query_to_edu"]["post_rerank"] = {
                        "top_k_edu_indices": top_k_edu_indices,
                        "top_k_edus": top_k_edus,
                    }
                    all_query_traces.append(query_trace)
                    continue 


                rerank_log = {'edus_before_rerank': candidate_edus, 'edus_after_rerank': top_k_edus}

                # Log the trace
                query_trace["query_to_edu"]["post_rerank"] = {
                    "top_k_edu_indices": top_k_edu_indices,
                    "top_k_edus": top_k_edus,
                }

            except Exception as e:
                logger.error(f"Error in EDU reranking: {str(e)}")
                top_k_edu_indices, top_k_edus, rerank_log = [], [], {'edus_before_rerank': [], 'edus_after_rerank': [], 'error': str(e)}
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_edus) == 0:
                if not self.global_config.query_to_session_retrieval:
                    logger.info(f"No facts found after reranking, using the pre-rerank {len(candidate_edus)} EDUs")
                    top_k_edu_indices, top_k_edus = candidate_edu_indices, candidate_edus
                    assert len(top_k_edu_indices) > 0 and len(top_k_edus) > 0


            # Assigning argument and edu weights based on selected edus from previous steps
            linking_score_map = {} # from argument to the average scores of the edus that contain the argument
            argument_scores = {} # store all edu scores for each argument regardless of whether they exist in the knowledge graph or not
            argument_weights = np.zeros(len(self.graph.vs['name']))
            edu_weights = np.zeros(len(self.graph.vs['name']))
            session_weights = np.zeros(len(self.graph.vs['name']))


            # Log the trace
            query_trace["query_ner_to_argument_linking"] = {
                "pre_rerank": [],
                "post_rerank": None
            }
            
            # ========== APPROACH: Retrieve top K similar arguments per entity, then filter with LLM ==========
            # Add query entity matching weights to arguments
            query_entities = query_to_entities.get(query, [])
            
            # Collect all candidate arguments from all query entities
            all_candidate_arg_info = {}  # arg_key -> (arg_content, max_similarity_score, source_entities)
            
            if len(query_entities) > 0 and len(entity_to_embedding) > 0:
                for entity in query_entities:
                    if entity in entity_to_embedding:
                        entity_embedding = entity_to_embedding[entity]
                        
                        # Compute similarity scores with all arguments
                        if len(self.argument_embeddings) > 0:
                            entity_arg_scores = np.dot(self.argument_embeddings, entity_embedding.T)
                            entity_arg_scores = np.squeeze(entity_arg_scores) if entity_arg_scores.ndim == 2 else entity_arg_scores
                            
                            # Get top 10 similar arguments for this entity (no threshold)
                            top_k_arg_indices = np.argsort(entity_arg_scores)[-10:][::-1]
                            
                            for arg_idx in top_k_arg_indices:
                                arg_key = self.argument_node_keys[arg_idx]
                                similarity_score = entity_arg_scores[arg_idx]
                                
                                # Get argument content
                                arg_row = self.argument_embedding_store.get_row(arg_key)
                                arg_content = arg_row["content"]
                                
                                # Store candidate argument info (keep track of max similarity and source entities)
                                if arg_key not in all_candidate_arg_info:
                                    all_candidate_arg_info[arg_key] = {
                                        "content": arg_content,
                                        "max_score": similarity_score,
                                        "source_entities": [entity],
                                        "arg_idx": arg_idx
                                    }
                                else:
                                    # Update max score if this entity has higher similarity
                                    if similarity_score > all_candidate_arg_info[arg_key]["max_score"]:
                                        all_candidate_arg_info[arg_key]["max_score"] = similarity_score
                                    all_candidate_arg_info[arg_key]["source_entities"].append(entity)
                                
                                # Store for pre-rerank tracing
                                query_trace["query_ner_to_argument_linking"]["pre_rerank"].append({
                                    "query_entity": entity,
                                    "arg_idx": int(arg_idx),
                                    "arg_content": arg_content,
                                    "arg_key": arg_key,
                                    "query_arg_similarity_score": float(similarity_score)
                                })
            
            logger.info(f"Num of candidate argument nodes before reranking={len(all_candidate_arg_info)}")
            
            # Prepare candidate arguments for filtering
            candidate_arg_keys = list(all_candidate_arg_info.keys())
            candidate_arg_contents = [all_candidate_arg_info[k]["content"] for k in candidate_arg_keys]
            candidate_arg_scores = [all_candidate_arg_info[k]["max_score"] for k in candidate_arg_keys]
            
            # Filter arguments using LLM-based reranker (similar to EDU filtering)
            filtered_arg_keys = []
            filtered_arg_contents = []
            filtered_arg_scores = []
            reranker_metadata = {}
            
            if len(candidate_arg_contents) > 0:
                try:
                    filtered_arg_keys, filtered_arg_contents, filtered_arg_scores, reranker_metadata = self.rerank_filter_arguments_v1(
                        query,
                        candidate_arg_contents,
                        candidate_arg_keys,
                        candidate_arg_scores,
                        len_after_rerank=None  # Keep all selected arguments
                    )
                    logger.info(f"Identified {len(filtered_arg_keys)} relevant argument nodes")
                except Exception as e:
                    logger.error(f"Error in argument reranking: {str(e)}")
                    # Fallback: use all candidates
                    filtered_arg_keys = candidate_arg_keys
                    filtered_arg_contents = candidate_arg_contents
                    filtered_arg_scores = candidate_arg_scores
                    reranker_metadata = {'error': str(e)}
            
            # Store post-rerank trace
            query_trace["query_ner_to_argument_linking"]["post_rerank"] = {
                "filtered_arg_keys": filtered_arg_keys,
                "filtered_arg_contents": filtered_arg_contents,
                "filtered_arg_scores": [float(s) for s in filtered_arg_scores],
                "reranker_metadata": reranker_metadata
            }
            
            logger.info(f"Num of linked argument nodes after reranking={len(filtered_arg_keys)}")
            
            # Initialize argument weights based on filtered arguments
            for idx, arg_key in enumerate(filtered_arg_keys):
                arg_id = self.node_name_to_vertex_idx.get(arg_key, None)
                
                if arg_id is not None:
                    similarity_score = filtered_arg_scores[idx]
                    
                    # Add argument weight (normalized by frequency)
                    # argument_weights[arg_id] += similarity_score
                    argument_weights[arg_id] += similarity_score / len(self.arg_id_to_session_ids[arg_key]) if (len(self.arg_id_to_session_ids.get(arg_key, set())) > 0) else similarity_score
                    
                    # Store in argument_scores for compatibility
                    argument_scores[filtered_arg_contents[idx]] = argument_weights[arg_id]
                else:
                    logger.warning("arg_key does not exists in self.node_name_to_vertex_idx")
            # ========== END APPROACH ==========


            # Log the trace
            query_trace["linking"] = {
                "argument_weights_sparse": {},
                "edu_weights_sparse": {},
                "passage_weights_sparse": {},
                "traces": {}
            }

            for rank, edu in enumerate(top_k_edus):
                edu_score = query_edu_scores[top_k_edu_indices[rank]] if query_edu_scores.ndim > 0 else query_edu_scores
                
                # Get the EDU key
                edu_key = compute_mdhash_id(content=edu, prefix="edu-")
                edu_id = self.node_name_to_vertex_idx.get(edu_key, None)
                
                if edu_id is not None:
                    # Assign EDU weight directly to the EDU node
                    edu_weights[edu_id] = edu_score
                    
                    # Normalize by frequency if the EDU appears in multiple chunks
                    if len(self.edu_id_to_session_ids.get(edu_key, set())) > 0:
                        edu_weights[edu_id] /= len(self.edu_id_to_session_ids[edu_key])
                    
                    # Log the trace
                    query_trace["linking"]["edu_weights_sparse"][(rank, edu, edu_key, edu_id)] = edu_weights[edu_id]
                    query_trace["linking"]["traces"][(rank, edu, edu_key, edu_id)] = {}
                    
                    # Note: EDU scores are tracked separately from linking_score_map to avoid conflicts with argument filtering
                    # Find connected argument nodes in the graph
                    neighbor_indices = self.graph.neighbors(self.graph.vs[edu_id])
                    for neighbor_idx in neighbor_indices:
                        neighbor_vertex = self.graph.vs[neighbor_idx]
                        if neighbor_vertex['name'].startswith("argument-"):
                            arg_id = self.node_name_to_vertex_idx[neighbor_vertex['name']]

                            argument_weights[arg_id] += edu_score / len(self.arg_id_to_session_ids[neighbor_vertex['name']]) if (len(self.arg_id_to_session_ids.get(neighbor_vertex['name'], set())) > 0) else edu_score # Normalize by frequency if the argument appears in multiple chunks

                            # # Normalize by frequency if the argument appears in multiple chunks
                            # if len(self.arg_id_to_chunk_ids.get(neighbor_vertex['name'], set())) > 0:
                            #     argument_weights[arg_id] /= len(self.arg_id_to_chunk_ids[neighbor_vertex['name']])

                            # Get argument content for linking score map
                            arg_row = self.argument_embedding_store.get_row(neighbor_vertex['name'])
                            arg_content = arg_row["content"]

                            # if arg_content not in argument_scores:
                            #     argument_scores[arg_content] = []
                            # argument_scores[arg_content].append(edu_score)
                            argument_scores[arg_content] = argument_weights[arg_id]

                            # Log the trace
                            query_trace["linking"]["argument_weights_sparse"][(arg_content, neighbor_vertex['name'], arg_id)] = argument_weights[arg_id]
                            query_trace["linking"]["traces"][(rank, edu, edu_key, edu_id)][(arg_content, neighbor_vertex['name'], arg_id)] = edu_score / len(self.arg_id_to_session_ids[neighbor_vertex['name']]) if (len(self.arg_id_to_session_ids.get(neighbor_vertex['name'], set())) > 0) else edu_score


            # Calculate average EDU score for each argument
            for arg_content, scores in argument_scores.items():
                linking_score_map[arg_content] = scores # float(np.mean(scores))

            # Apply top-k filtering for arguments if specified
            if self.global_config.linking_top_k:
                argument_weights, linking_score_map = self.get_top_k_arguments_weights(
                    self.global_config.linking_top_k,
                    argument_weights,
                    linking_score_map
                )
                logger.info(f"Number of final argument node with nonzero weights={len(linking_score_map)}")

            # Log the trace
            query_trace["top_k_argument_filtered"] = deepcopy(linking_score_map)


            if self.global_config.query_to_session_retrieval:
                # Get session scors according to dense retrieval model
                dpr_sorted_session_ids, dpr_sorted_session_scores = self.dense_session_retrieval(query)
                normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_session_scores)
                # Log the trace
                query_trace["top_30_query_dpr_results"] = [
                ]
                for i, dpr_sorted_session_id in enumerate(dpr_sorted_session_ids.tolist()):
                    session_node_key = self.session_node_keys[dpr_sorted_session_id]
                    session_dpr_score = normalized_dpr_sorted_scores[i]
                    session_node_id = self.node_name_to_vertex_idx[session_node_key]
                    if i <= 7:
                        session_weights[session_node_id] = session_dpr_score * self.global_config.passage_node_weight
                    session_node_text = self.session_id_to_summary[session_node_key]
                    linking_score_map[session_node_text] = session_dpr_score * self.global_config.passage_node_weight

                    # Log the trace
                    if i == 0: query_trace["session_node_weight_factor"] = self.global_config.passage_node_weight
                    if len(query_trace["top_30_query_dpr_results"]) < 30:
                        query_trace["top_30_query_dpr_results"].append({
                            "relative_index": i,
                            "dpr_sorted_session_id": dpr_sorted_session_id,
                            "session_node_key": session_node_key,
                            "session_dpr_score": session_dpr_score,
                            "session_node_id": session_node_id,
                            "session_node_weight": session_weights[session_node_id],
                            "session_node_text": session_node_text
                        })


            # Add EDU scores to linking_score_map for logging (after argument filtering is complete)
            for rank, edu in enumerate(top_k_edus):
                edu_score = query_edu_scores[top_k_edu_indices[rank]] if query_edu_scores.ndim > 0 else query_edu_scores
                linking_score_map[edu] = edu_score

            # Combining argument, edu, and passage scores into one array for PPR
            node_weights = argument_weights + edu_weights + session_weights

            # Recording top 30 items in linking_score_map
            if len(linking_score_map) > 30:
                linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

            assert sum(node_weights) > 0, f'No arguments or EDUs found in the graph for the given EDUs: {top_k_edus}'

            # Running PPR algorithm based on the passage, argument, and EDU weights previously assigned
            ppr_start = time.time()
            ppr_sorted_session_ids, ppr_sorted_session_scores, ppr_sorted_edu_ids, ppr_sorted_edu_scores = self.run_ppr_structuring(node_weights, damping=self.global_config.damping)
            ppr_end = time.time()

            self.ppr_time += (ppr_end - ppr_start)

            assert len(ppr_sorted_session_ids) == len(
                    self.session_node_idxs) and len(ppr_sorted_edu_ids) == len(self.edu_node_idxs)
            
            sorted_session_ids, sorted_session_scores, sorted_edu_ids, sorted_edu_scores = ppr_sorted_session_ids, ppr_sorted_session_scores, ppr_sorted_edu_ids, ppr_sorted_edu_scores

            # Log the trace
            query_trace["ppr_results"] = {
                "top_60_sorted_session_ids": sorted_session_ids[:60],
                "top_60_sorted_session_scores": sorted_session_scores[:60],
                "top_60_sorted_session_texts": [self.session_content_store.get_row(self.session_node_keys[idx])["content"] for idx in sorted_session_ids[:60]],
                "top_60_sorted_edu_ids": sorted_edu_ids[:60],
                "top_60_sorted_edu_scores": sorted_edu_scores[:60],
                "top_60_sorted_edu_texts": [self.edu_embedding_store.get_row(self.edu_node_keys[idx])["content"] for idx in sorted_edu_ids[:60]],
            }
        
            # # ========== APPLY FEATURE 2: Balance PPR-sorted EDUs by speaker ==========
            # sorted_edu_ids, sorted_edu_scores = balance_ppr_edus_by_speaker(
            #     sorted_edu_ids, sorted_edu_scores, self.edu_node_keys, self.edu_node_idxs,
            #     self.edu_node_idx_to_speaker, self.global_config.qa_top_k
            # )
            # # ========== END APPLY FEATURE 2 ==========

            
            # Enrich EDUs with speaker and temporal metadata
            enriched_edus = self._enrich_edus_with_metadata(sorted_edu_ids[:num_to_retrieve], sorted_edu_scores[:num_to_retrieve])
            
            retrieval_results.append(QuerySolution(
                question=query, 
                docs=[self.session_content_store.get_row(self.session_node_keys[idx])["content"] for idx in sorted_session_ids[:num_to_retrieve]], 
                doc_scores=sorted_session_scores[:num_to_retrieve], 
                edus=enriched_edus, 
                edu_scores=sorted_edu_scores[:num_to_retrieve]
            ))

            # Log the trace
            all_query_traces.append(query_trace)

        
        retrieve_end_time = time.time()  # Record end time
        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval

        # Log the trace
        if not no_query_trace_saving:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            trace_filename = f"query_retrieval_traces_{timestamp}.pkl"
            with open(os.path.join(self.working_dir, trace_filename), 'wb') as f:
                pickle.dump(all_query_traces, f)

        if no_query_trace_saving:
            return retrieval_results, all_query_traces
        else:
            return retrieval_results