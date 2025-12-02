import json
import re
import os
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import copy
from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import compute_mdhash_id, EDURawOutput, EventRawOutput, EDURERawOutput, QueryNerRawOutput, ConversationEDURawOutput, ConversationContextualEventRawOutput, ConversationEDURawOutputWithTurnIds, ConversationContextualEventRawOutputWithTurnIds
from ..utils.config_utils import get_support_json_schema
from ..llm.openai_gpt_batch import CacheOpenAI


from ..prompts.templates.unified_event_extraction import UnifiedEventExtraction
from ..prompts.templates.conversation_edu_extraction_v1 import ConversationEDUExtractionV1, ConversationEDUV1

logger = get_logger(__name__)



class OpenStructuring:
    def __init__(self, llm_model: CacheOpenAI, max_workers: Optional[int] = None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model

        self.max_workers = max_workers or (os.cpu_count() - os.cpu_count() // 6) # determines how many threads to use for calling LLM API


        # Read environment variable and assign to support_json_schema
        try:
            self.support_json_schema = get_support_json_schema() # indicate whether the LLM engine supports json schema or not. If set to False, we will need to integrate json schema into the prompt explicitly.
            if self.support_json_schema:
                logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'true' - JSON schema support enabled")
            else:
                logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'false' - JSON schema support disabled")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise


    def batch_unified_event_extraction(self, chunk_keys: List[str], edus: List[str]) -> List[EventRawOutput]:
        """
        Unified event extraction that performs both event detection and extraction in one step.
        Extracts both event type and role-argument pairs from EDUs directly.
        
        Args:
            chunk_keys: List of chunk identifiers
            edus: List of EDU texts to process
            
        Returns:
            List[EventRawOutput]: List of event extraction results with event_type and role_argument_pairs.
                                 event_triggers will be an empty list for backward compatibility.
        """
        batch_messages = [
            self.prompt_template_manager.render(
                name="unified_event_extraction",
                edu=edu
            )
            for edu in edus
        ]

        batch_raw_response = ["" for _ in range(len(edus))]
        batch_metadata = [{} for _ in range(len(edus))]

        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=UnifiedEventExtraction)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)

        batch_results = []

        for batch_idx in range(len(edus)):
            raw_response = batch_raw_response[batch_idx]
            metadata = batch_metadata[batch_idx]

            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response

                unified_event: UnifiedEventExtraction = UnifiedEventExtraction.model_validate_json(real_response)  

                result = EventRawOutput(
                    chunk_id=chunk_keys[batch_idx],
                    edu_id=compute_mdhash_id(content=edus[batch_idx], prefix='edu-'),
                    edu_text=edus[batch_idx],
                    response={"unified_event_extraction": raw_response},
                    event_type=unified_event.event_type,
                    event_triggers=[],  # Empty list for backward compatibility
                    event_role_argument_pairs=[{"role": role_argument_pair.role, "argument": role_argument_pair.argument} for role_argument_pair in unified_event.role_argument_pairs],
                    metadata={"unified_event_extraction": metadata},
                )

            except Exception as e:
                logger.warning(e)
                metadata.update({'error': str(e)})
                
                result = EventRawOutput(
                    chunk_id=chunk_keys[batch_idx],
                    edu_id=compute_mdhash_id(content=edus[batch_idx], prefix='edu-'),
                    edu_text=edus[batch_idx],
                    response={"unified_event_extraction": raw_response},
                    event_type=None,
                    event_triggers=None,
                    event_role_argument_pairs=None,
                    metadata={"unified_event_extraction": metadata},
                )

            batch_results.append(result)

        return batch_results


    def batch_query_ner(self, queries: List[str]) -> List[QueryNerRawOutput]:
        """
        Extract named entities from queries using batch processing.
        
        Args:
            queries (List[str]): List of query strings to extract entities from
            
        Returns:
            List[QueryNerRawOutput]: List of query NER results containing extracted entities
        """
        # Import the Pydantic model for query NER
        from ..prompts.templates.ner_query import QueryNerExtraction
        
        batch_messages = []
        for query in queries:
            # Create NER messages for each query with specific instructions for high precision
            messages = self.prompt_template_manager.render(
                name='ner_query', 
                query=query
            )
            batch_messages.append(messages)
        
        # Batch inference
        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=QueryNerExtraction)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)
        
        batch_results = []
        
        for batch_idx, query in enumerate(queries):
            raw_response = ""
            metadata = {}
            
            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                # Extract entities from response using Pydantic model
                query_ner_extraction = QueryNerExtraction.model_validate_json(real_response)
                unique_entities = list(dict.fromkeys(query_ner_extraction.named_entities))
                
                result = QueryNerRawOutput(
                    query=query,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.warning(f"Query NER failed for query: {query[:50]}... Error: {e}")
                metadata.update({'error': str(e)})
                
                result = QueryNerRawOutput(
                    query=query,
                    response=raw_response,
                    unique_entities=[],
                    metadata=metadata
                )
            
            batch_results.append(result)
        
        return batch_results

    
    def _format_session_text(self, session: Any, include_turn_ids: bool = False) -> str:
        """
        Helper method to format a session into text for processing.
        
        Args:
            session: Session object with date_time and turns
            include_turn_ids: Whether to include turn IDs in the output
            
        Returns:
            str: Formatted session text
        """
        lines = [f"Date: {session.date_time}", ""]
        
        for i, turn in enumerate(session.turns, start=1):
            if include_turn_ids:
                lines.append(f"Turn {i}:")
                lines.append(f"{turn.speaker}: {turn.text}")
                lines.append("")  # Empty line for spacing
            else:
                lines.append(f"{turn.speaker}: {turn.text}")
        
        # Remove trailing empty line if present
        if lines and lines[-1] == "":
            lines = lines[:-1]
        
        if include_turn_ids:
            return "\n".join(lines)
        else:
            return "\n".join(lines)
    
    
    def _create_fallback_edus_from_sentences(self, session_id: str, session_text: str,
                                            session_date: str, speaker_names: List[str]) -> List[ConversationEDURawOutput]:
        """
        Create fallback EDUs by splitting session text into sentences.
        This is the last-resort fallback when round-based parsing fails.
        
        Args:
            session_id: Identifier for the session
            session_text: The formatted session conversation text
            session_date: Date of the session
            speaker_names: List of valid speaker names
            
        Returns:
            List[ConversationEDURawOutput]: List of EDUs, one per sentence
        """
        try:
            # Remove date line if present
            lines = [line for line in session_text.split('\n') if line.strip() and not line.strip().startswith('Date:')]
            full_text = ' '.join(lines)
            
            if not full_text.strip():
                logger.error(f"No text content found for session {session_id}")
                return []
            
            # Split by sentence boundaries (., !, ?)
            # Simple split - could use nltk.sent_tokenize for better accuracy if needed
            import re
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            
            # Filter out empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                logger.error(f"No sentences found after splitting for session {session_id}")
                return []
            
            # Create EDUs from sentences
            edu_results = []
            for i, sentence in enumerate(sentences, 1):
                edu_result = ConversationEDURawOutput(
                    session_id=session_id,
                    edu_id=compute_mdhash_id(content=sentence, prefix='edu-'),
                    edu_text=sentence,
                    source_speakers=speaker_names,  # Assign all speakers since we can't determine from sentence alone
                    date=session_date,
                    response="",  # No LLM response for fallback
                    metadata={
                        'fallback': True,
                        'fallback_method': 'sentence_splitting',
                        'sentence_number': i,
                        'total_sentences': len(sentences)
                    }
                )
                edu_results.append(edu_result)
            
            logger.info(f"Created {len(edu_results)} sentence-based fallback EDUs for session {session_id}")
            return edu_results
            
        except Exception as e:
            logger.error(f"Sentence-split fallback EDU creation failed for session {session_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return []
    
    
    def _create_fallback_edus_from_rounds_with_turn_ids(self, session_id: str, session_item: Any, 
                                                       session_date: str, speaker_names: List[str]) -> List[ConversationEDURawOutputWithTurnIds]:
        """
        Create fallback EDUs by grouping consecutive speaker turns into rounds.
        This version uses the session_item object directly and adds Assistant turns as separate EDUs.
        Returns ConversationEDURawOutputWithTurnIds with turn ID tracking.
        
        Args:
            session_id: Identifier for the session
            session_item: Session object with turns
            session_date: Date of the session
            speaker_names: List of valid speaker names
            
        Returns:
            List[ConversationEDURawOutputWithTurnIds]: List of EDUs created from rounds and Assistant turns
        """
        try:
            turns = session_item.turns
            
            if not turns:
                logger.error(f"No turns found in session {session_id}")
                return []
            
            # Group consecutive turns into rounds (pairs)
            edu_results = []
            turn_idx = 0
            round_num = 0
            
            # Extract User EDUs by grouping turns
            while turn_idx < len(turns):
                # Collect turns for this round (typically 2 turns: A then B, but flexible)
                round_turn_indices = [turn_idx]
                round_turns = [turns[turn_idx]]
                current_speakers = {turns[turn_idx].speaker}
                turn_idx += 1
                
                # Add next turn if it's from a different speaker (completing the round)
                while turn_idx < len(turns) and turns[turn_idx].speaker not in current_speakers and len(round_turns) < 3:
                    round_turn_indices.append(turn_idx)
                    round_turns.append(turns[turn_idx])
                    current_speakers.add(turns[turn_idx].speaker)
                    turn_idx += 1
                
                # Create EDU text from the round
                round_texts = [f"{turn.speaker}: {turn.text}" for turn in round_turns]
                edu_text = "\n\n".join(round_texts)
                
                # Get speakers involved in this round
                round_speakers = list(current_speakers)
                
                # Validate speakers
                valid_speakers = [speaker for speaker in round_speakers if speaker in speaker_names]
                if not valid_speakers:
                    valid_speakers = speaker_names  # Fallback to all speakers
                
                # Create EDU output with turn IDs (1-indexed)
                round_num += 1
                edu_result = ConversationEDURawOutputWithTurnIds(
                    session_id=session_id,
                    edu_id=compute_mdhash_id(content=edu_text, prefix='edu-'),
                    edu_text=edu_text,
                    source_speakers=valid_speakers,
                    date=session_date,
                    response="",  # No LLM response for fallback
                    metadata={
                        'fallback': True,
                        'fallback_method': 'round_chunking_with_turn_ids',
                        'round_number': round_num,
                        'num_turns_in_round': len(round_turns)
                    },
                    source_turn_ids=[idx + 1 for idx in round_turn_indices]  # Convert to 1-indexed
                )
                edu_results.append(edu_result)
            
            logger.info(f"Created {len(edu_results)} fallback EDUs from rounds for session {session_id}")
            return edu_results
            
        except Exception as e:
            logger.error(f"Round-based fallback with turn IDs EDU creation failed for session {session_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Last resort: try sentence splitting
            logger.info(f"Attempting sentence-split fallback for session {session_id}")
            formatted_text = self._format_session_text(session_item, include_turn_ids=False)
            return self._create_fallback_edus_from_sentences(session_id, formatted_text, session_date, speaker_names)



    def batch_query_er(self, queries: List[str]) -> List[QueryNerRawOutput]:
        """
        Extract named entities from queries using batch processing.
        
        Args:
            queries (List[str]): List of query strings to extract entities from
            
        Returns:
            List[QueryNerRawOutput]: List of query NER results containing extracted entities
        """
        # Import the Pydantic model for query NER
        # from ..prompts.templates.ner_query import QueryNerExtraction
        from ..prompts.templates.er_query import QueryERExtraction

        batch_messages = []
        for query in queries:
            # Create NER messages for each query with specific instructions for high precision
            messages = self.prompt_template_manager.render(
                name='er_query', 
                query=query
            )
            batch_messages.append(messages)
        
        # Batch inference
        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=QueryERExtraction)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)
        
        batch_results = []
        
        for batch_idx, query in enumerate(queries):
            raw_response = ""
            metadata = {}
            
            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                # Extract entities from response using Pydantic model
                query_ner_extraction = QueryERExtraction.model_validate_json(real_response)
                unique_entities = list(dict.fromkeys(query_ner_extraction.entities_and_concepts))
                
                result = QueryNerRawOutput(
                    query=query,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.warning(f"Query NER failed for query: {query[:50]}... Error: {e}")
                metadata.update({'error': str(e)})
                
                result = QueryNerRawOutput(
                    query=query,
                    response=raw_response,
                    unique_entities=[],
                    metadata=metadata
                )
            
            batch_results.append(result)
        
        return batch_results
    

    def batch_conversation_edu_extraction_without_context_user_edus_only(self, session_ids: List[str], session_texts: List[str], 
                                                        session_dates: List[str], speaker_names_list: List[List[str]], session_items = None) -> List[List[ConversationEDURawOutputWithTurnIds]]:
        """
        Extract EDUs from multiple conversation sessions' User utterances ONLY without previous context in batch.
        Unlike batch_conversation_edu_extraction_without_context_user_edus, this method does NOT add Assistant turns as EDUs.
        Uses turn attribution to map extracted EDUs to specific turn IDs and determine source speakers.
        Returns ConversationEDURawOutputWithTurnIds which includes source_turn_ids field.
        
        This method is designed to be used in conjunction with batch_conversation_edu_extraction_without_context_assistant_edus
        to extract User and Assistant EDUs separately.
        """
        from ..prompts.templates.conversation_edu_extraction_longmemeval_user_side_w_turn_attribution import ConversationEDUExtractionWithTurnAttribution
        
        # Format session texts with turn IDs
        batch_messages = []
        batch_turn_mappings = []  # List of dicts mapping turn_id to turn object for each session
        
        for batch_idx, (session_text, speaker_names) in enumerate(zip(session_texts, speaker_names_list)):
            session_id = session_ids[batch_idx]
            session_item = session_items[batch_idx]
            
            # Create turn ID to turn mapping
            turn_id_to_turn = {}
            for i, turn in enumerate(session_item.turns, start=1):
                turn_id_to_turn[i] = turn
            batch_turn_mappings.append(turn_id_to_turn)
            
            # Format session text with turn IDs
            formatted_session_text = self._format_session_text(session_item, include_turn_ids=True)
            
            messages = self.prompt_template_manager.render(
                name="conversation_edu_extraction_longmemeval_user_side_w_turn_attribution",
                session_text=formatted_session_text,
                speaker_names=", ".join(speaker_names)
            )
            batch_messages.append(messages)
        
        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=ConversationEDUExtractionWithTurnAttribution)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)
        
        batch_results = []
        
        for batch_idx in range(len(session_texts)):
            session_id = session_ids[batch_idx]
            session_date = session_dates[batch_idx]
            speaker_names = speaker_names_list[batch_idx]
            turn_id_to_turn = batch_turn_mappings[batch_idx]
            session_item = session_items[batch_idx]
            
            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                conversation_edu_extraction = ConversationEDUExtractionWithTurnAttribution.model_validate_json(real_response)
                
                # Convert to ConversationEDURawOutputWithTurnIds objects (User side only)
                edu_results = []
                for edu in conversation_edu_extraction.edus:
                    # Determine source speakers from turn IDs
                    source_speakers = []
                    for turn_id in edu.source_turn_ids:
                        if turn_id in turn_id_to_turn:
                            turn = turn_id_to_turn[turn_id]
                            if turn.speaker not in source_speakers:
                                source_speakers.append(turn.speaker)
                        else:
                            logger.warning(f"Turn ID {turn_id} not found in session {session_id}")
                    
                    # Fallback to all speakers if no valid speakers found
                    if not source_speakers:
                        logger.warning(f"No valid speakers found for EDU: {edu.edu_text[:50]}... Using all provided speakers.")
                        source_speakers = speaker_names
                    
                    # Add edu_type to metadata for User EDUs
                    user_metadata = copy.deepcopy(metadata)
                    user_metadata['edu_type'] = 'user_edu'
                    
                    edu_result = ConversationEDURawOutputWithTurnIds(
                        session_id=session_id,
                        edu_id=compute_mdhash_id(content=edu.edu_text, prefix='edu-'),
                        edu_text=edu.edu_text,
                        source_speakers=source_speakers,
                        date=session_date,
                        response=raw_response,
                        metadata=user_metadata,
                        source_turn_ids=edu.source_turn_ids
                    )
                    edu_results.append(edu_result)
                
                batch_results.append(edu_results)
                
            except Exception as e:
                logger.warning(f"User-side EDU extraction failed for session {session_id}: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                
                # Fallback: Create User EDUs from User turns only
                logger.info(f"Using fallback User EDU chunking for session {session_id}: each User turn as an EDU")
                fallback_edus = []
                for i, turn in enumerate(session_item.turns, start=1):
                    if turn.speaker == "User":
                        edu_text = f"[{session_date}] User: {turn.text}"
                        edu_result = ConversationEDURawOutputWithTurnIds(
                            session_id=session_id,
                            edu_id=compute_mdhash_id(content=edu_text, prefix='edu-'),
                            edu_text=edu_text,
                            source_speakers=["User"],
                            date=session_date,
                            response="",
                            metadata={'fallback': True, 'fallback_method': 'user_turn_as_edu', 'edu_type': 'user_edu'},
                            source_turn_ids=[i]
                        )
                        fallback_edus.append(edu_result)
                batch_results.append(fallback_edus)
        
        return batch_results


    def batch_conversation_edu_extraction_without_context_assistant_edus_structured(self, session_ids: List[str], session_texts: List[str], 
                                                        session_dates: List[str], speaker_names_list: List[List[str]], session_items = None) -> List[List[ConversationEDURawOutputWithTurnIds]]:
        """
        Extract EDUs from multiple conversation sessions' Assistant utterances without previous context in batch.
        Uses turn attribution to map extracted EDUs/chunks to specific turn IDs and determine source speakers.
        This version extracts both simple EDUs and structured chunks from Assistant responses.
        
        Returns a list where each element contains ConversationEDURawOutputWithTurnIds with:
        - metadata['edu_type'] = 'assistant_edu' for simple EDUs
        - metadata['edu_type'] = 'assistant_chunk' for structured chunks
        - For chunks: edu_text contains chunk_summary, metadata['chunk_content'] contains full content
        """
        from ..prompts.templates.conversation_edu_extraction_longmemeval_assistant_side_structured_v2 import AssistantConversationStructuredExtractionWithTurnAttribution
        
        # Format session texts with turn IDs
        batch_messages = []
        batch_turn_mappings = []  # List of dicts mapping turn_id to turn object for each session
        
        for batch_idx, (session_text, speaker_names) in enumerate(zip(session_texts, speaker_names_list)):
            session_id = session_ids[batch_idx]
            session_item = session_items[batch_idx]
            
            # Create turn ID to turn mapping
            turn_id_to_turn = {}
            for i, turn in enumerate(session_item.turns, start=1):
                turn_id_to_turn[i] = turn
            batch_turn_mappings.append(turn_id_to_turn)
            
            # Format session text with turn IDs
            formatted_session_text = self._format_session_text(session_item, include_turn_ids=True)
            
            messages = self.prompt_template_manager.render(
                name="conversation_edu_extraction_longmemeval_assistant_side_structured_v2",
                session_text=formatted_session_text,
                speaker_names=", ".join(speaker_names)
            )
            batch_messages.append(messages)
        
        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=AssistantConversationStructuredExtractionWithTurnAttribution)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)
        
        batch_results = []
        
        for batch_idx in range(len(session_texts)):
            session_id = session_ids[batch_idx]
            session_date = session_dates[batch_idx]
            speaker_names = speaker_names_list[batch_idx]
            turn_id_to_turn = batch_turn_mappings[batch_idx]
            session_item = session_items[batch_idx]
            
            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                structured_extraction = AssistantConversationStructuredExtractionWithTurnAttribution.model_validate_json(real_response)
                
                # Convert both simple EDUs and structured chunks to output objects
                combined_results = []
                
                # Process simple EDUs
                for edu in structured_extraction.simple_edus:
                    # Determine source speakers from turn IDs
                    source_speakers = []
                    for turn_id in edu.source_turn_ids:
                        if turn_id in turn_id_to_turn:
                            turn = turn_id_to_turn[turn_id]
                            if turn.speaker not in source_speakers:
                                source_speakers.append(turn.speaker)
                        else:
                            logger.warning(f"Turn ID {turn_id} not found in session {session_id}")
                    
                    # Fallback to all speakers if no valid speakers found
                    if not source_speakers:
                        logger.warning(f"No valid speakers found for EDU: {edu.edu_text[:50]}... Using all provided speakers.")
                        source_speakers = speaker_names
                    
                    # Create metadata with type marker
                    edu_metadata = copy.deepcopy(metadata)
                    edu_metadata['edu_type'] = 'assistant_edu'
                    
                    edu_result = ConversationEDURawOutputWithTurnIds(
                        session_id=session_id,
                        edu_id=compute_mdhash_id(content=edu.edu_text, prefix='edu-'),
                        edu_text=edu.edu_text,
                        source_speakers=source_speakers,
                        date=session_date,
                        response=raw_response,
                        metadata=edu_metadata,
                        source_turn_ids=edu.source_turn_ids
                    )
                    combined_results.append(edu_result)
                
                # Process structured chunks
                for chunk in structured_extraction.structured_chunks:
                    # Determine source speakers from turn IDs
                    source_speakers = []
                    for turn_id in chunk.source_turn_ids:
                        if turn_id in turn_id_to_turn:
                            turn = turn_id_to_turn[turn_id]
                            if turn.speaker not in source_speakers:
                                source_speakers.append(turn.speaker)
                        else:
                            logger.warning(f"Turn ID {turn_id} not found in session {session_id}")
                    
                    # Fallback to all speakers if no valid speakers found
                    if not source_speakers:
                        logger.warning(f"No valid speakers found for chunk: {chunk.chunk_content[:50]}... Using all provided speakers.")
                        source_speakers = speaker_names
                    
                    # Create metadata with type marker and chunk_content
                    chunk_metadata = copy.deepcopy(metadata)
                    chunk_metadata['edu_type'] = 'assistant_chunk'
                    chunk_metadata['chunk_content'] = chunk.chunk_content
                    
                    # Use chunk_summary as edu_text (for retrieval)
                    chunk_result = ConversationEDURawOutputWithTurnIds(
                        session_id=session_id,
                        edu_id=compute_mdhash_id(content=chunk.chunk_summary, prefix='edu-'),
                        edu_text=chunk.chunk_summary,  # chunk_summary for retrieval
                        source_speakers=source_speakers,
                        date=session_date,
                        response=raw_response,
                        metadata=chunk_metadata,
                        source_turn_ids=chunk.source_turn_ids
                    )
                    combined_results.append(chunk_result)

                batch_results.append(combined_results)
                
            except Exception as e:
                logger.warning(f"Conversation Assistant structured EDU extraction failed for session {session_id}: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                
                # Fallback: Use each Assistant turn as a separate EDU (simple EDUs only)
                logger.info(f"Using fallback EDU chunking for Assistant side session {session_id}: each Assistant turn as a simple EDU")
                fallback_edus = []
                for i, turn in enumerate(session_item.turns, start=1):
                    if turn.speaker != "Assistant": 
                        continue
                    edu_text = f"[{session_date}] Assistant: {turn.text}"
                    edu_result = ConversationEDURawOutputWithTurnIds(
                        session_id=session_id,
                        edu_id=compute_mdhash_id(content=edu_text, prefix='edu-'),
                        edu_text=edu_text,
                        source_speakers=["Assistant"],
                        date=session_date,
                        response="",
                        metadata={'fallback': True, 'fallback_method': 'assistant_turn_as_edu', 'edu_type': 'assistant_edu'},
                        source_turn_ids=[i]
                    )
                    fallback_edus.append(edu_result)
                batch_results.append(fallback_edus)
        
        return batch_results


    def batch_conversation_edu_extraction_without_context_v1(self, session_ids: List[str], session_texts: List[str], 
                                                        session_dates: List[str], speaker_names_list: List[List[str]], session_items = None) -> List[List[ConversationEDURawOutputWithTurnIds]]:
        """
        Extract EDUs from multiple conversation sessions without previous context in batch.
        This V1 version adds turn attribution to track which turns each EDU was extracted from.
        Works with all speakers (not limited to User and Assistant).
        
        Note: The LLM only outputs edu_text and source_turn_ids. The source_speakers field is 
        automatically derived from source_turn_ids by looking up the speaker for each turn.
        
        Args:
            session_ids: List of session identifiers
            session_texts: List of formatted session conversation texts (will be reformatted with turn IDs)
            session_dates: List of session dates
            speaker_names_list: List of speaker names for each session
            session_items: List of Session objects (required for turn ID extraction and speaker derivation)
            
        Returns:
            List[List[ConversationEDURawOutputWithTurnIds]]: List of EDU lists with turn IDs for each session
        """
        # Format session texts with turn IDs
        batch_messages = []
        batch_turn_mappings = []  # List of dicts mapping turn_id to turn object for each session
        
        for batch_idx, (session_text, speaker_names) in enumerate(zip(session_texts, speaker_names_list)):
            session_id = session_ids[batch_idx]
            session_item = session_items[batch_idx]
            
            # Create turn ID to turn mapping
            turn_id_to_turn = {}
            for i, turn in enumerate(session_item.turns, start=1):
                turn_id_to_turn[i] = turn
            batch_turn_mappings.append(turn_id_to_turn)
            
            # Format session text with turn IDs
            formatted_session_text = self._format_session_text(session_item, include_turn_ids=True)
            
            messages = self.prompt_template_manager.render(
                name="conversation_edu_extraction_v1",
                session_text=formatted_session_text,
                speaker_names=", ".join(speaker_names)
            )
            batch_messages.append(messages)
        
        if self.support_json_schema:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages, response_format=ConversationEDUExtractionV1)
        else:
            batch_infer_results = self.llm_model.batch_infer(messages=batch_messages)
        
        batch_results = []
        
        for batch_idx in range(len(session_texts)):
            session_id = session_ids[batch_idx]
            session_date = session_dates[batch_idx]
            speaker_names = speaker_names_list[batch_idx]
            turn_id_to_turn = batch_turn_mappings[batch_idx]
            session_item = session_items[batch_idx]
            
            try:
                raw_response, metadata, cache_hit = batch_infer_results[batch_idx]
                metadata['cache_hit'] = cache_hit
                
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                conversation_edu_extraction = ConversationEDUExtractionV1.model_validate_json(real_response)
                
                # Convert to ConversationEDURawOutputWithTurnIds objects
                edu_results = []
                for edu in conversation_edu_extraction.edus:
                    # Determine source speakers from turn IDs
                    source_speakers = []
                    for turn_id in edu.source_turn_ids:
                        if turn_id in turn_id_to_turn:
                            turn = turn_id_to_turn[turn_id]
                            if turn.speaker not in source_speakers:
                                source_speakers.append(turn.speaker)
                        else:
                            logger.warning(f"Turn ID {turn_id} not found in session {session_id}")
                    
                    # Fallback to all speakers if no valid speakers found
                    if not source_speakers:
                        logger.warning(f"No valid speakers found for EDU: {edu.edu_text[:50]}... Using all provided speakers.")
                        source_speakers = speaker_names
                    
                    edu_result = ConversationEDURawOutputWithTurnIds(
                        session_id=session_id,
                        edu_id=compute_mdhash_id(content=edu.edu_text, prefix='edu-'),
                        edu_text=edu.edu_text,
                        source_speakers=source_speakers,
                        date=session_date,
                        response=raw_response,
                        metadata=metadata,
                        source_turn_ids=edu.source_turn_ids
                    )
                    edu_results.append(edu_result)
                
                batch_results.append(edu_results)
                
            except Exception as e:
                logger.warning(f"Conversation EDU extraction V1 failed for session {session_id}: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                
                # Fallback: Create default EDUs by grouping turns into rounds (A+B pairs)
                logger.info(f"Using fallback EDU chunking for session {session_id}: grouping turns into rounds")
                fallback_edus = self._create_fallback_edus_from_rounds_with_turn_ids(
                    session_id=session_id,
                    session_item=session_item,
                    session_date=session_date,
                    speaker_names=speaker_names
                )
                batch_results.append(fallback_edus)
        
        return batch_results


    def batch_conversation_openie_w_indep_summary_uniee_v1_wo_session_summary(self, sessions: Dict[str, Any], speaker_names: List[str], skip_edu_context_gen: bool = True) -> Dict[str, Any]:
        """
        Conduct batch conversation-based OpenIE with independent (non-cumulative) session summaries
        using unified event extraction (event detection + extraction in one step).
        
        V1 improvements:
        - Uses conversation_edu_extraction_v1 which adds turn attribution to track source turns for each EDU
        - Returns ConversationContextualEventRawOutputWithTurnIds with source_turn_ids field
        - Works with all speaker names (not limited to User and Assistant)
        
        Unlike batch_conversation_openie_w_indep_summary which uses separate event detection and extraction steps,
        this function combines both into a single unified extraction step for better efficiency.
        
        Args:
            sessions: Dict mapping from session hash id to Session instance
            speaker_names: List of valid speaker names
            skip_edu_context_gen: Whether to skip EDU context generation. If True, uses placeholder text instead.
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping session_id to session data containing:
                - 'edus': List[ConversationContextualEventRawOutputWithTurnIds] - list of contextual event results with turn IDs for this session
                - 'summary': str - session summary text (independent, not cumulative)
        """
        from ..utils.conversation_data_utils import Session, Turn
        
        # Convert sessions dict to list ordered by session_id
        session_items = [(session_id, session) for session_id, session in sessions.items()]
        session_items.sort(key=lambda x: x[1].session_id)
        
        logger.info(f"Starting conversation OpenIE V1 with independent summaries and unified event extraction for {len(session_items)} sessions")
        
        # Prepare session data
        session_ids = [item[0] for item in session_items]
        session_texts = [self._format_session_text(item[1], include_turn_ids=True) for item in session_items]
        session_dates = [item[1].date_time for item in session_items]
        speaker_names_list = [speaker_names for _ in session_items]  # Same speakers for all sessions
        
        # Step 1: Skip session summarization and use placeholder
        logger.info(f"Skipping batch session summarization for {len(session_items)} sessions (using placeholder)")
        session_summaries = ["No summary available"] * len(session_items)
        
        # Step 2: Batch EDU extraction V1 for all sessions (with turn attribution, without previous summaries)
        logger.info("Starting batch EDU extraction V1 for all sessions (with turn attribution, without previous summaries)")
        batch_edu_results = self.batch_conversation_edu_extraction_without_context_v1(
            session_ids=session_ids,
            session_texts=session_texts,
            session_dates=session_dates,
            speaker_names_list=speaker_names_list,
            session_items=[item[1] for item in session_items]
        )
        
        # Step 3: Prepare data for context generation and unified event extraction
        temp_session_ids = []
        temp_session_texts = []
        temp_edu_texts = []
        temp_edu_objects = []
        session_to_edu_mapping = {}
        
        for session_idx, (session_id, edu_results) in enumerate(zip(session_ids, batch_edu_results)):
            session_to_edu_mapping[session_id] = []
            session_text = session_texts[session_idx]
            
            for edu_result in edu_results:
                temp_session_ids.append(session_id)
                temp_session_texts.append(session_text)
                temp_edu_texts.append(edu_result.edu_text)
                temp_edu_objects.append(edu_result)
                session_to_edu_mapping[session_id].append(len(temp_edu_texts) - 1)
        
        if not temp_edu_texts:
            logger.warning("No EDUs extracted from any session")
            return {}
        
        # Step 4: Skip batch context generation
        assert skip_edu_context_gen
        logger.info(f"Skipping context generation for {len(temp_edu_texts)} EDUs (using placeholder text)")
        context_texts = ["No context information available"] * len(temp_edu_texts)
        context_metadata = [{'prompt_tokens': 0, 'completion_tokens': 0, 'reason_tokens': 0, 'cache_hit': False}] * len(temp_edu_texts)
        

        # Step 5: Batch unified event extraction (combines detection and extraction)
        logger.info(f"Starting batch unified event extraction for {len(temp_edu_texts)} EDUs")
        event_extraction_results = self.batch_unified_event_extraction(
            chunk_keys=temp_session_ids,
            edus=temp_edu_texts
        )
        
        # Step 6: Combine results into ConversationContextualEventRawOutputWithTurnIds
        logger.info("Combining results into final output format")
        all_session_results = {}
        for session_id in session_ids:
            all_session_results[session_id] = []
        
        for i, (context_text, event_result, edu_object) in enumerate(zip(context_texts, event_extraction_results, temp_edu_objects)):
            # edu_object is ConversationEDURawOutputWithTurnIds, preserve the turn IDs
            contextual_result = ConversationContextualEventRawOutputWithTurnIds(
                session_id=edu_object.session_id,
                edu_id=edu_object.edu_id,
                edu_text=edu_object.edu_text,
                source_speakers=edu_object.source_speakers,
                date=edu_object.date,
                context_text=context_text,
                response=event_result.response,
                metadata=event_result.metadata,
                event_type=event_result.event_type,
                event_triggers=event_result.event_triggers,
                event_role_argument_pairs=event_result.event_role_argument_pairs,
                source_turn_ids=edu_object.source_turn_ids  # Preserve turn IDs
            )
            
            all_session_results[edu_object.session_id].append(contextual_result)
        
        logger.info(f"Conversation OpenIE V1 without session summaries completed for {len(session_items)} sessions with {sum(len(results) for results in all_session_results.values())} total EDUs")
        
        # Combine EDUs and summaries for each session
        combined_results = {}
        for i, session_id in enumerate(session_ids):
            combined_results[session_id] = {
                'edus': all_session_results[session_id],
                'summary': session_summaries[i]
            }
        
        return combined_results


    def batch_conversation_openie_w_indep_summary_uniee_separate_edus_v2_wo_session_summary(self, sessions: Dict[str, Any], speaker_names: List[str], skip_edu_context_gen: bool = True) -> Dict[str, Any]:
        """
        Conduct batch conversation-based OpenIE with independent (non-cumulative) session summaries
        using unified event extraction (event detection + extraction in one step).
        
        This method extracts EDUs separately for User and Assistant sides using LLM-based extraction for both.
        **V2 difference**: Uses structured extraction for Assistant side that handles both simple EDUs and 
        structured chunks (for complex information that should be kept together).
        
        All types use ConversationContextualEventRawOutputWithTurnIds with metadata['edu_type'] discriminator:
        - 'user_edu': Simple EDU from User
        - 'assistant_edu': Simple EDU from Assistant  
        - 'assistant_chunk': Structured chunk from Assistant
        
        For assistant_chunk type:
        - edu_text contains chunk_summary (concise, entity-rich summary for retrieval)
        - metadata['chunk_content'] contains full structured content (stored for future use)
        
        Args:
            sessions: Dict mapping from session hash id to Session instance
            speaker_names: List of valid speaker names
            skip_edu_context_gen: Whether to skip EDU context generation. If True, uses placeholder text instead.
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping session_id to session data containing:
                - 'edus': List[ConversationContextualEventRawOutputWithTurnIds] - unified structure for all types
                - 'summary': str - session summary text (independent, not cumulative)
        """
        from ..utils.conversation_data_utils import Session, Turn
        
        # Convert sessions dict to list ordered by session_id
        session_items = [(session_id, session) for session_id, session in sessions.items()]
        session_items.sort(key=lambda x: x[1].session_id)
        
        logger.info(f"Starting conversation OpenIE V2 with independent summaries, unified event extraction, and structured assistant extraction for {len(session_items)} sessions")
        
        # Prepare session data
        session_ids = [item[0] for item in session_items]
        session_texts = [self._format_session_text(item[1], include_turn_ids=True) for item in session_items]
        session_dates = [item[1].date_time for item in session_items]
        speaker_names_list = [speaker_names for _ in session_items]  # Same speakers for all sessions
        
        # Step 1: Skip session summarization and use placeholder
        logger.info(f"Skipping batch session summarization for {len(session_items)} sessions (using placeholder)")
        session_summaries = ["No summary available"] * len(session_items)
        
        # Step 2a: Batch User-side EDU extraction for all sessions (without previous summaries)
        # Note: We need to extract ONLY User EDUs without adding Assistant turns
        logger.info("Starting batch User-side EDU extraction for all sessions (without previous summaries)")
        batch_user_edu_results = self.batch_conversation_edu_extraction_without_context_user_edus_only(
            session_ids=session_ids,
            session_texts=session_texts,
            session_dates=session_dates,
            speaker_names_list=speaker_names_list,
            session_items=[item[1] for item in session_items]
        )
        
        # Step 2b: Batch Assistant-side STRUCTURED EDU extraction for all sessions (without previous summaries)
        # This extracts both simple EDUs and structured chunks
        logger.info("Starting batch Assistant-side STRUCTURED EDU extraction for all sessions (without previous summaries)")
        batch_assistant_edu_results = self.batch_conversation_edu_extraction_without_context_assistant_edus_structured(
            session_ids=session_ids,
            session_texts=session_texts,
            session_dates=session_dates,
            speaker_names_list=speaker_names_list,
            session_items=[item[1] for item in session_items]
        )
        
        # Step 2c: Combine User and Assistant EDUs/chunks
        logger.info("Combining User-side EDUs and Assistant-side EDUs/chunks")
        batch_edu_results = []
        for user_edus, assistant_items in zip(batch_user_edu_results, batch_assistant_edu_results):
            # Combine User EDUs with Assistant items (both simple EDUs and structured chunks, all as ConversationEDURawOutputWithTurnIds)
            combined = user_edus + assistant_items
            # Sort by the first turn ID in each item's source_turn_ids to maintain conversation order
            combined.sort(key=lambda item: min(item.source_turn_ids) if item.source_turn_ids else float('inf'))
            batch_edu_results.append(combined)
        
        # Step 3: Prepare data for context generation and unified event extraction
        temp_session_ids = []
        temp_session_texts = []
        temp_edu_texts = []  # Use edu_text for all (chunk_summary for chunks, edu_text for EDUs)
        temp_edu_objects = []
        session_to_edu_mapping = {}
        
        for session_idx, (session_id, edu_results) in enumerate(zip(session_ids, batch_edu_results)):
            session_to_edu_mapping[session_id] = []
            session_text = session_texts[session_idx]
            
            for edu_result in edu_results:
                temp_session_ids.append(session_id)
                temp_session_texts.append(session_text)
                
                # Use edu_text for all types (chunk_summary for chunks, edu_text for EDUs)
                # chunk_content is stored in metadata and doesn't need processing here
                temp_edu_texts.append(edu_result.edu_text)
                
                temp_edu_objects.append(edu_result)
                session_to_edu_mapping[session_id].append(len(temp_edu_texts) - 1)
        
        if not temp_edu_texts:
            logger.warning("No EDUs/chunks extracted from any session")
            return {}
        
        # Step 4: Skip batch context generation
        assert skip_edu_context_gen
        logger.info(f"Skipping context generation for {len(temp_edu_texts)} EDUs/chunks (using placeholder text)")
        context_texts = ["No context information available"] * len(temp_edu_texts)
        context_metadata = [{'prompt_tokens': 0, 'completion_tokens': 0, 'reason_tokens': 0, 'cache_hit': False}] * len(temp_edu_texts)
        
        # Step 5: Batch unified event extraction (combines detection and extraction)
        # Use edu_text for all types (chunk_summary for chunks, edu_text for EDUs)
        logger.info(f"Starting batch unified event extraction for {len(temp_edu_texts)} EDUs/chunks")
        event_extraction_results = self.batch_unified_event_extraction(
            chunk_keys=temp_session_ids,
            edus=temp_edu_texts
        )
        
        # Step 6: Combine results into output objects
        logger.info("Combining results into final output format")
        all_session_results = {}
        for session_id in session_ids:
            all_session_results[session_id] = []
        
        for i, (context_text, event_result, edu_object) in enumerate(zip(context_texts, event_extraction_results, temp_edu_objects)):
            # Create unified ConversationContextualEventRawOutputWithTurnIds for all types
            # Merge event extraction metadata with existing metadata (which contains edu_type and optionally chunk_content)
            merged_metadata = copy.deepcopy(edu_object.metadata)
            merged_metadata.update(event_result.metadata)
            
            contextual_result = ConversationContextualEventRawOutputWithTurnIds(
                session_id=edu_object.session_id,
                edu_id=edu_object.edu_id,
                edu_text=edu_object.edu_text,  # For chunks, this is chunk_summary; for EDUs, this is edu_text
                source_speakers=edu_object.source_speakers,
                date=edu_object.date,
                context_text=context_text,
                response=event_result.response,
                metadata=merged_metadata,  # Contains edu_type and chunk_content (if applicable)
                event_type=event_result.event_type,
                event_triggers=event_result.event_triggers,
                event_role_argument_pairs=event_result.event_role_argument_pairs,
                source_turn_ids=edu_object.source_turn_ids
            )
            all_session_results[edu_object.session_id].append(contextual_result)
        
        logger.info(f"Conversation OpenIE V2 with independent summaries and unified event extraction completed for {len(session_items)} sessions with {sum(len(results) for results in all_session_results.values())} total EDUs/chunks")
        
        # Combine EDUs and summaries for each session
        combined_results = {}
        for i, session_id in enumerate(session_ids):
            combined_results[session_id] = {
                'edus': all_session_results[session_id],
                'summary': session_summaries[i]
            }
        
        return combined_results