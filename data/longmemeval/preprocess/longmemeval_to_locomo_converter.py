#!/usr/bin/env python3
"""
LongMemEval to LoCoMo Format Converter

This script converts LongMemEval dataset format to LoCoMo-compatible format
to enable reuse of existing EMem processing pipeline.

Author: Assistant
Date: 2025-09-29
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import sys

# Add the src directory to the path to import EMem modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from emem.utils.conversation_data_utils import (
    LoCoMoSample, QA, Turn, Session, Conversation, EventSummary, Observation
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def map_question_type_to_category(question_id: str, question_type: str) -> int:
    """
    Map LongMemEval question type to LoCoMo category.
    
    Args:
        question_id: Question ID from LongMemEval
        question_type: Question type from LongMemEval
        
    Returns:
        Category integer according to LoCoMo schema
    """
    # Check for adversarial questions (category 5)
    if question_id.endswith("_abs"):
        return 5
    
    # Map other question types
    if "single" in question_type:
        return 4
    elif question_type == "multi-session":
        return 1
    elif question_type == "temporal-reasoning":
        return 2
    elif question_type == "knowledge-update":
        return 3
    else:
        logger.warning(f"Unknown question type: {question_type}, defaulting to category 1")
        return 5


def convert_longmemeval_session_to_locomo(session_data: List[Dict], 
                                        session_id: int, 
                                        date_time: str) -> Session:
    """
    Convert a LongMemEval session to LoCoMo Session format.
    
    Args:
        session_data: List of turns in LongMemEval format
        session_id: Session ID (1-indexed)
        date_time: Session date-time string in LongMemEval format
        
    Returns:
        Session object in LoCoMo format
    """
    turns = []
    
    for turn_idx, turn_data in enumerate(session_data, start=1):
        # Create dia_id in LoCoMo format: "D{session_id}:{turn_id}"
        dia_id = f"D{session_id}:{turn_idx}"
        
        # Map role: LongMemEval uses "user"/"assistant", LoCoMo uses speaker names
        # We'll use generic names for now
        speaker = "User" if turn_data["role"] == "user" else "Assistant"
        
        # Get text content
        text = turn_data["content"]
        
        turns.append(Turn(
            speaker=speaker,
            dia_id=dia_id,
            text=text
        ))
    
    # Keep original LongMemEval date format - EMem will handle formatting based on date_format_type config
    return Session(
        session_id=session_id,
        date_time=date_time,
        turns=turns
    )


def convert_evidence_format(answer_session_ids: List[str], 
                          session_id_mapping: Dict[str, int]) -> List[str]:
    """
    Convert LongMemEval evidence format to LoCoMo format.
    
    Args:
        answer_session_ids: List of session IDs from LongMemEval
        session_id_mapping: Mapping from LongMemEval session IDs to LoCoMo session indices
        
    Returns:
        List of evidence strings in LoCoMo format (e.g., ["D1:3", "D2:1"])
    """
    evidence = []
    
    for session_id in answer_session_ids:
        if session_id in session_id_mapping:
            locomo_session_idx = session_id_mapping[session_id]
            # For now, we'll reference the first turn of the session
            # In a more sophisticated version, we could analyze which turn contains the evidence
            evidence.append(f"D{locomo_session_idx}:1")
        else:
            logger.warning(f"Session ID {session_id} not found in mapping")
    
    return evidence


def convert_longmemeval_sample_to_locomo(sample: Dict, sample_idx: int) -> LoCoMoSample:
    """
    Convert a single LongMemEval sample to LoCoMo format.
    
    Args:
        sample: LongMemEval sample dictionary
        sample_idx: Sample index for generating sample_id
        
    Returns:
        LoCoMoSample object
    """
    # Create sample ID
    sample_id = str(sample_idx)
    
    # Create session ID mapping (LongMemEval session IDs to 1-indexed LoCoMo session IDs)
    session_id_mapping = {}
    sessions = {}
    
    for idx, (session_id, session_data, date_time) in enumerate(
        zip(sample["haystack_session_ids"], sample["haystack_sessions"], sample["haystack_dates"]), 
        start=1
    ):
        session_id_mapping[session_id] = idx
        sessions[idx] = convert_longmemeval_session_to_locomo(session_data, idx, date_time)
    
    # Create conversation
    conversation = Conversation(
        speaker_a="User",  # Generic speaker names
        speaker_b="Assistant",
        sessions=sessions
    )
    
    # Convert question and answer
    category = map_question_type_to_category(sample["question_id"], sample["question_type"])
    
    # Keep original question text - the question_date context will be handled by EMem during QA
    question_text = sample['question']
    
    # Convert evidence format
    evidence = convert_evidence_format(sample["answer_session_ids"], session_id_mapping)
    
    qa = [QA(
        question=question_text,
        answer=sample["answer"],
        evidence=evidence,
        category=category
    )]
    
    # Create empty event summary and observation (required by LoCoMo format)
    # These would need to be generated separately if needed
    event_summary = EventSummary(events={})
    observation = Observation(observations={})
    session_summary = {}
    
    return LoCoMoSample(
        sample_id=sample_id,
        qa=qa,
        conversation=conversation,
        event_summary=event_summary,
        observation=observation,
        session_summary=session_summary
    )


def convert_longmemeval_to_locomo(input_file: str, output_file: str, max_samples: Optional[int] = None):
    """
    Convert entire LongMemEval dataset to LoCoMo format.
    
    Args:
        input_file: Path to LongMemEval JSON file
        output_file: Path to output LoCoMo JSON file
        max_samples: Maximum number of samples to convert (None for all)
    """
    logger.info(f"Loading LongMemEval data from {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        longmemeval_data = json.load(f)
    
    logger.info(f"Found {len(longmemeval_data)} samples in LongMemEval dataset")
    
    if max_samples:
        longmemeval_data = longmemeval_data[:max_samples]
        logger.info(f"Processing first {len(longmemeval_data)} samples")
    
    locomo_samples = []
    sample_question_dates = []
    for idx, sample in enumerate(longmemeval_data):
        try:
            locomo_sample = convert_longmemeval_sample_to_locomo(sample, idx)
            locomo_samples.append(locomo_sample)
            
            sample_question_dates.append(sample["question_date"])

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1} samples")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue
    
    # Convert to JSON-serializable format
    logger.info("Converting to JSON format")
    locomo_json = []
    
    for jdx, sample in enumerate(locomo_samples):
        # Convert dataclass to dict
        sample_dict = {
            "qa": [
                {
                    "question": f"Date of user query: {sample_question_dates[jdx]}\nUser: {qa.question}",
                    "answer": qa.answer,
                    "evidence": qa.evidence,
                    "category": qa.category
                }
                for qa in sample.qa
            ],
            "conversation": {
                "speaker_a": sample.conversation.speaker_a,
                "speaker_b": sample.conversation.speaker_b,
                **{
                    f"session_{session_id}": [
                        {
                            "speaker": turn.speaker,
                            "dia_id": turn.dia_id,
                            "text": turn.text
                        }
                        for turn in session.turns
                    ]
                    for session_id, session in sample.conversation.sessions.items()
                },
                **{
                    f"session_{session_id}_date_time": session.date_time
                    for session_id, session in sample.conversation.sessions.items()
                }
            },
            "event_summary": sample.event_summary.events,
            "observation": sample.observation.observations,
            "session_summary": sample.session_summary
        }
        locomo_json.append(sample_dict)
    
    # Save to output file
    logger.info(f"Saving {len(locomo_json)} converted samples to {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(locomo_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Conversion completed successfully!")
    logger.info(f"Output saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert LongMemEval dataset to LoCoMo format")
    parser.add_argument("input_file", help="Path to LongMemEval JSON file")
    parser.add_argument("output_file", help="Path to output LoCoMo JSON file")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to convert (default: all)")
    parser.add_argument("--log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run conversion
    convert_longmemeval_to_locomo(args.input_file, args.output_file, args.max_samples)


if __name__ == "__main__":
    main()
