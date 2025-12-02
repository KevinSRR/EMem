from pydantic import BaseModel, Field
from typing import List
from string import Template

from ...utils.logging_utils import get_logger
from ...utils.config_utils import get_support_json_schema

logger = get_logger(__name__)

# Read environment variable and assign to support_json_schema
try:
    support_json_schema = get_support_json_schema()
    if support_json_schema:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'true' - JSON schema support enabled")
    else:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'false' - JSON schema support disabled")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise


class ConversationQAResponse(BaseModel):
    """Pydantic model for structured conversation QA response"""
    final_answer: str = Field(..., description="The final answer to the question based on the information provided")
    

class ConversationQAThoughtResponse(BaseModel):
    """Pydantic model for structured conversation QA response with reasoning"""
    thought: str = Field(..., description="Reasoning process for arriving at the answer")
    final_answer: str = Field(..., description="The final answer to the question based on the information provided")


if support_json_schema:
    conversation_qa_system = (
        "You are an intelligent memory assistant tasked with answering the given question by leveraging the retrieved conversation memories. "
        "Think step-by-step through your reasoning process before providing your final answer."
    )
else:
    conversation_qa_system = (
        "You are an intelligent memory assistant tasked with answering the given question by leveraging the retrieved conversation memories. "
        "Think step-by-step through your reasoning process before providing your final answer.\n"
        f"Make sure your final output is a valid JSON string following the JSON Schema:\n"
        f"{ConversationQAThoughtResponse.model_json_schema()}"
    )


conversation_qa_user = """# CONTEXT:

You will be provided with memories extracted from a conversation${date_span_info}${speaker_info}. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:

Before answering, you must first reason through the problem step-by-step in the "thought" field, then provide your final answer.

**In your reasoning (thought field), explicitly:**
1. Identify which memories are relevant to the question
2. Extract key information from each relevant memory (facts, dates, numbers, details)
3. If multiple memories are needed, explain how they connect
4. If calculations are required (arithmetic or temporal), show your work step-by-step
5. For temporal questions, convert any relative time references (e.g., "last year", "two months ago") to absolute dates using the memory's timestamp
6. Resolve any contradictions (prioritize more recent memories)
7. Explain your reasoning before stating the answer

**Guidelines:**
- Carefully analyze all provided memories for relevant information
- Pay attention to timestamps, speakers, and contextual details
- For factual questions, look for direct evidence in the provided memories
- For open-domain questions requiring reasoning or external knowledge, integrate provided information with your knowledge of related topics
- Verify arithmetic calculations carefully
- Convert relative time references to specific dates, months, or years. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021
- The final answer should typically be concise (e.g. less than 5-6 words), but you may provide slightly longer answers when necessary
- Avoid vague time references in your final answer

Memories:

${context}

Question: ${question}"""


prompt_template = [
    {"role": "system", "content": conversation_qa_system},
    {"role": "user", "content": conversation_qa_user}
]
