from pydantic import BaseModel, Field


from ...utils.logging_utils import get_logger
from ...utils.config_utils import get_support_json_schema

logger = get_logger(__name__)

# Read environment variable and assign to support_json_schema
try:
    support_json_schema = get_support_json_schema() # indicate whether the LLM engine supports json schema or not. If set to False, we will need to integrate json schema into the prompt explicitly.
    if support_json_schema:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'true' - JSON schema support enabled")
    else:
        logger.info("SUPPORT_JSON_SCHEMA environment variable set to 'false' - JSON schema support disabled")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise


class QueryERExtraction(BaseModel):
    entities_and_concepts: list[str] = Field(..., description="List of important entities and concepts extracted from the query")


if support_json_schema:
    ner_system = """You're a very effective entity extraction system for questions. Extract all entities and key concepts that are important for solving the question. Entities include both named entities (e.g., people, organizations, locations, events, products) and non-named but salient objects or concepts (e.g., "train", "taxi", "hotel", "airport"). The entities should be self-complete (no abbreviations, use full names) to be easy to use for exact matching or matching with encoded embeddings. Focus on high precision — only extract clear, unambiguous entities or concepts that are central to the question."""
else:
    ner_system = f"""You're a very effective entity extraction system for questions. Extract all entities and key concepts that are important for solving the question. Entities include both named entities (e.g., people, organizations, locations, events, products) and non-named but salient objects or concepts (e.g., "train", "taxi", "hotel", "airport"). The entities should be self-complete (no abbreviations, use full names) to be easy to use for exact matching or matching with encoded embeddings. Focus on high precision — only extract clear, unambiguous entities or concepts that are central to the question.

Make sure your final output is a valid JSON string following the JSON Schema:
{QueryERExtraction.model_json_schema()}"""

query_prompt_one_shot_input = """Should I visit the Louvre Museum or the Musée d'Orsay in Paris if I'm interested in Impressionist paintings and want to take the metro instead of a taxi?"""

query_prompt_one_shot_output = QueryERExtraction(entities_and_concepts=["Louvre Museum", "Musée d'Orsay", "Paris", "Impressionist paintings", "metro", "taxi"]).model_dump_json()

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "Question: ${query}"}
]