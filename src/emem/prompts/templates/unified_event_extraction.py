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


class EventRoleArgument(BaseModel):
    role: str = Field(..., description="Argument role")
    argument: str = Field(..., description="Argument")
    
class UnifiedEventExtraction(BaseModel):
    event_type: str = Field(..., description="Summarized event type")
    role_argument_pairs: list[EventRoleArgument]


if support_json_schema:
    unified_event_extraction_system = (
        f"Given an Elementary Discourse Unit (EDU) which describes an event, your task is to:\n"
        f"1. Summarize the event type\n"
        f"2. Extract all of the event argument roles and corresponding arguments\n\n"
        f"Requirements:\n"
        f"1. Avoid pronouns or ambiguous references in the event type, argument roles and arguments. Make sure they are informative.\n"
        f"2. Make sure the event type concisely and accurately summarizes the EDU while being general.\n"
        f"3. The extracted roles and arguments must include all the information conveyed in the input EDU."
    )
else:
    unified_event_extraction_system = (
        f"Given an Elementary Discourse Unit (EDU) which describes an event, your task is to:\n"
        f"1. Summarize the event type\n"
        f"2. Extract all of the event argument roles and corresponding arguments\n\n"
        f"Requirements:\n"
        f"1. Avoid pronouns or ambiguous references in the event type, argument roles and arguments. Make sure they are informative.\n"
        f"2. Make sure the event type concisely and accurately summarizes the EDU while being general.\n"
        f"3. The extracted roles and arguments must include all the information conveyed in the input EDU.\n"
        f"4. Make sure your final output is a valid JSON string following the JSON Schema:\n"
        f"{UnifiedEventExtraction.model_json_schema()}"
    )


one_shot_unified_event_extraction_edu = (
    'Tirzepatide, under the brand name Mounjaro, received its Food and Drug Administration (FDA) approval for Type 2 diabetes in May 2022.'
)

one_shot_unified_event_extraction_input = f"EDU: {one_shot_unified_event_extraction_edu}"

one_shot_unified_event_extraction_output_dict = {
    "event_type": "Drug Approval for Medical Condition",
    "role_argument_pairs": [
        {"role": "drug generic name", "argument": "Tirzepatide"},
        {"role": "drug brand name", "argument": "Mounjaro"},
        {"role": "approval type", "argument": "Food and Drug Administration (FDA) approval"},
        {"role": "approval authority", "argument": "Food and Drug Administration (FDA)"},
        {"role": "approved medical condition", "argument": "Type 2 diabetes"},
        {"role": "approval date", "argument": "May 2022"}
    ]
}

one_shot_unified_event_extraction_output = UnifiedEventExtraction(
    event_type=one_shot_unified_event_extraction_output_dict["event_type"],
    role_argument_pairs=[
        EventRoleArgument(role=role_arg["role"], argument=role_arg["argument"]) 
        for role_arg in one_shot_unified_event_extraction_output_dict["role_argument_pairs"]
    ]
).model_dump_json()


prompt_template = [
    {"role": "system", "content": unified_event_extraction_system},
    {"role": "user", "content": one_shot_unified_event_extraction_input},
    {"role": "assistant", "content": one_shot_unified_event_extraction_output},
    {"role": "user", "content": "EDU: ${edu}"}
]

