from pydantic import BaseModel, Field
from typing import List

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


class ConversationEDUV1(BaseModel):
    edu_text: str = Field(..., description="Content of the extracted EDU - should be self-contained and informative")
    source_turn_ids: List[int] = Field(..., description="List of turn IDs (integers) from which this EDU was extracted")


class ConversationEDUExtractionV1(BaseModel):
    edus: List[ConversationEDUV1]


if support_json_schema:
    conversation_edu_extraction_v1_system = (
        "Given a conversation session between speakers with numbered turns, your task is to decompose it into Elementary Discourse Units (EDUs) "
        "- short spans of text that are minimal yet complete in meaning. Each EDU should express a single fact, event, or "
        "proposition and be atomic (not easily divisible further while still making sense). "
        "It is important that you preserve all information from the conversation - no detail should be lost in the extraction process."
        "\n"
        "Requirements for Conversation EDUs with Turn Attribution:\n"
        "1. Each EDU should be a self-contained unit of meaning that can be understood independently. It should not depend on any other EDU for understanding, although it may relate to it\n"
        "2. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "3. The extracted EDUs must include all the information conveyed in the current conversation session. The extracted EDUs should collectively capture everything discussed\n"
        "4. For each EDU, you must provide the source_turn_ids field containing a list of turn ID integers from which the EDU was extracted or referenced (e.g., [1], [3, 5], etc.)\n"
        "5. EDUs can span multiple turns if they represent the same factual unit - in such cases, include all relevant turn IDs\n"
        "6. Focus on extracting facts, events, and substantive information rather than conversational pleasantries\n"
        "7. Infer and add complete temporal context where needed for clarity\n"
        "8. Pay attention to capturing all details, facts, decisions, concerns, and substantive information from all speakers"
    )
else:
    conversation_edu_extraction_v1_system = (
        "Given a conversation session between speakers with numbered turns, your task is to decompose it into Elementary Discourse Units (EDUs) "
        "- short spans of text that are minimal yet complete in meaning. Each EDU should express a single fact, event, or "
        "proposition and be atomic (not easily divisible further while still making sense). "
        "It is important that you preserve all information from the conversation - no detail should be lost in the extraction process."
        "\n"
        "Requirements for Conversation EDUs with Turn Attribution:\n"
        "1. Each EDU should be a self-contained unit of meaning that can be understood independently. It should not depend on any other EDU for understanding, although it may relate to it\n"
        "2. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
        "3. The extracted EDUs must include all the information conveyed in the current conversation session. The extracted EDUs should collectively capture everything discussed\n"
        "4. For each EDU, you must provide the source_turn_ids field containing a list of turn ID integers from which the EDU was extracted or referenced (e.g., [1], [3, 5], etc.)\n"
        "5. EDUs can span multiple turns if they represent the same factual unit - in such cases, include all relevant turn IDs\n"
        "6. Focus on extracting facts, events, and substantive information rather than conversational pleasantries\n"
        "7. Infer and add complete temporal context where needed for clarity\n"
        "8. Pay attention to capturing all details, facts, decisions, concerns, and substantive information from all speakers\n"
        f"9. Make sure your final output is a valid JSON string following the JSON Schema:\n"
        f"{ConversationEDUExtractionV1.model_json_schema()}"
    )


# One-shot example for conversation EDU extraction with turn attribution
one_shot_conversation_session_v1 = """Date: 2:30 pm on 15 March, 2024

Turn 1:
Alice: Hey Bob! How was your trip to Tokyo?

Turn 2:
Bob: It was amazing! I spent 5 days there for the Global AI Innovation Symposium 2024. The conference at Tokyo University was incredible.

Turn 3:
Alice: That sounds exciting! What was the main focus?

Turn 4:
Bob: They had sessions on large language models and robotics. I presented our recent work on multimodal learning.

Turn 5:
Alice: How did it go?

Turn 6:
Bob: Really well! We got great feedback and Dr. Yamamoto from Sony AI wants to collaborate on our next project.

Turn 7:
Alice: That's fantastic! When are you planning to start that collaboration? Also, I remember our department head mentioned we should prioritize industry partnerships this quarter.

Turn 8:
Bob: We're aiming for next month. He said they have a $2 million budget for joint research.

Turn 9:
Alice: Wow, that's significant! That would definitely meet our Q2 funding targets. What about your flight back?

Turn 10:
Bob: Flight was delayed by 3 hours due to weather, but I made it back safely on March 14th evening.

Turn 11:
Alice: At least you're back safely. I've scheduled a team meeting for Monday to discuss how we can support this collaboration."""

one_shot_conversation_input_v1 = f"Session conversation:\n{one_shot_conversation_session_v1}"

one_shot_conversation_edus_v1 = [
    ConversationEDUV1(
        edu_text="Bob traveled to Tokyo for 5 days to attend the Global AI Innovation Symposium 2024 in March 2024",
        source_turn_ids=[2]
    ),
    ConversationEDUV1(
        edu_text="The Global AI Innovation Symposium 2024 was held at Tokyo University in Tokyo",
        source_turn_ids=[2]
    ),
    ConversationEDUV1(
        edu_text="The Global AI Innovation Symposium 2024 included sessions on large language models and robotics",
        source_turn_ids=[4]
    ),
    ConversationEDUV1(
        edu_text="Bob presented his team's recent work on multimodal learning at the Global AI Innovation Symposium 2024",
        source_turn_ids=[4]
    ),
    ConversationEDUV1(
        edu_text="Bob's presentation on multimodal learning at the Global AI Innovation Symposium 2024 received great feedback",
        source_turn_ids=[6]
    ),
    ConversationEDUV1(
        edu_text="Dr. Yamamoto from Sony AI expressed interest in collaborating with Bob on a joint research project after his presentation at the Global AI Innovation Symposium 2024",
        source_turn_ids=[4, 6]
    ),
    ConversationEDUV1(
        edu_text="Alice and Bob's department head mentioned prioritizing industry partnerships in Q2 2024",
        source_turn_ids=[7]
    ),
    ConversationEDUV1(
        edu_text="Bob and Dr. Yamamoto from Sony AI plan to start the Bob-Sony AI collaboration project in April 2024",
        source_turn_ids=[8]
    ),
    ConversationEDUV1(
        edu_text="Dr. Yamamoto from Sony AI has a $2 million budget for the Bob-Sony AI collaboration project",
        source_turn_ids=[8]
    ),
    ConversationEDUV1(
        edu_text="The Bob-Sony AI collaboration project would meet the Q2 2024 funding targets of Bob and Alice's department",
        source_turn_ids=[9]
    ),
    ConversationEDUV1(
        edu_text="Bob's return flight from Tokyo in March 2024 was delayed by 3 hours due to weather",
        source_turn_ids=[10]
    ),
    ConversationEDUV1(
        edu_text="Bob arrived safely back from Tokyo on March 14th, 2024 evening despite the flight delay",
        source_turn_ids=[10]
    ),
    ConversationEDUV1(
        edu_text="Alice scheduled a team meeting for Monday March 18th, 2024 to discuss supporting the Bob-Sony AI collaboration project",
        source_turn_ids=[11]
    )
]

one_shot_conversation_output_v1 = ConversationEDUExtractionV1(edus=one_shot_conversation_edus_v1).model_dump_json()


prompt_template = [
    {"role": "system", "content": conversation_edu_extraction_v1_system},
    {"role": "user", "content": one_shot_conversation_input_v1 + "\n\nSpeaker names: Alice, Bob"},
    {"role": "assistant", "content": one_shot_conversation_output_v1},
    {"role": "user", "content": "Session conversation:\n${session_text}\n\nSpeaker names: ${speaker_names}"}
]

