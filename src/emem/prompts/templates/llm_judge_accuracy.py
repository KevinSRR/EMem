"""
LLM Judge accuracy evaluation prompt template with structured output.
"""

from pydantic import BaseModel, Field
from typing import Literal


class LLMJudgeAccuracy(BaseModel):
    """Response model for LLM judge accuracy evaluation."""
    
    reasoning: str = Field(
        description="A short (one sentence) explanation of your reasoning for the judgment"
    )
    label: Literal["CORRECT", "WRONG"] = Field(
        description="The judgment label: CORRECT if the generated answer matches the gold answer, WRONG otherwise"
    )


# Template content for the LLM judge accuracy evaluation
llm_judge_accuracy_system = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question, 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something based on prior conversations between two speakers.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Respond in JSON format with your reasoning and label."""

llm_judge_accuracy_user = """Question: ${question}
Gold answer: ${gold_answer}
Generated answer: ${generated_answer}"""

prompt_template = [
    {"role": "system", "content": llm_judge_accuracy_system},
    {"role": "user", "content": llm_judge_accuracy_user}
]
