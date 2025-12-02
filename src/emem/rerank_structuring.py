import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_structuring_default_prompt import best_dspy_structuring_prompt

class EDU(BaseModel):
    edu: list[str] = Field(description="A list of EDUs (Elementary Discourse Units), each EDU is a string sentence")


class DSPyFilterStructuring:
    def __init__(self, emem):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages for EDU filtering.

        Parameters:
        emem : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = getattr(emem.global_config, 'rerank_dspy_file_path', None)
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## edu_before_filter ## ]]\n{edu_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## edu_after_filter ## ]]` (must be formatted as a valid Python EDU), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## edu_after_filter ## ]]\n{edu_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = emem.llm_model.infer
        self.model_name = emem.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_structuring_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], edu_before_filter=demo["edu_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(edu_after_filter=demo["edu_after_filter"])})
        return message_template

    def parse_filter(self, response):
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "edu_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(EDU).validate_python(parsed_value).edu
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, edu_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, edu_before_filter=edu_before_filter)})
        # call openai

        self.default_gen_kwargs['max_completion_tokens'] = 4096 # 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        # Properly handle the tuple response from BatchCacheOpenAI.infer()
        # which returns (response_str, metadata, cache_hit)
        if isinstance(response, tuple) and len(response) >= 2:
            return response[0]  # Return only the response string
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[str],
               candidate_indices: List[int],
               len_after_rerank: int = None) -> Tuple[List[int], List[str], dict]:
        """
        Rerank candidate EDUs based on their relevance to the query.
        
        Args:
            query: The search query
            candidate_items: List of candidate EDU strings 
            candidate_indices: List of indices corresponding to candidate_items
            len_after_rerank: Maximum number of EDUs to return after reranking
            
        Returns:
            Tuple of (sorted_candidate_indices, sorted_candidate_items, confidence_dict)
        """
        edu_before_filter = {"edu": candidate_items}
        try:
            response = self.llm_call(query, json.dumps(edu_before_filter))
            generated_edus = self.parse_filter(response)
        except Exception as e:
            print('exception', e)
            generated_edus = []
        
        result_indices = []
        for generated_edu in generated_edus:
            # Find the closest match among the candidate items
            closest_matched_edu = difflib.get_close_matches(str(generated_edu), [str(i) for i in candidate_items], n=1, cutoff=0.0)
            if closest_matched_edu:
                try:
                    result_indices.append(candidate_items.index(closest_matched_edu[0]))
                except Exception as e:
                    print('result_indices exception', e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices if i < len(candidate_indices)]
        sorted_candidate_items = [candidate_items[i] for i in result_indices if i < len(candidate_items)]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}
