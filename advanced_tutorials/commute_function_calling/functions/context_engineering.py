import xml.etree.ElementTree as ET
import re
import inspect
from typing import get_type_hints
import json
import datetime
import torch
import sys
import pandas as pd
from openai import OpenAI
from functions.transport_functions import predict_late_departure_at_scheduled_time, get_departures_history_in_date_range
from typing import Any, Dict, List


def get_type_name(t: Any) -> str:
    """Get the name of the type."""
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def serialize_function_to_json(func: Any) -> str:
    """Serialize a function to JSON."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": type_hints.get('return', 'void').__name__
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def get_function_calling_prompt(user_query):
    fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
    example_1 = """{"name": "get_departures_history_in_date_range", "arguments": {"date_start": "2024-01-10", "date_end": "2024-01-14"}}"""
    example_2 = """{"name": "predict_late_departure_at_scheduled_time", "arguments": {"scheduled_time": "2024-09-18"}}"""

    prompt = f"""<|im_start|>system
You are a helpful assistant with access to the following functions:

{serialize_function_to_json(predict_late_departure_at_scheduled_time)}

{serialize_function_to_json(get_departures_history_in_date_range)}

###INSTRUCTIONS:
- You need to choose one function to use and retrieve paramenters for this function from the user input.
- If the user query contains 'will' or 'tomorrow', it is very likely that you will need to use the predict_late_departure_at_scheduled_time function.
- Do not include feature_view and model_deployment.
- Provide dates STRICTLY in the YYYY-MM-DD format.
- Generate an 'No Function needed' string if the user query does not require function calling.

IMPORTANT: Today is {datetime.date.today().strftime("%A")}, {datetime.date.today()}.

To use one of there functions respond STRICTLY with:
<onefunctioncall>
    <functioncall> {fn} </functioncall>
</onefunctioncall>

###EXAMPLES

EXAMPLE 1:
- User: Hi! How are you?
- AI Assistant: No Function needed.

EXAMPLE 2:
- User: Is the public transportation in Stockholm efficient?
- AI Assistant: No Function needed.

EXAMPLE 3:
- User: How many departures were late from 2024-09-17 till 2024-09-18?
- AI Assistant:
<onefunctioncall>
    <functioncall> {example_1} </functioncall>
</onefunctioncall>

EXAMPLE 4:
- User: What is the probability that a departure scheduled today is late?
- AI Assistant:
<onefunctioncall>
    <functioncall> {example_2} </functioncall>
</onefunctioncall>
<|im_end|>

<|im_start|>user
{user_query}
<|im_end|>

<|im_start|>assistant"""
    
    return prompt


def generate_hermes(user_query: str, model_llm, tokenizer) -> str:
    """Retrieves a function name and extracts function parameters based on the user query."""

    prompt = get_function_calling_prompt(user_query)
    
    tokens = tokenizer(prompt, return_tensors="pt").to(model_llm.device)
    input_size = tokens.input_ids.numel()
    with torch.inference_mode():
        generated_tokens = model_llm.generate(
            **tokens, 
            use_cache=True, 
            do_sample=True, 
            temperature=0.2, 
            top_p=1.0, 
            top_k=0, 
            max_new_tokens=512, 
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        generated_tokens.squeeze()[input_size:], 
        skip_special_tokens=True,
    )


def function_calling_with_openai(user_query: str, client) -> str:
    pass    


def extract_function_calls(completion: str) -> List[Dict[str, Any]]:
    """Extract function calls from completion."""    
    completion = completion.strip()
    pattern = r"(<onefunctioncall>(.*?)</onefunctioncall>)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        return None
    
    multiplefn = match.group(1)
    root = ET.fromstring(multiplefn)
    functions = root.findall("functioncall")

    return [json.loads(fn.text) for fn in functions]

def invoke_function(function, feature_view, model_deployment) -> pd.DataFrame:
    """Invoke a function with given arguments."""
    # Extract function name and arguments from input_data
    function_name = function['name']
    arguments = function['arguments']

    # Using Python's getattr function to dynamically call the function by its name and passing the arguments
    function_output = getattr(sys.modules[__name__], function_name)(
        **arguments, 
        feature_view=feature_view,
        model_deployment=model_deployment,
    )
    
    if type(function_output) == str:
        return function_output
    
    # Round the 'pm2_5' value to 2 decimal places
#     function_output['pm2_5'] = function_output['pm2_5'].apply(round, ndigits=2)
    return function_output


def get_context_data(user_query: str, feature_view, model_deployment, model_llm=None, tokenizer=None, client=None) -> str:
    """
    Retrieve context data based on user query.

    Args:
        user_query (str): The user query.
        feature_view: Feature View for data retrieval.
        model_deployment: Model deployment.
        model_llm: The language model.
        tokenizer: The tokenizer.
        client: Optionally, OpenAI client

    Returns:
        str: The context data.
    """
    
    if client:
        # Generate a response using LLM
        completion = function_calling_with_openai(user_query, client) 
    else:
        # Generate a response using LLM
        completion = generate_hermes(
            user_query, 
            model_llm, 
            tokenizer,
        )

    # Extract function calls from the completion
    functions = extract_function_calls(completion)

    # If function calls were found
    if functions:
        
        # Invoke the function with provided arguments
        data = invoke_function(functions[0], feature_view, model_deployment)
        
        # Return formatted data as string
        if isinstance(data, pd.DataFrame):
            if len(data) == 0:
                return f'Not information found about Departures on this date range'
            elif len(data) == 1:
                if "lateness_probability" in data.columns:
                    return f'Lateness prediction by {functions[0]["arguments"]["scheduled_time"]}:\n' + '\n'.join(
                        [f'Scheduled time for departures: {row["scheduled_time"]}; Likelihood of departures being late: {row["lateness_probability"]}' for _, row in data.iterrows()]
                    )
                else:
                    return f'Departures information on the {functions[0]["arguments"]["date_start"]}:\n' + '\n'.join(
                        [f'Date: {row["scheduled"]}; Expected: {row["expected"]}; Number of issues: {row["issues_count"]}; Number of late departures: {row["late_count"]}; Number of deviations: {row["deviations_count"]}; Deviations severity: {row["deviations_severity"]};' for _, row in data.iterrows()]
                    ) 
            else:
                return f'Departures information between {functions[0]["arguments"]["date_start"]} and {functions[0]["arguments"]["date_end"]}:\n' + '\n'.join(
                    [f'Date: {row["scheduled"]}; Expected: {row["expected"]}; Number of issues: {row["issues_count"]}; Number of late departures: {row["late_count"]}; Number of deviations: {row["deviations_count"]}; Deviations severity: {row["deviations_severity"]};' for _, row in data.iterrows()]
                )

        # Return message if data is not updated
        return data
    
    # If no function calls were found, return an empty string
    return ''