# ================= INSTALL =================
# Run separately in notebook
# %pip install numpy matplotlib ibm-watsonx-ai

# ================= IMPORTS =================
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    DecodingMethods,
)

# ================= LOAD DATA =================
file_path = "California-Culinary-Map.txt"

with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(data[:100])

# ================= SPLIT =================
restaurant_list = data.split("\n\n")
restaurant_list = restaurant_list[1:]

print("Total restaurants:", len(restaurant_list))
print("First item:\n", restaurant_list[0])

# ================= LLM =================
def llm_model(system_msg, prompt_txt):

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    model = ModelInference(
        model_id="ibm/granite-4-h-small",
        credentials=credentials,
        project_id="skills-network"
    )

    full_prompt = system_msg + "\n\n" + prompt_txt

    response = model.generate(
        prompt=full_prompt,
        params={
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: 500
        }
    )

    return response["results"][0]["generated_text"]

# ================= PROMPT =================
EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1]

EXAMPLE_OUTPUT = """
{
"name": "Mar de Cortez",
"location": "Santa Monica",
"type": "casual taqueria",
"food_style": "Baja-style seafood",
"rating": 4.2,
"price_range": 1,
"signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
"vibe": "salt-air energy",
"environment": "a premier sun-drenched spot for open-air dining near the pier.",
"shortcomings": []
}
"""

def restaurant_data_structure_prompt_generation(restaurant_paragraph):

    system_msg = """
You are a data extraction assistant.
Return ONLY valid JSON.
"""

    user_prompt = f"""
Extract structured restaurant data into JSON.

Fields:
name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings

Rules:
- price_range = count of $
- signatures must be list
- shortcomings must be list

Restaurant:
{restaurant_paragraph}

Example:
{EXAMPLE_RESTAURANT_PARAGRAPH}

Output:
{EXAMPLE_OUTPUT}
"""

    return system_msg, user_prompt

# ================= VALIDATION =================
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float]
    price_range: Optional[int]
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str]
    environment: str
    shortcomings: List[str] = Field(default_factory=list)

# ================= JSON REPAIR =================
def JSON_auto_repair_prompts(candidate_json_output, error_message):

    sys_msg = "Fix JSON strictly."

    prompt = f"""
Invalid JSON:
{candidate_json_output}

Error:
{error_message}

Return ONLY valid JSON.
"""

    return sys_msg, prompt

# ================= MAIN LOOP =================
structured_restaurant_lists = []

for i, restaurant_paragraph in enumerate(restaurant_list):

    system_msg, prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)
    response = llm_model(system_msg, prompt)

    while True:
        try:
            Restaurant.model_validate_json(response)
            break
        except ValidationError as e:
            sys_msg, repair_prompt = JSON_auto_repair_prompts(response, e)
            response = llm_model(sys_msg, repair_prompt)

    structured_restaurant_lists.append(response)

    if (i+1) % 20 == 0:
        print(f"{i+1} done")

print("ALL DONE")

# ================= SCREENSHOT STEP =================
print(structured_restaurant_lists[49])

# ================= SAVE =================
structured_restaurant_lists_json = []

for response in structured_restaurant_lists:
    try:
        structured_restaurant_lists_json.append(json.loads(response))
    except:
        structured_restaurant_lists_json.append({})

for i, response in enumerate(structured_restaurant_lists_json):
    response["itemId"] = 1000001 + i

with open("structured_restaurant_data.json", "w") as f:
    json.dump(structured_restaurant_lists_json, f, indent=4)

print("Saved successfully")
