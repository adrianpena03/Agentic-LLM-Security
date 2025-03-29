from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a short horror story (5 sentences) as if you're christopher nolan"}
  ]
)

print(completion.choices[0].message)



# Load attack prompts
with open("attack_prompts.json", "r") as file:
    attack_data = json.load(file)

# Example function to test attack prompts
def test_attack_prompts(llm_function):
    results = []
    for attack in attack_data["attacks"]:
        response = llm_function(attack["prompt"])
        results.append({"id": attack["id"], "type": attack["type"], "success": response})  # Replace with success metric
    return results

# Example: Pass attack prompts into your LLM function
def dummy_llm_function(prompt):
    return "Response blocked"  # Replace with actual LLM call

results = test_attack_prompts(dummy_llm_function)
print(results)
