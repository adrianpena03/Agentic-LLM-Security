import json
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load DeepSeek Model
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Wrap the model in a LangChain-compatible format
llm = HuggingFacePipeline(pipeline("text-generation", model=model, tokenizer=tokenizer))

# Load Attack Prompts
with open("attack_prompts.json", "r") as f:
    attack_prompts = json.load(f)

# Define attack function
def attack_model():
    prompt = attack_prompts["jailbreaks"][0]  # Pick first attack (you can randomize)
    response = llm(prompt)
    return f"Attack: {prompt}\nResponse: {response}"

# Define defense function
def apply_defense():
    return "Applied defense: Input filtering enabled."

# Register tools
tools = [
    Tool(name="Attack LLM", func=attack_model, description="Launches an attack on the model."),
    Tool(name="Apply Defense", func=apply_defense, description="Adds a security measure to block attacks."),
]

# Create an agent that launches attacks & applies defenses
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Lets the agent choose actions
    verbose=True
)

# Run the agent
agent.run("Try to jailbreak the LLM, then apply defenses.")
