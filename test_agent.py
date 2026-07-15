from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import read_file
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

tools = [read_file]
system_prompt = "You are a helpful AI Assistant. Use tools to answer questions."

agent_executor = create_agent(llm, tools=tools, system_prompt=system_prompt)

print("Testing agent execution")
response = agent_executor.invoke({"messages": [("user", "What is in sample.txt?")]})
print(response)
