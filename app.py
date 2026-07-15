from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from dotenv import load_dotenv

from tools import calculator
from tools import current_time
from tools import read_file
from tools import web_search

load_dotenv()


llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_retries=0,
    max_tokens=1024
)

tools = [
    calculator,
    current_time,
    read_file,
    web_search
]

system_prompt = """
You are a helpful AI Assistant.

If the user asks for personal information like their name or what they are learning, you must use the read_file tool on 'sample.txt'.
"""

agent_executor = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt
)

print("=" * 50)
print("Simple AI Agent")
print("Type exit to quit")
print("=" * 50)

chat_history = []

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break
    
    chat_history.append(("user", question))

    try:
        response = agent_executor.invoke(
            {
                "messages": chat_history
            }
        )

        ai_message = response["messages"][-1].content
        print("\nAI :", ai_message)
        chat_history.append(("assistant", ai_message))
    
    except KeyboardInterrupt:
        print("\n[Interrupted] You stopped the AI. Let's start a new question.")
        # Remove the last user message since it was interrupted
        chat_history.pop()
    except Exception as e:
        print(f"\n[Error] The AI encountered an issue (e.g., API Rate Limit): {e}")
        print("Please wait a moment and try again.")
        # Remove the last user message since it failed
        chat_history.pop()