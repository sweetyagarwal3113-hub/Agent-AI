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
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

tools = [
    calculator,
    current_time,
    read_file,
    web_search
]

llm_with_tools = llm.bind_tools(tools)

system_prompt = "You are a helpful AI Assistant."

print("=" * 50)
print("Simple AI Agent")
print("Type exit to quit")
print("=" * 50)

chat_history = []
chat_history.append(("system", system_prompt))

tool_map = {tool.name: tool for tool in tools}

while True:
    question = input("\nYou : ")
    if question.lower() == "exit":
        break
    
    chat_history.append(("user", question))

    try:
        response = llm_with_tools.invoke(chat_history)
        chat_history.append(response)

        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_msg_id = tool_call["id"]
                
                print(f"[System] Calling tool: {tool_name}...")
                if tool_name in tool_map:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                else:
                    tool_output = f"Error: Tool {tool_name} not found."
                
                from langchain_core.messages import ToolMessage
                chat_history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_msg_id))
            
            response = llm_with_tools.invoke(chat_history)
            chat_history.append(response)

        ai_message = response.content
        print("\nAI :", ai_message)
    
    except KeyboardInterrupt:
        print("\n[Interrupted] You stopped the AI. Let's start a new question.")
        chat_history.pop()
    except Exception as e:
        print(f"\n[Error] The AI encountered an issue (e.g., API Rate Limit): {e}")
        print("Please wait a moment and try again.")
        chat_history.pop()