import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from tools import calculator
from tools import current_time
from tools import read_file
from tools import web_search


load_dotenv()


# Streamlit page config
# Streamlit page config
st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 Simple AI Agent")

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# Initialize tools and LLM
@st.cache_resource
def get_agent():
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

    return llm.bind_tools(tools)

llm_with_tools = get_agent()
tool_map = {
    "calculator": calculator,
    "current_time": current_time,
    "read_file": read_file,
    "web_search": web_search
}


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful AI Assistant."),
        AIMessage(content="Hello! I am your AI assistant. How can I help you today?")
    ]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    if isinstance(msg, ToolMessage):
        continue  # Don't show raw tool outputs as chat bubbles
    
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    # Only render if there's actual text content to show
    if msg.content and str(msg.content).strip():
        with st.chat_message(role):
            st.markdown(str(msg.content))

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("Thinking..."):
        try:
            # First invocation
            response = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)
            
            # Loop for tool calls
            while response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_msg_id = tool_call["id"]
                    
                    if tool_name in tool_map:
                        tool_output = tool_map[tool_name].invoke(tool_args)
                    else:
                        tool_output = f"Error: Tool {tool_name} not found."
                    
                    tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_msg_id)
                    st.session_state.messages.append(tool_msg)
                
                # Re-invoke after tool responses
                response = llm_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(response)
                
            ai_response = response.content
            
        except Exception as e:
            ai_response = f"[Error] The AI encountered an issue: {e}"
            st.session_state.messages.append(AIMessage(content=ai_response))

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_response)

