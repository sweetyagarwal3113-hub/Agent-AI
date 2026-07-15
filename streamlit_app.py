import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

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
        {"role": "system", "content": "You are a helpful AI Assistant."},
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # We need to construct the message history for LangChain
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    
    lc_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls", [])))
        elif msg["role"] == "tool":
            lc_messages.append(ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"]))

    with st.spinner("Thinking..."):
        try:
            # First invocation
            response = llm_with_tools.invoke(lc_messages)
            
            # Loop for tool calls
            while response.tool_calls:
                lc_messages.append(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_msg_id = tool_call["id"]
                    
                    if tool_name in tool_map:
                        tool_output = tool_map[tool_name].invoke(tool_args)
                    else:
                        tool_output = f"Error: Tool {tool_name} not found."
                    
                    tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_msg_id)
                    lc_messages.append(tool_msg)
                    st.session_state.messages.append({
                        "role": "tool",
                        "content": str(tool_output),
                        "tool_call_id": tool_msg_id
                    })
                
                # Re-invoke after tool responses
                response = llm_with_tools.invoke(lc_messages)
                
            ai_response = response.content
            
        except Exception as e:
            ai_response = f"[Error] The AI encountered an issue: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
