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
st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 Simple AI Agent")


# Initialize tools and LLM
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        model="llama3-groq-70b-8192-tool-use-preview",
        temperature=0
    )

    tools = [
        calculator,
        current_time,
        read_file,
        web_search
    ]

    system_prompt = """
You are a helpful AI Assistant.

Whenever possible use available tools.

Do not guess answers if a tool can help.
"""

    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )

agent_executor = get_agent()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Convert session state messages to format expected by agent
    agent_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            agent_messages.append(("user", msg["content"]))
        else:
            agent_messages.append(("assistant", msg["content"]))

    with st.spinner("Thinking..."):
        # Invoke agent
        response = agent_executor.invoke(
            {
                "messages": agent_messages
            }
        )
        
        # Get final response content
        ai_response = response["messages"][-1].content
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
