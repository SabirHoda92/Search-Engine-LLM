import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv

load_dotenv()

###


#Arxiv and Wikipedia Wrapper Tools
arix_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)

arix = ArxivQueryRun(api_wrapper=arix_wrapper)

wiki_rapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)

wiki = WikipediaQueryRun(api_wrapper=wiki_rapper)


# DuckDuckGoSearchRun is a LangChain tool that lets the agent 
# perform real-time web searches using DuckDuckGo and return results.
search = DuckDuckGoSearchRun(name="search")


st.title("Langchain - Chat with search")

#sidebar for setting
st.sidebar.title("setting")
api_key = st.sidebar.text_input("Enter Your Groq API Key:",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assisstant", "content": "Hi I am Chatbot who can search the web. How Can I Help You?"}
    ]
    

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
    
if prompt:= st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role": "user", "content":prompt})
    st.chat_message("user").write(prompt)
    
    
    llm = ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant",streaming=True)
    
    # ✅ Now initialize DuckDuckGo *inside* the app flow
    search = DuckDuckGoSearchRun(name="search")
    tools = [arix,wiki,search]
    
    
    # Initialize an agent that uses the given tools and LLM.
    # AgentType.ZERO_SHOT_REACT_DESCRIPTION means the agent decides 
    # which tool to use based only on the input query and tool descriptions 
    # (no prior examples needed).
    search_agent = initialize_agent(tools,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    
    
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content":response})
        st.write(response)
        
        
        
    
    
    







