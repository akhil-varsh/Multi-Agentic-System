"""
Multi-Agent Research System with Streamlit Interface

A beautiful Streamlit app for multi-agent research using LangGraph and Mistral AI

Requirements:
pip install streamlit langchain langchain-openai langchain-community langgraph tavily-python python-dotenv
"""

import os
import streamlit as st
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Mistral LLM
@st.cache_resource
def get_mistral_llm(temperature=0):
    return ChatOpenAI(
        model="mistral-tiny",
        temperature=temperature,
        openai_api_key=os.getenv("MISTRAL_API_KEY"),
        openai_api_base="https://api.mistral.ai/v1"
    )

# Initialize search tool
@st.cache_resource
def get_search_tool():
    return TavilySearch(max_results=3)

# Define the state structure
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    research_topic: str
    research_findings: str
    analysis: str
    final_report: str
    next_agent: str

# Agent prompts
RESEARCHER_PROMPT = """You are a research specialist. Your job is to:
1. Search for relevant information about the given topic
2. Gather facts, statistics, and credible sources
3. Summarize your findings concisely

Use the search tool to find information. Be thorough but concise.
Topic: {topic}
"""

ANALYZER_PROMPT = """You are an analytical expert. Your job is to:
1. Review the research findings
2. Identify key patterns, trends, and insights
3. Draw meaningful conclusions
4. Highlight any gaps or contradictions

Research Findings:
{findings}

Provide a structured analysis."""

WRITER_PROMPT = """You are a professional writer. Your job is to:
1. Create a comprehensive, well-structured report
2. Synthesize the research and analysis
3. Write in a clear, engaging style
4. Include an introduction, main findings, analysis, and conclusion

Research Findings:
{findings}

Analysis:
{analysis}

Create a polished final report."""

SUPERVISOR_PROMPT = """You are a supervisor managing a research team. 
Based on the current state, decide which agent should act next.

Available agents:
- researcher: Gathers information (use first)
- analyzer: Analyzes findings (use after researcher)
- writer: Creates final report (use after analyzer)
- FINISH: Complete the workflow (use after writer)

Current progress:
- Research findings: {has_findings}
- Analysis: {has_analysis}
- Final report: {has_report}

Respond with ONLY the agent name or FINISH."""

# Create agent nodes
def researcher_node(state: ResearchState, status_placeholder):
    """Research agent that searches for information"""
    status_placeholder.info("🔍 Researcher Agent: Gathering information...")
    
    llm = get_mistral_llm()
    tools = [get_search_tool()]
    
    # Create a direct prompt without MessagesPlaceholder
    system_prompt = RESEARCHER_PROMPT.format(topic=state["research_topic"])
    user_message = f"Please search for information about: {state['research_topic']}"
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Invoke with direct messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    result = llm_with_tools.invoke(messages)
    
    if result.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [result]})
        messages_with_tools = messages + [result] + tool_results["messages"]
        final_result = llm_with_tools.invoke(messages_with_tools)
        findings = final_result.content
    else:
        findings = result.content
    
    status_placeholder.success("✅ Research completed!")
    time.sleep(0.5)
    
    return {
        **state,
        "research_findings": findings,
        "messages": state["messages"] + [HumanMessage(content=f"Research completed: {findings[:200]}...")]
    }

def analyzer_node(state: ResearchState, status_placeholder):
    """Analyzer agent that analyzes research findings"""
    status_placeholder.info("📊 Analyzer Agent: Processing findings...")
    
    llm = get_mistral_llm(temperature=0.3)
    prompt = ANALYZER_PROMPT.format(findings=state["research_findings"])
    
    # Ensure we have a proper message
    messages = [HumanMessage(content=prompt)]
    result = llm.invoke(messages)
    
    status_placeholder.success("✅ Analysis completed!")
    time.sleep(0.5)
    
    return {
        **state,
        "analysis": result.content,
        "messages": state["messages"] + [HumanMessage(content=f"Analysis completed: {result.content[:200]}...")]
    }

def writer_node(state: ResearchState, status_placeholder):
    """Writer agent that creates the final report"""
    status_placeholder.info("✍️ Writer Agent: Creating final report...")
    
    llm = get_mistral_llm(temperature=0.5)
    prompt = WRITER_PROMPT.format(
        findings=state["research_findings"],
        analysis=state["analysis"]
    )
    
    # Ensure we have a proper message
    messages = [HumanMessage(content=prompt)]
    result = llm.invoke(messages)
    
    status_placeholder.success("✅ Report completed!")
    time.sleep(0.5)
    
    return {
        **state,
        "final_report": result.content,
        "messages": state["messages"] + [HumanMessage(content="Final report completed")]
    }

def supervisor_node(state: ResearchState):
    """Supervisor that decides the next agent using simple logic"""
    # Use deterministic logic instead of LLM for more reliable routing
    
    has_findings = bool(state.get("research_findings"))
    has_analysis = bool(state.get("analysis"))
    has_report = bool(state.get("final_report"))
    
    # Deterministic routing logic
    if not has_findings:
        next_agent = "researcher"
    elif not has_analysis:
        next_agent = "analyzer"
    elif not has_report:
        next_agent = "writer"
    else:
        next_agent = "FINISH"
    
    return {
        **state,
        "next_agent": next_agent
    }

def route_to_agent(state: ResearchState) -> Literal["researcher", "analyzer", "writer", "__end__"]:
    """Route to the appropriate agent based on supervisor decision"""
    next_agent = state.get("next_agent", "researcher").lower().strip()
    
    # Map supervisor decision to valid routes
    if next_agent in ["finish", "__end__", "end", "complete"]:
        return "__end__"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "analyzer":
        return "analyzer"
    elif next_agent == "writer":
        return "writer"
    else:
        # Fallback logic if supervisor gives unclear direction
        if not state.get("research_findings"):
            return "researcher"
        elif not state.get("analysis"):
            return "analyzer"
        elif not state.get("final_report"):
            return "writer"
        else:
            return "__end__"

# Build the graph with status updates
def create_research_workflow(status_placeholder):
    """Create the multi-agent research workflow"""
    workflow = StateGraph(ResearchState)
    
    # Wrapper functions to pass status_placeholder
    def researcher_wrapper(state):
        return researcher_node(state, status_placeholder)
    
    def analyzer_wrapper(state):
        return analyzer_node(state, status_placeholder)
    
    def writer_wrapper(state):
        return writer_node(state, status_placeholder)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("analyzer", analyzer_wrapper)
    workflow.add_node("writer", writer_wrapper)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "analyzer": "analyzer",
            "writer": "writer",
            "__end__": END
        }
    )
    
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyzer", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    return workflow.compile()

# Streamlit App
def main():
    st.title("🔬 AI Multi-Agent Research Assistant")
    st.markdown("### Powered by Mistral AI & LangGraph")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mistral_key = st.text_input(
            "Mistral API Key",
            type="password",
            value=os.getenv("MISTRAL_API_KEY", ""),
            help="Get your API key from console.mistral.ai"
        )
        
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            help="Get your free API key from tavily.com"
        )
        
        if mistral_key:
            os.environ["MISTRAL_API_KEY"] = mistral_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This multi-agent system uses:
        - 🔍 **Researcher**: Gathers information
        - 📊 **Analyzer**: Analyzes findings
        - ✍️ **Writer**: Creates reports
        - 👔 **Supervisor**: Coordinates workflow
        """)
    
    # Main content
    st.markdown("---")
    
    # Research topic input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_topic = st.text_input(
            "Enter Research Topic",
            placeholder="e.g., Recent developments in quantum computing",
            help="Enter a topic you want to research in depth"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_research = st.button("🚀 Start Research", use_container_width=True)
    
    # Check if API keys are provided
    if start_research:
        if not mistral_key or not tavily_key:
            st.error("⚠️ Please provide both Mistral and Tavily API keys in the sidebar!")
            return
        
        if not research_topic:
            st.warning("⚠️ Please enter a research topic!")
            return
        
        # Initialize session state
        if 'research_complete' not in st.session_state:
            st.session_state.research_complete = False
        
        # Status placeholder
        status_placeholder = st.empty()
        
        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            with st.spinner("Initializing research workflow..."):
                progress_bar.progress(10)
                progress_text.text("Initializing agents...")
                
                app = create_research_workflow(status_placeholder)
                
                initial_state = {
                    "messages": [HumanMessage(content=f"Research the following topic: {research_topic}")],
                    "research_topic": research_topic,
                    "research_findings": "",
                    "analysis": "",
                    "final_report": "",
                    "next_agent": "researcher"
                }
                
                progress_bar.progress(30)
                progress_text.text("Running research workflow...")
                
                # Run the workflow with recursion limit
                final_state = app.invoke(
                    initial_state,
                    config={"recursion_limit": 50}
                )
                
                progress_bar.progress(100)
                progress_text.text("Research complete!")
                time.sleep(0.5)
                progress_bar.empty()
                progress_text.empty()
                status_placeholder.empty()
                
                # Display results in tabs
                st.markdown("---")
                st.success("✅ Research completed successfully!")
                
                tab1, tab2, tab3 = st.tabs(["📝 Final Report", "🔍 Research Findings", "📊 Analysis"])
                
                with tab1:
                    st.markdown("### Final Report")
                    st.markdown(final_state.get("final_report", "No report generated"))
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=final_state.get("final_report", ""),
                        file_name=f"research_report_{research_topic.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                with tab2:
                    st.markdown("### Research Findings")
                    st.markdown(final_state.get("research_findings", "No findings"))
                
                with tab3:
                    st.markdown("### Analysis")
                    st.markdown(final_state.get("analysis", "No analysis"))
                
                # Store in session state
                st.session_state.last_research = final_state
                st.session_state.research_complete = True
                
        except Exception as e:
            st.error(f"❌ Error during research: {str(e)}")
            st.exception(e)
    
    # Show example topics
    if not start_research:
        st.markdown("---")
        st.markdown("### 💡 Example Research Topics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🤖 AI and Machine Learning trends in 2025")
        with col2:
            st.info("🌍 Impact of climate change on global agriculture")
        with col3:
            st.info("💊 Recent breakthroughs in cancer research")

if __name__ == "__main__":
    main()