import os
from typing import Dict, List, Any, TypedDict, Optional
import json
import sqlite3
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pdfplumber
from dotenv import load_dotenv
from spotify_data import get_top_artists, get_top_tracks
from user_data import get_user_data 


# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.graph import StateGraph
import streamlit as st

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Spotify client (reusing your existing code)
# def init_spotify():
#     CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
#     CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    
#     sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#         client_id=CLIENT_ID,
#         client_secret=CLIENT_SECRET,
#         redirect_uri="http://127.0.0.1:5000/callback",
#         scope="user-top-read user-library-read"
#     ))
#     return sp

# Resume extraction function (from your resume_parser.py)
def extract_resume_info(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join([page.extract_text() or '' for page in pdf.pages])
    return text

def format_resume_text(text):
    lines = text.split("\n")
    formatted_resume = []
    
    section_titles = ["Summary", "Strengths", "Experience", "Education"]
    education_keywords = ["Bachelor", "Master", "PhD", "University", "College", "Diploma", "Dean's List", "Minor in"]
    
    experience_buffer = []
    education_buffer = []
    
    current_section = None
    added_sections = set()

    for line in lines:
        line = line.strip().replace("", "-")  # Replace bullet points with dashes
        if not line:
            continue

        # Detect section headers
        if any(line.lower().startswith(title.lower()) for title in section_titles):
            if line not in added_sections:
                formatted_resume.append(f"\n## {line} ##\n")
                added_sections.add(line)
            current_section = line
            continue

        # Remove redundant "Experience" and "Education" between sections
        if current_section in ["Experience", "Education"] and line in section_titles:
            continue

        # Ensure TELUS Communications Inc. stays on one line
        if "TELUS" in line and "Communications Inc." in line:
            formatted_resume.append("TELUS Communications Inc. - 2 yrs 11 mos")
            continue

        # Identify and move misplaced education details
        if any(keyword in line for keyword in education_keywords):
            education_buffer.append(line)
        elif current_section == "Experience":
            experience_buffer.append(line)
        else:
            formatted_resume.append(line)

    # Ensure sections appear only once and in order
    if experience_buffer and "Experience" not in added_sections:
        formatted_resume.append("\n## Experience ##\n" + "\n".join(experience_buffer))
    if education_buffer and "Education" not in added_sections:
        formatted_resume.append("\n## Education ##\n" + "\n".join(education_buffer))

    return "\n".join(formatted_resume)

# User data function (from your user_data.py)
# def get_user_data():
#     conn = sqlite3.connect("user_data.db")
#     cursor = conn.cursor()

#     # Fetch user data
#     cursor.execute("SELECT * FROM users WHERE full_name = ?", ("Cem Kaspi",))
#     row = cursor.fetchone()

#     # Get column names
#     column_names = [description[0] for description in cursor.description]

#     # Convert to dictionary
#     user_data = dict(zip(column_names, row)) if row else {}

#     conn.close()
#     return user_data

# Function to prepare resume for vector storage
def prepare_resume_vectorstore():
    pdf_path =  r"C:\Users\cemka\OneDrive\Desktop\HowToCem\cem-info\Cem_Kaspi_Resume.pdf"
    resume_text = extract_resume_info(pdf_path)
    formatted_resume = format_resume_text(resume_text)
    
    # Split text for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(formatted_resume)
    
    # Create documents for the vectorstore
    from langchain_core.documents import Document
    documents = [Document(page_content=t) for t in texts]
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# Tool definitions for LangGraph
@tool
def get_resume_info(query: str) -> str:
    """Search through the resume for relevant information."""
    # Create or load the vectorstore
    try:
        vectorstore = prepare_resume_vectorstore()
        docs = vectorstore.similarity_search(query, k=2)
        
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        else:
            return "No relevant information found in the resume."
    except Exception as e:
        return f"Error searching resume: {str(e)}"

@tool
def get_personal_info(info_type: str = None) -> str:
    """Fetch personal information from the user database and return all available data."""
    try:
        user_data = json.loads(get_user_data())  # Retrieve user data as a dictionary

        if not user_data:
            return "I couldn't find any personal information."

        # If info_type is specified, return only that field (optional feature)
        # if info_type and info_type.lower() in user_data:
        #     return f"{info_type.capitalize()}: {user_data[info_type.lower()]}"
        
        # Convert the full dictionary into a readable string format
        return "\n".join(f"{key.capitalize()}: {value}" for key, value in user_data.items())

    except Exception as e:
        return f"Error retrieving personal info: {str(e)}"

# def get_personal_info(info_type: str = None) -> Dict[str, Any]:
#     """Get personal information from the user database."""
#     try:
#         user_data = json.loads(get_user_data())
        
#         if info_type and info_type.lower() in user_data:
#             return {info_type.lower(): user_data[info_type.lower()]}
#         else:
#             # Return a subset of non-sensitive info if no specific type requested
#             safe_fields = [
#                 "full_name", "age", "gender", "city", "hometown", 
#                 "languages_spoken", "favorite_cuisines", "hobbies"
#             ]
#             return {k: v for k, v in user_data.items() if k in safe_fields}
#     except Exception as e:
#         return {"error": f"Error retrieving personal info: {str(e)}"}

@tool
def get_music_taste() -> Dict[str, List[Dict]]:
    """Get the user's music taste from Spotify data."""
    try:
        # sp = init_spotify()
        
        top_artists_data = get_top_artists(limit=5)
        top_tracks_data = get_top_tracks(limit=5)
        
        # Format the data for more readable output
        top_artists = [
            {
                "name": artist["name"],
                "genres": artist["genres"] if "genres" in artist else [],
                "popularity": artist["popularity"] if "popularity" in artist else None
            }
            for artist in top_artists_data
        ]
        
        top_tracks = [
            {
                "name": track["name"],
                "artist": track["artists"][0]["name"] if track["artists"] else "Unknown",
                "album": track["album"]["name"] if "album" in track else "Unknown"
            }
            for track in top_tracks_data
        ]
        
        return {
            "top_artists": top_artists,
            "top_tracks": top_tracks
        }
    except Exception as e:
        return {"error": f"Error retrieving Spotify data: {str(e)}"}

@tool
def get_linkedin_career_info() -> Dict[str, Any]:
    """Get career information from LinkedIn."""
    # This is a placeholder - you'll implement this with your LinkedIn API code
    return {
        "positions": [
            {
                "title": "Data Engineer",
                "company": "TELUS Communications Inc.",
                "startDate": "2023-01",
                "endDate": "present", 
                "description": "Data Engineering and API design"
            },
            {
                "title": "Business Intelligence Analyst",
                "company": "TELUS Communications Inc.",
                "startDate": "2021-06",
                "endDate": "2022-12",
                "description": "Data analysis and visualization"
            }
        ]
    }

# Create LangGraph for orchestration
def create_personal_assistant():
    # Initialize the language model
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    
    # Define the state
    class GraphState(TypedDict):
        messages: List[Dict[str, str]]
        next_step: str
        tool_result: str

    # Create graph
    workflow = StateGraph(GraphState)
    
    # Define system prompt
    system_prompt = """You are a personal assistant chatbot for Cem Kaspi. The year is 2025.
    You have access to Cem's resume, personal information, Spotify listening history, and LinkedIn profile.
    Use the available tools to retrieve the most relevant information to answer queries about Cem.
    Always be helpful, friendly, and professional. If you don't know something, say so honestly.
    
    Important details about Cem:
    - He is 28 years old (born March 12, 1997)
    - He is from Istanbul, Turkey, and currently lives in Vancouver, Canada
    - He works as a Data Engineer at TELUS Communications Inc.
    - He speaks Turkish, English, and Spanish
    
    When answering questions, try to be personable as if Cem created you to help people learn about him.
    """
    
    # Define node functions (separate from registration)    
    def route_query(state):
        """Determine what kind of query this is and what tool to use."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
    
        # Define the prompt template properly
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine what kind of query this is and what tool to use.
            Options:
            - "resume" for questions about education, work experience, skills
            - "personal" for questions about personal information, background, interests
            - "spotify" for questions about music taste, favorite artists, songs
            - "linkedin" for questions about career development, job promotions
            - "conversation" for general chat, greetings, questions that don't fit other categories
            
            Respond with only one word from the options above.
            """),
            ("human", "{query}")
        ])
    
        # Invoke the prompt with the user query
        prompt_value = route_prompt.invoke({"query": last_message})
        
        # Get the response from the LLM
        response = llm.invoke(prompt_value)
        query_type = response.content.strip().lower()
        
        # Ensure the query type is one of the valid options
        valid_types = ["resume", "personal", "spotify", "linkedin", "conversation"]
        if query_type not in valid_types:
            query_type = "conversation"  # Default to conversation for unrecognized types
    
        return {"messages": messages, "next_step": query_type}
    
    def handle_resume_query(state):
        messages = state["messages"]
        last_message = messages[-1]["content"]
        
        result = get_resume_info.invoke(last_message)
        return {"messages": messages, "tool_result": result, "next_step": "generate_response"}
    
    def handle_personal_query(state):
        messages = state["messages"]
        last_message = messages[-1]["content"]
        
        # Extract what type of personal info is being requested
        info_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract what specific type of personal information is being requested from this query. Return just the info type like 'age', 'hometown', 'languages_spoken', etc."),
            ("human", "{query}")
        ])
        
        prompt_value = info_prompt.invoke({"query": last_message})
        info_type = llm.invoke(prompt_value).content.strip()
        
        result = get_personal_info.invoke(info_type)
        return {"messages": messages, "tool_result": json.dumps(result), "next_step": "generate_response"}
    
    def handle_spotify_query(state):
        messages = state["messages"]
    
        try:
            # Fetch top artists and tracks (WITHOUT passing `sp`)
            top_artists = get_top_artists()
            top_tracks = get_top_tracks()
        
            # Check if the results are empty
            if not top_artists.strip() or not top_tracks.strip():
                raise ValueError("No listening history found. Try a different time range.")

            result = f"**🎵 Top Artists:**\n{top_artists}\n\n**🎶 Top Tracks:**\n{top_tracks}"

        except Exception as e:
            result = f"Error fetching Spotify data: {str(e)}"
            print(f"[DEBUG] Spotify API Error: {e}")  # Log error to console

        return {"messages": messages, "tool_result": result, "next_step": "generate_response"}
    
    def handle_linkedin_query(state):
        messages = state["messages"]
        result = get_linkedin_career_info.invoke()
        return {"messages": messages, "tool_result": json.dumps(result), "next_step": "generate_response"}
    
    def handle_conversation(state):
        messages = state["messages"]
        return {"messages": messages, "tool_result": "", "next_step": "generate_response"}
    
    def generate_response(state):
        print(state)
        messages = state["messages"] 
        tool_result = state.get("tool_result", "")
        
        # Convert messages dict format to LangChain message format
        from langchain_core.messages import HumanMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "human":
                langchain_messages.append(HumanMessage(content=content))
            elif msg["role"] == "ai":
                langchain_messages.append(AIMessage(content=content))
        
        # Create a prompt with system message and the tool result
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", f"Here is relevant information to help answer: {tool_result}")
        ])
        
        # First invoke the prompt to get the system messages
        prompt_value = prompt.invoke({})
        
        # Combine system prompts with conversation messages
        combined_messages = prompt_value.messages + langchain_messages
        
        # Get the response from the LLM using the combined messages
        response = llm.invoke(combined_messages)
        
        # Add the response to messages
        messages.append({"role": "ai", "content": response.content})
        
        return {"messages": messages, "next_step": "end"}
    
    # Add nodes to graph
    workflow.add_node("route_query", route_query)
    workflow.add_node("handle_resume_query", handle_resume_query)
    workflow.add_node("handle_personal_query", handle_personal_query)
    workflow.add_node("handle_spotify_query", handle_spotify_query)
    workflow.add_node("handle_linkedin_query", handle_linkedin_query)
    workflow.add_node("handle_conversation", handle_conversation)
    workflow.add_node("generate_response", generate_response)
    
    # Define edges - route based on query type
    workflow.add_conditional_edges(
        "route_query",
        lambda x: x["next_step"],
        {
            "resume": "handle_resume_query",
            "personal": "handle_personal_query",
            "spotify": "handle_spotify_query",
            "linkedin": "handle_linkedin_query",
            "conversation": "handle_conversation"
        }
    )
    
    # Add edges from every handler to generate_response
    workflow.add_edge("handle_resume_query", "generate_response")
    workflow.add_edge("handle_personal_query", "generate_response")
    workflow.add_edge("handle_spotify_query", "generate_response")
    workflow.add_edge("handle_linkedin_query", "generate_response")
    workflow.add_edge("handle_conversation", "generate_response")
    
    # Set the entry point
    workflow.set_entry_point("route_query")
    
    # Compile the graph
    return workflow.compile()

# Streamlit UI
def main():
    st.title("HowToCem")
    st.write("👋 Hi! I'm Cem Kaspi's virtual assistant. Ask me anything about his resume, personal info, career, or music taste!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        st.session_state.messages.append({
            "role": "ai", 
            "content": "Hello! I'm Cem's personal AI assistant. How can I help you learn more about him today?"
        })
    
    # Create assistant graph
    assistant_graph = create_personal_assistant()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask me something about Cem..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "content": prompt})
        
        # Display human message
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Generate response with thinking spinner
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    # Process with LangGraph
                    state = {
                        "messages": st.session_state.messages[:-1],  # Exclude just added message
                        "next_step": "",
                        "tool_result": ""
                    }
                    
                    # Add the latest user message
                    state["messages"].append({"role": "human", "content": prompt})
                    
                    # Run the graph
                    response_state = assistant_graph.invoke(state)
                    
                    # Extract the assistant's response (the last message)
                    assistant_response = response_state["messages"][-1]["content"]
                    
                    # Update the session state
                    st.session_state.messages = response_state["messages"]
                    
                    # Display the response
                    st.markdown(assistant_response)
                except Exception as e:
                    import traceback
                    error_message = f"I'm sorry, I encountered an error: {str(e)}\n\n"
                    st.markdown(error_message)
                    st.code(traceback.format_exc())
                    st.session_state.messages.append({"role": "ai", "content": error_message})

if __name__ == "__main__":
    main()
                