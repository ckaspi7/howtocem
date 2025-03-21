import os
from typing import Dict, List, Any, TypedDict, Optional
import json
import sqlite3
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pdfplumber
import streamlit as st
from spotify_data import get_top_artists, get_top_tracks
from user_data import get_user_data 

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.graph import StateGraph

# Load API keys from Streamlit secrets
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Error loading OpenAI API key: {e}")

# Resume extraction function
def extract_resume_info(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join([page.extract_text() or '' for page in pdf.pages])
        return text
    except Exception as e:
        st.error(f"Error extracting resume: {e}")
        return "Could not extract resume information."

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

# Function to prepare resume for vector storage
def prepare_resume_vectorstore():
    try:
        # Use relative path for the PDF file
        # First check if file exists in data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Created data directory at {os.path.abspath(data_dir)}")
            
        pdf_path = os.path.join(data_dir, "Cem_Kaspi_Resume.pdf")
        
        # If file doesn't exist, check in the current directory
        if not os.path.exists(pdf_path):
            pdf_path = "Cem_Kaspi_Resume.pdf"
            print(f"Trying to find resume at {os.path.abspath(pdf_path)}")
            
        if not os.path.exists(pdf_path):
            print(f"Resume not found at {os.path.abspath(pdf_path)}")
            st.error(f"Resume not found at {pdf_path}")
            return None

        print(f"Found resume at {os.path.abspath(pdf_path)}")
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
    except Exception as e:
        st.error(f"Error preparing resume vectorstore: {e}")
        return None

# Tool definitions for LangGraph
@tool
def get_resume_info(query: str) -> str:
    """Search through the resume for relevant information."""
    # Create or load the vectorstore
    try:
        vectorstore = prepare_resume_vectorstore()
        if vectorstore is None:
            return "Resume information is not currently available."
            
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

        # Convert the full dictionary into a readable string format
        return "\n".join(f"{key.capitalize()}: {value}" for key, value in user_data.items())

    except Exception as e:
        return f"Error retrieving personal info: {str(e)}"

@tool
def get_music_taste() -> Dict[str, List[Dict]]:
    """Get the user's music taste from Spotify data."""
    try:
        top_artists_data = get_top_artists()
        top_tracks_data = get_top_tracks()
        
        # Format the data for more readable output
        top_artists = [
            {
                "name": artist["name"],
                "genres": artist["genres"].split(", ") if artist["genres"] != "Unknown" else [],
            }
            for artist in top_artists_data
        ]

        top_tracks = [
            {
                "name": track["title"],
                "artist": track["artists"],
            }
            for track in top_tracks_data
        ]

        # top_artists = [
        #     {
        #         "name": artist["name"],
        #         "genres": artist["genres"] if "genres" in artist else [],
        #         "popularity": artist["popularity"] if "popularity" in artist else None
        #     }
        #     for artist in top_artists_data
        # ]
        
        # top_tracks = [
        #     {
        #         "name": track["name"],
        #         "artist": track["artists"][0]["name"] if track["artists"] else "Unknown",
        #         "album": track["album"]["name"] if "album" in track else "Unknown"
        #     }
        #     for track in top_tracks_data
        # ]
        
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
                "title": "AI/ML Engineer",
                "company": "TELUS Communications Inc.",
                "startDate": "2022-01",
                "endDate": "present", 
                "description": "GenAI application and ML model development"
            },
            {
                "title": "Co-Founder",
                "company": "NeoWise",
                "startDate": "2019-09",
                "endDate": "2022-01",
                "description": "Startup venture for a personal heating and cooling wearable product that optimizes your daily comfort and performance. Responsibilities include: Product R&D, Hardware Design, 3D Modeling & Rendering"
            },
            {
                "title": "Manufacturing Engineering Intern",
                "company": "Mercedes-Benz Canada",
                "startDate": "2018-01",
                "endDate": "2019-08",
                "description": "Worked as a part of the manufacturing engineering team with a group of multidisciplinary engineers and technicians at the Mercedes-Benz Fuel Cell Division. Performed statistical process analysis on bipolar plate welding and sealing."
            }
        ]
    }

# Add this function to check for database connectivity
def check_database_connection():
    try:
        from user_data import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return True, f"Connected to database. Tables found: {tables}"
    except Exception as e:
        return False, f"Database connection error: {str(e)}"

# Add this function to check for Spotify authentication
def check_spotify_connection():
    try:
        from spotify_data import sp
        if sp is None:
            return False, "Spotify client failed to initialize. Check logs for details."
            
        try:
            user_info = sp.current_user()
            return True, f"Connected to Spotify as: {user_info['display_name']}"
        except spotipy.exceptions.SpotifyException as e:
            return False, f"Spotify API error: {str(e)}"
    except ImportError as e:
        return False, f"Spotify module import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error with Spotify connection: {str(e)}"

# Create LangGraph for orchestration
def create_personal_assistant():
    # Initialize the language model
    try:
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        # Fallback to a simpler model if needed
        try:
            llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        except:
            st.error("Could not initialize any language model. Check your API key.")
            return None
    
    # Define the state
    class GraphState(TypedDict):
        messages: List[Dict[str, str]]
        next_step: str
        tool_result: str

    # Create graph
    workflow = StateGraph(GraphState)
    
    # Define system prompt
    system_prompt = """You are a helpful personal assistant chatbot for Cem Kaspi. The year is 2025.
    You have access to Cem's resume, personal information, Spotify listening history, and LinkedIn profile. 
    Inform the user of your capabilities when asked about it, with questions such as 'what can you do' or 'what are your capabilities'.
    Use the available tools to retrieve the most relevant information to answer queries about Cem. 
    If you did not receive any relevant information from the tools to answer the query, say so. 
    Do not make up any false or fake information to answer the query and only base your responses on the information given to you.
    Always be helpful, friendly, and professional. You can be humorful and lighthearted to imitate Cem's personality.
    If you don't know something, say so honestly.
    
    Important details about Cem:
    - He is 28 years old (born March 12, 1997)
    - He is from Istanbul, Turkey, and currently lives in Vancouver, Canada
    - He works as an AI/ML Engineer at TELUS Communications Inc.
    - He is bilingual in Turkish and English, and speaks beginner-level Spanish
    
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
        return {"messages": messages, "tool_result": result, "next_step": "generate_response"}

    def handle_spotify_query(state):
        messages = state["messages"]
    
        try:
            # Fetch music taste data using the @tool function with .invoke()
            result = get_music_taste.invoke({})

            # If the result contains an error message, raise it
            if "error" in result:
                raise ValueError(result["error"])
            
            # Extract top artists and tracks from the result
            top_artists = result["top_artists"]
            top_tracks = result["top_tracks"]
        
            # Check if the results are empty
            if not top_artists or not top_tracks:
                raise ValueError("No listening history found. Try a different time range.")
            
            # Format artists and tracks for display
            # Format artists and tracks for display
            top_artists_str = "\n".join(
                [f"{i+1}. {artist['name']} (Genres: {', '.join(artist['genres'])})" for i, artist in enumerate(top_artists)]
            )

            top_tracks_str = "\n".join(
                [f"{i+1}. {track['name']} - {track['artist']}" for i, track in enumerate(top_tracks)]
            )

            result = f"🎵 **Top Artists:**\n{top_artists_str}\n\n🎶 **Top Tracks:**\n{top_tracks_str}"

        except Exception as e:
            result = f"Error fetching Spotify data: {str(e)}"
            print(f"[DEBUG] Spotify API Error: {e}")  # Log error to console

        return {"messages": messages, "tool_result": result, "next_step": "generate_response"}

    # def handle_spotify_query(state):
    #     messages = state["messages"]
    
    #     try:
    #         # Fetch top artists and tracks (WITHOUT passing `sp`)
    #         top_artists = get_top_artists()
    #         top_tracks = get_top_tracks()
        
    #         # Check if the results are empty
    #         if not top_artists or not top_tracks:
    #             raise ValueError("No listening history found. Try a different time range.")
            
    #         # Format artists and tracks for display
    #         top_artists_str = "\n".join(
    #             [f"{artist['rank']}. {artist['name']} (Genres: {artist['genres']})" for artist in top_artists]
    #         )

    #         top_tracks_str = "\n".join(
    #             [f"{track['rank']}. {track['title']} - {track['artists']}" for track in top_tracks]
    #         )

    #         result = f"🎵 **Top Artists:**\n{top_artists_str}\n\n🎶 **Top Tracks:**\n{top_tracks_str}"

    #     except Exception as e:
    #         result = f"Error fetching Spotify data: {str(e)}"
    #         print(f"[DEBUG] Spotify API Error: {e}")  # Log error to console

    #     return {"messages": messages, "tool_result": result, "next_step": "generate_response"}
    
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
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
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
        try:
            response = llm.invoke(combined_messages)
            response_content = response.content
        except Exception as e:
            response_content = f"I'm sorry, I encountered an error while generating a response: {str(e)}"
        
        # Add the response to messages
        messages.append({"role": "ai", "content": response_content})
        
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
    
    # Debug Mode
    with st.sidebar:
        debug_mode = st.checkbox("Debug Mode")
        if debug_mode:
            st.write("### Environment Check")
            st.write(f"Current Working Directory: {os.getcwd()}")
            st.write(f"Files in Directory: {os.listdir()}")
            
            # Check for data directory
            if os.path.exists("data"):
                st.write(f"Files in data directory: {os.listdir('data')}")
            else:
                st.write("Data directory not found")
                
            # Check for API keys
            st.write("API Keys:")
            try:
                has_openai = "OPENAI_API_KEY" in st.secrets
                st.write(f"- OpenAI API Key set: {has_openai}")
            except:
                st.write("- OpenAI API Key: Not found in secrets")
                
            try:
                has_spotify_id = "SPOTIFY_CLIENT_ID" in st.secrets
                has_spotify_secret = "SPOTIFY_CLIENT_SECRET" in st.secrets
                st.write(f"- Spotify Client ID set: {has_spotify_id}")
                st.write(f"- Spotify Client Secret set: {has_spotify_secret}")
            except:
                st.write("- Spotify credentials: Not found in secrets")

            st.write("### Connection Tests")
            db_connected, db_message = check_database_connection()
            st.write(f"Database: {'✅' if db_connected else '❌'} {db_message}")
    
            spotify_connected, spotify_message = check_spotify_connection()
            st.write(f"Spotify: {'✅' if spotify_connected else '❌'} {spotify_message}")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        st.session_state.messages.append({
            "role": "ai", 
            "content": "Hello! I'm Cem's personal AI assistant. How can I help you learn more about him today?"
        })
    
    # Create assistant graph
    try:
        assistant_graph = create_personal_assistant()
    except Exception as e:
        st.error(f"Error creating assistant: {e}")
        assistant_graph = None
    
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
                    if assistant_graph is None:
                        st.markdown("I'm sorry, I couldn't initialize the assistant. Please check the logs in debug mode.")
                        st.session_state.messages.append({
                            "role": "ai", 
                            "content": "I'm sorry, I couldn't initialize the assistant. Please check the logs in debug mode."
                        })
                    else:
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
                
