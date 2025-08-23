import streamlit as st
import os
import PyPDF2
from pathlib import Path
import tempfile
import time
import chardet

from crewai import Agent, Task, Crew, Process
from crewai_tools import FileWriterTool, SerperDevTool, GithubSearchTool, LinkupSearchTool, EXASearchTool
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()

# Set environment variables for custom API endpoint
os.environ["OPENAI_API_BASE"] = "https://api.aimlapi.com/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("AIML_API_KEY", "<YOUR_API_KEY>")

# Configure LLM for CrewAI
llm_config = "openai/gpt-5-chat-latest"

# Initialize tools
file_writer = FileWriterTool()
serper_tool = SerperDevTool()
github_search_tool = GithubSearchTool(
    gh_token=os.getenv("GITHUB_TOKEN"),
    content_types=['code', 'issue'],
    max_results=500
)
linkup_tool = LinkupSearchTool(api_key=os.getenv("LINKUP_API_KEY"))
exascience_tool = EXASearchTool(api_key=os.getenv("EXA_API_KEY"))

def read_pdf_content(pdf_file) -> str:
    """Read and extract text content from uploaded PDF with better encoding handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            try:
                page_text = page.extract_text()
                if page_text:
                    # Clean up common PDF encoding issues
                    page_text = page_text.replace('\x96', '-')  # Replace problematic byte
                    page_text = page_text.replace('\x97', '-')  # Replace problematic byte
                    page_text = page_text.replace('\x94', '"')  # Replace problematic byte
                    page_text = page_text.replace('\x93', '"')  # Replace problematic byte
                    text_content += page_text + "\n"
            except Exception as e:
                st.warning(f"Warning: Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        if not text_content.strip():
            return "Warning: No text could be extracted from the PDF. The file might be image-based or corrupted."
        
        return text_content
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def safe_read_file(file_path):
    """Safely read file content with encoding detection"""
    try:
        # First try UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] if detected['encoding'] else 'latin-1'
            
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path.name}: {str(e)}"

def create_agents():
    """Create and return the CrewAI agents"""
    
    # Project Analysis Agent
    project_analyst = Agent(
        role='Project Analyst',
        goal='Analyze project documents to identify risks, strengths, and opportunities.',
        backstory=(
            "You are a skilled project analyst who can read project documents and "
            "extract meaningful insights, risks, and opportunities. "
            "You have access to the full project document content and can research "
            "similar projects on the web to provide comprehensive market analysis."
        ),
        tools=[file_writer, serper_tool],
        verbose=True,
        memory=True,
        llm=llm_config
    )

    # Resource Search Agent
    resource_search_agent = Agent(
        role='Resource Search Specialist',
        goal='Efficiently locate and curate the most relevant, high-quality resources from multiple platforms.',
        backstory=(
            "A specialized research librarian with deep knowledge of developer communities, "
            "academic databases, and open-source ecosystems. Expert at evaluating resource quality, "
            "licensing compatibility, and relevance scoring."
        ),
        tools=[file_writer, serper_tool, github_search_tool, linkup_tool],
        verbose=True,
        memory=True,
        llm=llm_config
    )

    # Coding Agent
    coding_agent = Agent(
        role='Senior Full-Stack Developer & Code Architect',
        goal='Generate production-ready, modular code that accelerates development while maintaining best practices.',
        backstory=(
            "A senior full-stack developer and architect with expertise across multiple programming languages, "
            "frameworks, and design patterns. Specializes in rapid prototyping while maintaining code quality and scalability. "
            "Expert in Python, JavaScript/TypeScript, React, Node.js, and modern development practices."
        ),
        tools=[file_writer],
        verbose=True,
        memory=True,
        allow_code_execution=True,
        llm=llm_config
    )
    
    return project_analyst, resource_search_agent, coding_agent

def create_tasks(agents, pdf_content):
    """Create and return the CrewAI tasks"""
    
    project_analyst, resource_search_agent, coding_agent = agents
    
    # Create output folders
    output_folder = Path("project_analysis_output")
    output_folder.mkdir(exist_ok=True)
    
    resource_folder = Path("resource_output")
    resource_folder.mkdir(exist_ok=True)
    
    code_folder = Path("code_output")
    code_folder.mkdir(exist_ok=True)
    
    # Project Analysis Tasks
    project_context_task = Task(
        description=(
            f"Analyze the following project document content and identify key project goals, risks, challenges, and potential improvements. "
            f"Your final answer MUST be structured as a Markdown report with sections for Summary, Risks, Strengths, and Opportunities.\n\n"
            f"CRITICAL: You MUST use the FileWriterTool to save your analysis to '{output_folder}/project_analysis.md'\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output="A structured Markdown analysis report of the project saved to project_analysis.md",
        agent=project_analyst
    )
    
    # Resource Search Tasks
    resource_discovery_task = Task(
        description=(
            f"Based on the project analysis, search for relevant resources across GitHub, Kaggle, ArXiv, and other platforms. "
            f"Focus on AI/ML tools, social media APIs, and development frameworks.\n\n"
            f"CRITICAL: You MUST use the FileWriterTool to save your findings to '{resource_folder}/resource_discovery.md'\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output="A comprehensive resource discovery report saved to resource_discovery.md",
        agent=resource_search_agent
    )
    
    # Code Generation Tasks
    architecture_task = Task(
        description=(
            f"Based on the project analysis, design the system architecture for the AI-powered social media marketing platform. "
            f"Create a scalable, modular design with clear module boundaries and data flow.\n\n"
            f"CRITICAL: You MUST use the FileWriterTool to save your architecture design to '{code_folder}/architecture_design.md'\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output="Complete architecture design saved to architecture_design.md",
        agent=coding_agent
    )
    
    return [project_context_task, resource_discovery_task, architecture_task]

def run_crew_analysis(pdf_content):
    """Run the CrewAI analysis and return results"""
    
    # Create agents and tasks
    agents = create_agents()
    tasks = create_tasks(agents, pdf_content)
    
    # Build the crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential
    )
    
    # Run the analysis
    result = crew.kickoff()
    return result

def wait_for_files_and_refresh():
    """Wait for files to be generated and refresh the display"""
    st.info("⏳ Waiting for files to be generated...")
    
    # Wait a bit for file operations to complete
    time.sleep(2)
    
    # Force a rerun to refresh the display
    st.rerun()

def main():
    st.set_page_config(
        page_title="PDF Analysis with CrewAI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 PDF Analysis with CrewAI Agents")
    st.markdown("Upload a PDF and let our AI agents analyze it for you!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to analyze"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Display PDF info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.metric("File Type", "PDF")
        with col3:
            st.metric("Status", "Ready for Analysis")
        
        # Analysis button
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Agents are analyzing your PDF..."):
                # Read PDF content
                pdf_content = read_pdf_content(uploaded_file)
                
                if pdf_content.startswith("Error"):
                    st.error(f"❌ {pdf_content}")
                    return
                
                if pdf_content.startswith("Warning"):
                    st.warning(f"⚠️ {pdf_content}")
                
                # Show PDF content preview
                with st.expander("📄 PDF Content Preview", expanded=False):
                    st.text_area("PDF Text", pdf_content, height=200, disabled=True)
                
                # Run CrewAI analysis
                try:
                    result = run_crew_analysis(pdf_content)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Show the main result
                    st.subheader("📊 Analysis Results")
                    st.markdown(result)
                    
                    # Add a refresh button to check for generated files
                    if st.button("🔄 Check Generated Files", type="secondary"):
                        wait_for_files_and_refresh()
                    
                    # Show generated files
                    st.subheader("📁 Generated Files")
                    
                    # Check for generated files
                    output_folders = ["project_analysis_output", "resource_output", "code_output"]
                    
                    files_found = False
                    for folder in output_folders:
                        folder_path = Path(folder)
                        if folder_path.exists():
                            files = list(folder_path.glob("*.md"))
                            if files:
                                files_found = True
                                st.write(f"**{folder.replace('_', ' ').title()}:**")
                                for file in files:
                                    try:
                                        # Use safe file reading
                                        content = safe_read_file(file)
                                        
                                        with st.expander(f"📄 {file.name}", expanded=False):
                                            if content.startswith("Error reading file"):
                                                st.error(content)
                                            else:
                                                st.markdown(content)
                                        
                                        # Download button for each file
                                        if not content.startswith("Error"):
                                            st.download_button(
                                                label=f"💾 Download {file.name}",
                                                data=content,
                                                file_name=file.name,
                                                mime="text/markdown"
                                            )
                                    except Exception as e:
                                        st.error(f"❌ Error reading {file.name}: {str(e)}")
                    
                    if not files_found:
                        st.warning("⚠️ No generated files found yet. The agents may still be processing. Click 'Check Generated Files' to refresh.")
                        st.info("💡 Generated files will appear here once the agents complete their tasks.")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 Make sure your API keys are properly configured in the .env file")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload PDF**: Choose a PDF file to analyze
        2. **Start Analysis**: Click the button to begin AI analysis
        3. **View Results**: See the analysis results and generated files
        4. **Check Files**: Click "Check Generated Files" to refresh
        5. **Download**: Download any generated markdown files
        
        **Required API Keys:**
        - AIML_API_KEY (for GPT-5)
        - GITHUB_TOKEN (optional)
        - LINKUP_API_KEY (optional)
        - EXA_API_KEY (optional)
        """)
        
        st.header("🔧 Setup")
        st.markdown("""
        Make sure you have a `.env` file with:
        ```
        AIML_API_KEY=your_api_key_here
        GITHUB_TOKEN=your_github_token
        LINKUP_API_KEY=your_linkup_key
        EXA_API_KEY=your_exa_key
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Powered by CrewAI 🤖 | Built with Streamlit 📱</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
