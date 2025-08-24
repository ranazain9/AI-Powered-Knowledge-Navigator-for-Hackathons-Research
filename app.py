import sys

import streamlit as st
import os
import PyPDF2
from pathlib import Path
import tempfile
import time
import chardet
import traceback


# Force Python to use pysqlite3 instead of system sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from crewai import Agent, Task, Crew, Process
from crewai_tools import FileWriterTool, SerperDevTool, GithubSearchTool, LinkupSearchTool, EXASearchTool
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()

try:
    st.set_page_config(page_title="AI Knowledge Navigator", layout="wide")
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.text(traceback.format_exc())

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
        # allow_code_execution=True,
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
    
    # ===== FIRST AGENT TASKS (Project Analysis) =====
    
    # Task 1: Project Context Analysis with web research
    project_context_task = Task(
        description=(
            f"CRITICAL: You MUST read the PDF content from the input context and extract REAL information.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Look at the 'full_content' field in the input\n"
            f"2. Extract the actual project name from the 'project_name' field\n"
            f"3. Find real requirements mentioned in the PDF content\n"
            f"4. Identify actual technology stack mentioned in the 'technology_stack' field\n"
            f"5. Extract real project objectives and scope from the PDF text\n"
            f"6. Find actual constraints and limitations mentioned\n"
            f"7. Identify real dependencies and relationships\n"
            f"8. Extract actual risks and mitigation strategies\n"
            f"9. Find real timeline and milestones from the 'timeline_info' field\n"
            f"10. Extract actual team information and roles from the 'team_info' field\n"
            f"11. Find real budget and resource information from the 'budget_info' field\n\n"
            f"WEB RESEARCH: Use SerperDevTool to search for similar AI-powered social media marketing platforms and analyze:\n"
            f"- Current market leaders (Hootsuite, Buffer, Sprout Social, Later)\n"
            f"- Their key features and pricing\n"
            f"- Market gaps and opportunities\n"
            f"- Recent funding and acquisition news\n\n"
            f"USE FileWriterTool to write this to '{output_folder}/project_analysis.md'\n"
            f"DO NOT use placeholders like [Detail] or [Project Name] - use REAL data from the PDF!\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output=(
            f"A REAL project analysis report saved as '{output_folder}/project_analysis.md' containing:\n"
            "- ACTUAL project name from the PDF (not [Project Name])\n"
            "- REAL requirements extracted from the PDF content (not [Detail])\n"
            "- ACTUAL technology stack mentioned in the PDF (not [Technology 1])\n"
            "- REAL project objectives and scope from the PDF content\n"
            "- ACTUAL constraints and limitations found in the PDF\n"
            "- REAL dependencies and relationships identified\n"
            "- ACTUAL risks and mitigation strategies\n"
            "- REAL timeline and milestones from the PDF\n"
            "- ACTUAL team information and roles\n"
            "- REAL budget and resource information\n"
            "- WEB RESEARCH: Market analysis of similar platforms, competitors, and opportunities\n\n"
            "Use FileWriterTool to save this with real data extracted from the PDF content and web research."
        ),
        agent=project_analyst
    )

    # Task 2: Objective Clarification with market research
    objective_task = Task(
        description=(
            f"CRITICAL: Based on the ACTUAL PDF content in the input, break down real project goals into specific objectives.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the 'full_content' field in the input\n"
            f"2. Extract the actual project goals mentioned in the PDF\n"
            f"3. Identify real secondary goals from the PDF content\n"
            f"4. Find specific acceptance criteria mentioned in the PDF\n"
            f"5. Determine recommended project phases based on the real scope\n\n"
            f"WEB RESEARCH: Use SerperDevTool to research:\n"
            f"- Successful social media marketing platform launches\n"
            f"- Market validation strategies for AI tools\n"
            f"- User acquisition patterns in the social media space\n"
            f"- Industry benchmarks for engagement and ROI metrics\n\n"
            f"USE FileWriterTool to write this to '{output_folder}/project_objectives.md'\n"
            f"DO NOT use placeholders - use REAL data from the PDF!\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output=(
            f"A REAL objectives document saved as '{output_folder}/project_objectives.md' containing:\n"
            "- ACTUAL primary goals extracted from the PDF content (not [Objective 1])\n"
            "- REAL secondary goals identified in the project (not [Goal 1])\n"
            "- SPECIFIC acceptance criteria based on the actual requirements (not [Criterion 1])\n"
            "- RECOMMENDED project phases based on the real project scope (not [Phase 1])\n"
            "- WEB RESEARCH: Market validation insights and industry benchmarks\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=project_analyst
    )

    # Task 3: Technical Feasibility Assessment with technology research
    technical_task = Task(
        description=(
            f"CRITICAL: Evaluate the technical complexity of the ACTUAL project described in the PDF content.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the 'full_content' field in the input\n"
            f"2. Analyze the real technology stack mentioned in the 'technology_stack' field\n"
            f"3. Assess the actual project complexity based on real requirements\n"
            f"4. Identify real prerequisite skills needed for the specific technologies\n"
            f"5. Recommend project-specific fallback solutions based on real constraints\n\n"
            f"WEB RESEARCH: Use SerperDevTool to research:\n"
            f"- Latest AI/ML technologies for social media content generation\n"
            f"- Social media API limitations and best practices\n"
            f"- Successful tech stacks used by similar platforms\n"
            f"- Development timeframes for AI-powered marketing tools\n"
            f"- Technical challenges in social media automation\n\n"
            f"USE FileWriterTool to write this to '{output_folder}/technical_assessment.md'\n"
            f"DO NOT use placeholders - use REAL data from the PDF!\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output=(
            f"A REAL technical assessment saved as '{output_folder}/technical_assessment.md' containing:\n"
            "- ACTUAL project complexity rating based on the real requirements (not generic ratings)\n"
            "- REAL prerequisite skills analysis for the specific technologies mentioned (not [Language 1])\n"
            "- PROJECT-SPECIFIC fallback solution recommendations based on real constraints (not generic solutions)\n"
            "- WEB RESEARCH: Latest AI/ML technologies, API limitations, and technical challenges\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=project_analyst
    )

    # Task 4: Resource Requirements Planning with market insights
    resource_task = Task(
        description=(
            f"CRITICAL: Based on the ACTUAL project requirements from the PDF, determine what real resources are needed.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the 'full_content' field in the input\n"
            f"2. Analyze the actual technology stack mentioned in the 'technology_stack' field\n"
            f"3. Identify real datasets needed for the specific project\n"
            f"4. Determine actual documentation requirements based on the real scope\n"
            f"5. Find real resources and tools needed for the specific technologies\n"
            f"6. Identify actual APIs and services required for the project\n\n"
            f"WEB RESEARCH: Use SerperDevTool to research:\n"
            f"- Current costs for AI/ML development services\n"
            f"- Social media API pricing and rate limits\n"
            f"- Market rates for social media marketing developers\n"
            f"- Required infrastructure and hosting costs\n"
            f"- Legal and compliance costs for social media tools\n\n"
            f"USE FileWriterTool to write this to '{output_folder}/resource_planning.md'\n"
            f"DO NOT use placeholders - use REAL data from the PDF!\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output=(
            f"A REAL resource plan saved as '{output_folder}/resource_planning.md' containing:\n"
            "- ACTUAL datasets needed for the specific project (not generic datasets)\n"
            "- REAL documentation requirements based on the project scope (not generic docs)\n"
            "- ACTUAL resources and tools needed for the specific technologies (not generic tools)\n"
            "- REAL APIs and services required for the project (not generic APIs)\n"
            "- WEB RESEARCH: Current market costs, developer rates, and infrastructure pricing\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=project_analyst
    )

    # ===== SECOND AGENT TASKS (Resource Search) =====
    
    # Task 5: Multi-platform resource discovery
    multi_platform_discovery_task = Task(
        description=(
            f"Based on the project analysis files created by the first agent, search across GitHub, Kaggle, ArXiv, StackOverflow, and documentation sites using "
            f"intelligent query expansion and semantic matching.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files from '{output_folder}/' to understand the project requirements\n"
            f"2. Use SerperDevTool to search for relevant resources based on the technology stack and requirements identified\n"
            f"3. Focus on AI/ML tools, social media APIs, and development frameworks mentioned in the project\n"
            f"4. Search across multiple platforms for comprehensive coverage\n"
            f"5. CRITICAL: You MUST find and document AT LEAST 10 different resources across all platforms\n"
            f"6. Use multiple search queries to ensure comprehensive coverage\n"
            f"7. Search for: AI marketing tools, social media automation, content generation APIs, analytics platforms\n\n"
            f"USE FileWriterTool to write this to '{resource_folder}/multi_platform_resources.md'\n"
        ),
        expected_output=(
            f"A comprehensive resource discovery report saved as '{resource_folder}/multi_platform_resources.md' containing:\n"
            "- AT LEAST 10 different resources with relevance scores\n"
            "- Platform source for each resource\n"
            "- Brief descriptions and use cases\n"
            "- Direct links to the resources\n"
            "- Categorized by platform (GitHub, Kaggle, ArXiv, StackOverflow, Documentation)\n\n"
            "Use FileWriterTool to save this with real search results. Ensure you have at least 10 resources total."
        ),
        agent=resource_search_agent
    )

    # Task 6: Code repository analysis and filtering
    code_repository_analysis_task = Task(
        description=(
            f"Based on the project analysis, analyze GitHub repositories for code quality, maintenance status, licensing, and "
            f"compatibility with project requirements.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files from '{output_folder}/' to understand the project scope\n"
            f"2. Use SerperDevTool AND GithubSearchTool to find relevant GitHub repositories\n"
            f"3. Evaluate repositories for quality, maintenance, and licensing compatibility\n"
            f"4. Focus on repositories related to AI/ML, social media tools, and the specific technologies mentioned\n"
            f"5. CRITICAL: You MUST find and analyze AT LEAST 10 different GitHub repositories\n"
            f"6. Search for: social media marketing tools, AI content generation, marketing automation, analytics platforms\n"
            f"7. Use multiple search terms and filters to ensure comprehensive coverage\n\n"
            f"USE FileWriterTool to write this to '{resource_folder}/code_repositories.md'\n"
        ),
        expected_output=(
            f"A curated repository analysis saved as '{resource_folder}/code_repositories.md' containing:\n"
            "- AT LEAST 10 different GitHub repositories with quality metrics\n"
            "- License compatibility flags\n"
            "- Integration difficulty assessments\n"
            "- Maintenance status and community activity\n"
            "- Direct links to each repository\n"
            "- Star count, last updated, and language information\n\n"
            "Use FileWriterTool to save this with real repository analysis. Ensure you have at least 10 repositories."
        ),
        agent=resource_search_agent
    )

    # Task 7: Dataset discovery and validation
    dataset_discovery_task = Task(
        description=(
            f"Based on the project requirements, find relevant datasets on Kaggle, academic repositories, and government data portals, "
            f"validating data quality and format compatibility.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files from '{output_folder}/' to understand data needs\n"
            f"2. Use SerperDevTool to search for relevant datasets\n"
            f"3. Focus on social media data, AI/ML training datasets, and marketing analytics data\n"
            f"4. Evaluate data quality, format, and licensing\n"
            f"5. CRITICAL: You MUST find and document AT LEAST 10 different datasets\n"
            f"6. Search for: social media engagement data, marketing campaign data, user behavior datasets, content performance data\n"
            f"7. Include datasets from: Kaggle, UCI ML Repository, Google Dataset Search, AWS Open Data, academic sources\n"
            f"8. Use multiple search queries to ensure comprehensive coverage\n\n"
            f"USE FileWriterTool to write this to '{resource_folder}/datasets.md'\n"
        ),
        expected_output=(
            f"A dataset catalog saved as '{resource_folder}/datasets.md' containing:\n"
            "- AT LEAST 10 different datasets with size metrics\n"
            "- Format specifications and quality indicators\n"
            "- Usage examples and integration notes\n"
            "- Licensing and access information\n"
            "- Direct links to download/access each dataset\n"
            "- Source platform and last updated information\n\n"
            "Use FileWriterTool to save this with real dataset findings. Ensure you have at least 10 datasets."
        ),
        agent=resource_search_agent
    )

    # Task 8: Academic paper and documentation retrieval
    academic_paper_task = Task(
        description=(
            f"Based on the project scope, search ArXiv, research databases, and technical documentation for relevant papers, "
            f"tutorials, and implementation guides.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files from '{output_folder}/' to understand research needs\n"
            f"2. Use SerperDevTool to search academic and technical sources\n"
            f"3. Focus on AI/ML research, social media analytics, and marketing automation papers\n"
            f"4. Look for implementation guides and best practices\n"
            f"5. CRITICAL: You MUST find and document AT LEAST 10 different academic resources\n"
            f"6. Search for: social media marketing research, AI content generation papers, marketing automation studies, analytics methodologies\n"
            f"7. Include sources from: ArXiv, Google Scholar, ResearchGate, IEEE Xplore, ACM Digital Library, arXiv.org\n"
            f"8. Use multiple search queries and keywords to ensure comprehensive coverage\n\n"
            f"USE FileWriterTool to write this to '{resource_folder}/academic_resources.md'\n"
        ),
        expected_output=(
            f"An annotated bibliography saved as '{resource_folder}/academic_resources.md' containing:\n"
            "- AT LEAST 10 different academic resources with paper summaries\n"
            "- Implementation difficulty ratings\n"
            "- Code availability indicators\n"
            "- Relevance to project requirements\n"
            "- Direct links to papers and documentation\n"
            "- Publication date and author information\n\n"
            "Use FileWriterTool to save this with real academic research findings. Ensure you have at least 10 resources."
        ),
        agent=resource_search_agent
    )

    # Task 9: Real-time resource monitoring
    realtime_monitoring_task = Task(
        description=(
            f"Based on the project timeline and requirements, continuously monitor for new releases, updates, or trending resources "
            f"related to the project domain.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files from '{output_folder}/' to understand monitoring priorities\n"
            f"2. Use SerperDevTool to search for recent developments and trends\n"
            f"3. Focus on AI/ML tools, social media platforms, and marketing technology updates\n"
            f"4. Identify emerging tools and libraries that could benefit the project\n"
            f"5. CRITICAL: You MUST find and document AT LEAST 10 different recent/trending resources\n"
            f"6. Search for: latest AI marketing tools, new social media APIs, emerging marketing technologies, recent platform updates\n"
            f"7. Include sources from: tech blogs, GitHub trending, product hunt, tech news sites, developer blogs\n"
            f"8. Focus on resources published/updated in the last 6 months\n"
            f"9. Use multiple search queries to ensure comprehensive coverage\n\n"
            f"USE FileWriterTool to write this to '{resource_folder}/realtime_updates.md'\n"
        ),
        expected_output=(
            f"A live resource update feed saved as '{resource_folder}/realtime_updates.md' containing:\n"
            "- AT LEAST 10 different recent/trending resources\n"
            "- Change impact analysis\n"
            "- Integration recommendations\n"
            "- Trend analysis and future predictions\n"
            "- Direct links to each resource\n"
            "- Publication/update dates\n\n"
            "Use FileWriterTool to save this with real-time findings. Ensure you have at least 10 resources."
        ),
        agent=resource_search_agent
    )

    # ===== THIRD AGENT TASKS (Coding and Development) =====
    
    # Task 10: Project Architecture Design
    architecture_design_task = Task(
        description=(
            f"Based on the project analysis files from '{output_folder}/', design the overall system architecture "
            f"for the AI-powered social media marketing platform.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the analysis files to understand project requirements and technology stack\n"
            f"2. Design a scalable, modular architecture using modern best practices\n"
            f"3. Define clear module boundaries and data flow\n"
            f"4. Create folder structure templates and coding conventions\n"
            f"5. Consider microservices vs monolithic architecture based on project scope\n"
            f"6. Include API design patterns and database schema considerations\n"
            f"7. Address security, scalability, and maintainability concerns\n\n"
            f"USE FileWriterTool to write this to '{code_folder}/architecture_design.md'\n"
            f"Also generate a Python script that creates the basic folder structure."
        ),
        expected_output=(
            f"Complete architecture design saved as '{code_folder}/architecture_design.md' containing:\n"
            "- System architecture diagrams and descriptions\n"
            "- Module boundaries and responsibilities\n"
            "- Technology stack recommendations\n"
            "- Folder structure templates\n"
            "- Coding conventions and standards\n"
            "- Security and scalability considerations\n"
            "- Python script for creating project structure\n\n"
            "Use FileWriterTool to save this with detailed architectural guidance."
        ),
        agent=coding_agent
    )

    # Task 11: Starter Template Generation
    starter_template_task = Task(
        description=(
            f"Based on the architecture design and project requirements, generate a complete project scaffolding "
            f"with boilerplate code, configuration files, and basic functionality implementations.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the architecture design and project analysis files\n"
            f"2. Create a fully functional project template with proper structure\n"
            f"3. Include dependency management (requirements.txt, pyproject.toml)\n"
            f"4. Generate basic configuration files (.env.example, config.py)\n"
            f"5. Create starter API endpoints and basic functionality\n"
            f"6. Include database models and connection setup\n"
            f"7. Add authentication and basic security features\n"
            f"8. Create comprehensive setup instructions\n\n"
            f"USE FileWriterTool to write setup instructions to '{code_folder}/setup_instructions.md'\n"
            f"Generate all necessary Python files in the '{code_folder}/project_template/' directory."
        ),
        expected_output=(
            f"Complete project template saved in '{code_folder}/project_template/' containing:\n"
            "- Fully functional project structure\n"
            "- All necessary Python files with basic implementations\n"
            "- Configuration and dependency files\n"
            "- Basic API endpoints and functionality\n"
            "- Database models and setup\n"
            "- Authentication and security features\n"
            "- Comprehensive setup instructions\n\n"
            "Use FileWriterTool to save setup instructions and generate all code files."
        ),
        agent=coding_agent
    )

    # Task 12: Custom Function and Component Creation
    custom_components_task = Task(
        description=(
            f"Based on the project requirements, generate specific functions, classes, and components "
            f"for the AI-powered social media marketing platform.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the project analysis to understand specific feature requirements\n"
            f"2. Create core business logic functions for social media management\n"
            f"3. Generate AI content generation and analysis components\n"
            f"4. Build user management and authentication systems\n"
            f"5. Create analytics and reporting modules\n"
            f"6. Include proper error handling, logging, and documentation\n"
            f"7. Add unit tests for critical functions\n"
            f"8. Ensure code follows best practices and is production-ready\n\n"
            f"USE FileWriterTool to write this to '{code_folder}/custom_components.md'\n"
            f"Generate all Python files with well-documented, tested code modules."
        ),
        expected_output=(
            f"Custom components and functions saved in '{code_folder}/components/' containing:\n"
            "- Core business logic functions\n"
            "- AI content generation components\n"
            "- User management systems\n"
            "- Analytics and reporting modules\n"
            "- Comprehensive error handling and logging\n"
            "- Unit tests for critical functions\n"
            "- Usage examples and integration guidelines\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=coding_agent
    )

    # Task 13: API Integration Code Generation
    api_integration_task = Task(
        description=(
            f"Create wrapper functions and integration code for external APIs, databases, and third-party services "
            f"required for the social media marketing platform.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the project requirements to identify needed integrations\n"
            f"2. Create API clients for social media platforms (Twitter, Facebook, Instagram, LinkedIn)\n"
            f"3. Build database connection and ORM implementations\n"
            f"4. Generate authentication and rate limiting handlers\n"
            f"5. Create webhook handlers for real-time updates\n"
            f"6. Include comprehensive error handling and retry logic\n"
            f"7. Add logging and monitoring capabilities\n"
            f"8. Ensure security best practices for API keys and tokens\n\n"
            f"USE FileWriterTool to write this to '{code_folder}/api_integrations.md'\n"
            f"Generate all Python files with complete API client implementations."
        ),
        expected_output=(
            f"API integration code saved in '{code_folder}/integrations/' containing:\n"
            "- Social media platform API clients\n"
            "- Database connection and ORM implementations\n"
            "- Authentication and rate limiting handlers\n"
            "- Webhook handlers for real-time updates\n"
            "- Comprehensive error handling and retry logic\n"
            "- Logging and monitoring capabilities\n"
            "- Security best practices implementation\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=coding_agent
    )

    # Task 14: Testing and Validation Code Creation
    testing_validation_task = Task(
        description=(
            f"Generate comprehensive unit tests, integration tests, and validation scripts "
            f"to ensure code reliability and performance for the social media marketing platform.\n\n"
            f"SPECIFIC INSTRUCTIONS:\n"
            f"1. Read the generated code components to understand what needs testing\n"
            f"2. Create unit tests for all critical functions and classes\n"
            f"3. Generate integration tests for API endpoints and database operations\n"
            f"4. Build performance testing scripts for load testing\n"
            f"5. Create validation scripts for data integrity and business rules\n"
            f"6. Include test data generators and mock objects\n"
            f"7. Set up automated testing configurations\n"
            f"8. Add coverage reporting and quality metrics\n\n"
            f"USE FileWriterTool to write this to '{code_folder}/testing_framework.md'\n"
            f"Generate all test files with comprehensive test suites."
        ),
        expected_output=(
            f"Testing framework saved in '{code_folder}/tests/' containing:\n"
            "- Unit tests for all critical functions\n"
            "- Integration tests for APIs and databases\n"
            "- Performance testing scripts\n"
            "- Data validation scripts\n"
            "- Test data generators and mocks\n"
            "- Automated testing configurations\n"
            "- Coverage reporting setup\n\n"
            "Use FileWriterTool to save this with real data, not placeholders."
        ),
        agent=coding_agent
    )

    # Task 15: Additional Analysis Task
    analysis_task = Task(
        description=(
            f"Analyze the following project document content and identify key project goals, risks, challenges, and potential improvements. "
            f"Your final answer MUST be structured as a Markdown report with sections for Summary, Risks, Strengths, and Opportunities.\n\n"
            f"Project Document Content:\n{pdf_content}"
        ),
        expected_output="A structured Markdown analysis report of the project.",
        agent=project_analyst
    )
    
    # Return all 15 tasks in the correct order
    return [
        # First agent tasks (project analysis) - 4 tasks
        project_context_task, 
        objective_task, 
        technical_task, 
        resource_task,
        # Second agent tasks (resource search) - 5 tasks
        multi_platform_discovery_task,
        code_repository_analysis_task,
        dataset_discovery_task,
        academic_paper_task,
        realtime_monitoring_task,
        # Third agent tasks (coding and development) - 5 tasks
        architecture_design_task,
        starter_template_task,
        custom_components_task,
        api_integration_task,
        testing_validation_task,
        # Additional analysis task - 1 task
        analysis_task
    ]

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
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced Custom CSS for professional, modern styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        line-height: 1.3;
        color: #1e293b;
    }
    
    /* Header Styling - Compact */
    .main-header {
        background: #005476 !important;
        padding: 1rem 2rem;
        margin: -3rem 0 1rem 0;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 8px 25px rgba(0, 84, 118, 0.15);
        position: relative;
        overflow: hidden;
        top: 0;
        z-index: 1001;
        width: 100%;
        max-width: 100%;
        left: 0;
        transform: none;
        text-align: center;
        display: block;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-title {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 0 3px 6px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
        opacity: 0.9;
    }
    
    /* Main Container - Compact */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem 2rem 1rem;
    }
    

    
    /* Card Styling - Compact */
    .stCard {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(20px);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 80px rgba(0,0,0,0.12);
    }
    
    /* Enhanced Button Styling - Compact */
    .stButton > button {
        background: linear-gradient(135deg, #005476 0%, #0077a3 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(0, 84, 118, 0.25);
        position: relative;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 84, 118, 0.35);
        background: linear-gradient(135deg, #004a6b 0%, #005476 100%);
    }
    
    /* File Upload Styling */
    .uploadedFile {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 3px dashed #3b82f6;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    
    .uploadedFile:hover {
        border-color: #1e40af;
        background: linear-gradient(135deg, #bfdbfe 0%, #93c5fd 100%);
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.25);
    }
    
    /* File Uploader Container Styling */
    .stFileUploader {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-color: #0284c7;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.2);
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1e40af, #3b82f6, #60a5fa);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Enhanced Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        font-weight: 600;
        color: #1e293b;
        padding: 1rem 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f1f5f9 0%, #dbeafe 100%);
        border-color: #3b82f6;
    }
    
    /* Enhanced Message Styling - Compact */
    .message {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: currentColor;
        opacity: 0.3;
    }
    
    .message-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border-color: #bbf7d0;
    }
    
    .message-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-color: #fde68a;
    }
    
    .message-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-color: #fecaca;
    }
    
    .message-info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border-color: #bfdbfe;
    }
    
    /* Content Box Styling - Compact */
    .content-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .content-box:hover {
        border-color: #3b82f6;
        box-shadow: 0 15px 50px rgba(0,0,0,0.08);
    }
    
    .box-title {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .box-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        border-radius: 2px;
    }
    
    /* File Info Grid */
    .file-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .info-item {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .info-item:hover::before {
        transform: scaleX(1);
    }
    
    .info-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
    
    .info-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .info-label {
        color: #64748b;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section Styling */
    .content-section {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .section-title {
        color: #1e293b;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        position: relative;
        padding-bottom: 0.75rem;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        border-radius: 2px;
    }
    
    /* Footer Styling - Compact */
    .footer {
        background: #005476 !important;
        color: white;
        text-align: center;
        padding: 1rem 2rem;
        margin: 0;
        border-radius: 0;
        box-shadow: 0 -5px 20px rgba(0, 84, 118, 0.1);
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        z-index: 1000;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain2" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.05)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.05)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain2)"/></svg>');
        opacity: 0.3;
    }
    
    .footer h4 {
        margin: 0 0 0.25rem 0;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .footer p {
        margin: 0;
        color: rgba(255,255,255,0.9);
        font-size: 0.8rem;
        position: relative;
        z-index: 1;
    }
    
    .footer small {
        display: block;
        margin-top: 0.25rem;
        opacity: 0.7;
        position: relative;
        z-index: 1;
        font-size: 0.7rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .main-subtitle {
            font-size: 1.1rem;
        }
        .main-container {
            padding: 0 1rem;
        }
    .file-info {
            grid-template-columns: 1fr;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Download Button Styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #005476 0%, #0077a3 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 84, 118, 0.3);
        margin: 0.5rem 0;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 84, 118, 0.4);
        background: linear-gradient(135deg, #004a6b 0%, #005476 100%);
    }
    
    /* Status Messages */
    .status-message {
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .status-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border-color: #bbf7d0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-color: #fde68a;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-color: #fecaca;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Custom Streamlit Element Styling */
    .st-emotion-cache-r44huj h1 {
        font-size: 2.75rem !important;
        font-weight: 700 !important;
        padding: -2.75rem 0px 1rem !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Header Section
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 AI-Powered PDF Analysis</h1>
        <p class="main-subtitle">Transform your documents into actionable insights with advanced AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    

    

    
    # Add some space above the file uploader
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to analyze",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Success Message
        st.markdown(f"""
        <div class="message message-success">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <strong>File Successfully Uploaded!</strong><br>
                    <span style="font-size: 0.9rem; opacity: 0.8;">{uploaded_file.name}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File Information Display
        st.markdown("""
        <h3 style="text-align: center; color: #1e293b; margin: 2rem 0 1.5rem 0; font-size: 1.5rem; font-weight: 700;">
            📊 Document Information
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-value">{uploaded_file.size / 1024:.1f} KB</div>
                <div class="info-label">File Size</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-value">PDF</div>
                <div class="info-label">File Type</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-value">Ready</div>
                <div class="info-label">Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add spacing between file info and analysis button
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        # Analysis Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                st.markdown("""
                <h3 style="text-align: center; color: #1e293b; margin: 2rem 0 1.5rem 0; font-size: 1.5rem; font-weight: 700;">
                    🤖 AI Analysis in Progress
                </h3>
                <div class="message message-info">
                    Our AI agents are analyzing your PDF document. This may take a few minutes.
                </div>
                """, unsafe_allow_html=True)
            
            # Read PDF content
            pdf_content = read_pdf_content(uploaded_file)
            
            if pdf_content.startswith("Error"):
                st.markdown(f"""
                <div class="status-message status-error">
                        ❌ <strong>Error:</strong> {pdf_content}
                </div>
                """, unsafe_allow_html=True)
                return
            
            if pdf_content.startswith("Warning"):
                st.markdown(f"""
                <div class="status-message status-warning">
                        ⚠️ <strong>Warning:</strong> {pdf_content}
                </div>
                """, unsafe_allow_html=True)
            

            
            with st.expander("📄 PDF Content Preview", expanded=False):
                    st.text_area("PDF Text", pdf_content, height=300, disabled=True)
            
            # Run CrewAI analysis
            try:
                with st.spinner("🤖 AI Agents are working on your document..."):
                    result = run_crew_analysis(pdf_content)
                
                # Success Message
                st.markdown("""
                <div class="status-message status-success">
                    ✅ <strong>Analysis Complete!</strong><br>
                    Your document has been successfully analyzed by our AI agents.
                </div>
                """, unsafe_allow_html=True)
                
                    # Analysis Results
                st.markdown("""
                <div class="content-section">
                    <h3 class="section-title">📊 Analysis Results</h3>
                        <p style="text-align: center; color: #64748b; margin-bottom: 1.5rem;">
                            Comprehensive insights and recommendations from our AI analysis
                        </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(result)
                
                    # File Check Section
                st.markdown("""
                <div class="content-section">
                        <h3 class="section-title">🔄 Generated Files</h3>
                        <p style="text-align: center; color: #64748b; margin-bottom: 1.5rem;">
                            Check and download the files generated by our AI agents
                        </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Check Generated Files Button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                        if st.button("🔄 Check Generated Files", type="secondary", use_container_width=True):
                            wait_for_files_and_refresh()
                
                    # Show Generated Files
                st.markdown("""
                <div class="content-section">
                        <h3 class="section-title">📁 Available Files</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Check for generated files
                output_folders = ["project_analysis_output", "resource_output", "code_output"]
                
                files_found = False
                for folder in output_folders:
                    folder_path = Path(folder)
                    if folder_path.exists():
                        # Get all markdown files in the main folder
                        main_files = list(folder_path.glob("*.md"))
                        
                        # Get all markdown files in subfolders (recursive)
                        subfolder_files = list(folder_path.rglob("*.md"))
                        
                        # Get all Python files in subfolders (recursive)
                        python_files = list(folder_path.rglob("*.py"))
                        
                        # Get all other common file types
                        other_files = list(folder_path.rglob("*.txt")) + list(folder_path.rglob("*.yaml")) + list(folder_path.rglob("*.yml"))
                        
                        # Combine all files
                        all_files = main_files + subfolder_files + python_files + other_files
                        
                        if all_files:
                            files_found = True
                            st.markdown(f"""
                                <div class="content-box">
                                    <h4 style="color: #1e293b; margin-bottom: 1rem; text-align: center;">
                                        📂 {folder.replace('_', ' ').title()}
                                    </h4>
                                """, unsafe_allow_html=True)
                            
                            # Group files by type for better organization
                            md_files = [f for f in all_files if f.suffix == '.md']
                            py_files = [f for f in all_files if f.suffix == '.py']
                            other_file_types = [f for f in all_files if f.suffix not in ['.md', '.py']]
                            
                            # Show markdown files first
                            if md_files:
                                    st.markdown("**📄 Markdown Files:**")
                                    for file in md_files:
                                        try:
                                            # Use safe file reading
                                            content = safe_read_file(file)
                                            
                                            # Show relative path for better organization
                                            relative_path = file.relative_to(folder_path)
                                            
                                            with st.expander(f"📄 {relative_path}", expanded=False):
                                                if content.startswith("Error reading file"):
                                                    st.error(content)
                                                else:
                                                    st.markdown(content)
                                            
                                            # Download button for each file
                                            if not content.startswith("Error"):
                                                st.download_button(
                                                    label=f"💾 Download {relative_path}",
                                                    data=content,
                                                    file_name=relative_path.name,
                                                    mime="text/markdown"
                                                )
                                        except Exception as e:
                                            st.error(f"❌ Error reading {file.name}: {str(e)}")
                                
                            # Show Python files
                            if py_files:
                                st.markdown("**🐍 Python Files:**")
                                for file in py_files:
                                    try:
                                        # Use safe file reading
                                        content = safe_read_file(file)
                                        
                                        # Show relative path for better organization
                                        relative_path = file.relative_to(folder_path)
                                        
                                        with st.expander(f"🐍 {relative_path}", expanded=False):
                                            if content.startswith("Error reading file"):
                                                st.error(content)
                                            else:
                                                    st.code(content, language=relative_path.suffix[1:])
                                        
                                        # Download button for each file
                                        if not content.startswith("Error"):
                                            st.download_button(
                                                label=f"💾 Download {relative_path}",
                                                data=content,
                                                file_name=relative_path.name,
                                                mime="text/plain"
                                            )
                                    except Exception as e:
                                        st.error(f"❌ Error reading {file.name}: {str(e)}")
                            
                            # Show other file types
                            if other_file_types:
                                st.markdown("**📁 Other Files:**")
                                for file in other_file_types:
                                    try:
                                        # Use safe file reading
                                        content = safe_read_file(file)
                                        
                                        # Show relative path for better organization
                                        relative_path = file.relative_to(folder_path)
                                        
                                        # Determine file type for display
                                        if file.suffix in ['.json', '.yaml', '.yml']:
                                            with st.expander(f"📁 {relative_path}", expanded=False):
                                                if content.startswith("Error reading file"):
                                                    st.error(content)
                                                else:
                                                    st.code(content, language=file.suffix[1:])  # Remove the dot
                                        else:
                                            with st.expander(f"📁 {relative_path}", expanded=False):
                                                if content.startswith("Error reading file"):
                                                    st.error(content)
                                                else:
                                                    st.text_area("File Content", content, height=200, disabled=True)
                                        
                                        # Download button for each file
                                        if not content.startswith("Error"):
                                            st.download_button(
                                                label=f"💾 Download {relative_path}",
                                                data=content,
                                                file_name=relative_path.name,
                                                mime="text/plain"
                                            )
                                    except Exception as e:
                                        st.error(f"❌ Error reading {file.name}: {str(e)}")
                            
                            # Show folder structure for code_output
                            if folder == "code_output":
                                st.markdown("**📂 Folder Structure:**")
                                try:
                                    # Get all directories and files recursively
                                    all_items = []
                                    for item in folder_path.rglob("*"):
                                        if item.is_file():
                                            all_items.append(f"📄 {item.relative_to(folder_path)}")
                                        elif item.is_dir():
                                            all_items.append(f"📁 {item.relative_to(folder_path)}/")
                                    
                                    # Sort items (folders first, then files)
                                    all_items.sort(key=lambda x: (x.startswith("📁"), x))
                                    
                                    # Display in a nice format
                                    for item in all_items:
                                        st.write(f"  {item}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error reading folder structure: {str(e)}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                
                if not files_found:
                    st.markdown("""
                    <div class="message message-warning">
                        ⚠️ <strong>No Generated Files Found</strong><br>
                        The agents may still be processing. Click 'Check Generated Files' to refresh.
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 Generated files will appear here once the agents complete their tasks.")
                
            except Exception as e:
                st.markdown(f"""
                <div class="status-message status-error">
                    ❌ <strong>Error during analysis:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 Make sure your API keys are properly configured in the .env file")
    
    # Close the main container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("""
    <div class="footer">
        <h4>🚀 AI-Powered Document Analysis</h4>
        <p>Powered by CrewAI 🤖 | Built with Streamlit 📱</p>
        <small>Advanced AI agents for comprehensive document insights and analysis</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


  
