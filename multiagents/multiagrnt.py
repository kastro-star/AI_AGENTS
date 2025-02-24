
from agno.agent import Agent  
from agno.models.openai import OpenAIChat  
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader  
from agno.vectordb.pgvector import PgVector, SearchType  
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.ollama import Ollama  
from agno.tools.yfinance import YFinanceTools
from agno.models.openrouter import OpenRouter
from agno.tools.wikipedia import WikipediaTools
from agno.tools.google_maps import GoogleMapTools
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.qdrant import Qdrant
from agno.models.openrouter import OpenRouter



#vector db
embeddings = OllamaEmbedder(id="llama3.2:latest")
#vector db configuration
vector_db = Qdrant(
    collection="madurai_AI_FINAL_CONTENT",
    url="ENTER YOUR URL",
    api_key="ENTER YOUR API KEY",
    embedder=embeddings,
)
#knowledge base
knowledge_base = PDFKnowledgeBase(
    path="madurai_project.py/244 Submission.pdf",  
    vector_db=vector_db,
    reader=PDFReader(chunk=True) # Vector DB configuration
)


# knowledge_base.load(recreate=True, upsert=True) 

#knowledge base agent
knowledge_baseagent = Agent(
    model=OpenRouter(id="gpt-4o-mini", api_key="OPENROUTER API KEY"),
    role="you are a knowledge base agent",
    description="use this for every question user ask and you have to answer the question based on the knowledge base give the top 4 result",
    instructions=[
      """you have to answer the question based on the knowledge base give the top 4 result when ever user ask about the madurai city you have to give the 4 result based on the user search """
    ],
    knowledge=knowledge_base,  
    read_chat_history=True,  
    show_tool_calls=True,  
    markdown=True,  
    
)

#web search agent
web_search_agent =  Agent(
    tools=[GoogleSearchTools()],
    role="you are a news agent",
    description="""when every the user ask about the news you have to search the web for the information which is related to the user question and from their time what will be the current news from these give the top 4 news from the web
    if the user ask fro the hotel with their price range you have to give the top hotel with consider the rating and price range you have to give the name and link fro the hotels  always give the top rated at the first and follow next """,
    model=OpenRouter(id="gpt-4o-mini", api_key="OPENROUTER API KEY"),
    instructions=[
        "Given a topic by the user, respond with 4 latest news items about that topic.",
        "Search for 10 news items and select the top 4 unique items.",
        "Search in the last 24 hours",
    ],
    show_tool_calls=True,
    debug_mode=True,
)

# Finance agent
finance_agent = Agent(
    name="Finance AI agent",
    description="first understand the user question and select which agent will choose to their question and get the answer from the partifular agent ",
    model=OpenRouter(id="gpt-4o-mini", api_key="OPENROUTER API KEY"),  # Use the OpenRouter API key,  # Use the OpenRouter API key
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

#wikipedia agent
wikipedia_agent = Agent(
    name="wikipedia agent",
    description="search the web for the information about the wikipedia and give the live update to the user ",
    model=OpenRouter(id="gpt-4o-mini", api_key="OPENROUTER API KEY"),  # Use the OpenRouter API key
    tools=[WikipediaTools()], 
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
    )



#multi agent
multi_agent = Agent(
    team=[knowledge_baseagent, web_search_agent, finance_agent, wikipedia_agent],
    name="multi agent",
    description="""you are a multi agent you have to only say about the madurai city and you have to give the top 4 result based on the user search and
    you have to give the information with little bit information about the thing and you have to give the information with the link and the name of the thing """,
    model=OpenRouter(id="gpt-4o-mini", api_key="OPENROUTER API KEY"),  # Use the OpenRouter API key,  # Use the OpenRouter API key
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
)

question = input("Enter your question:")
multi_agent.print_response(question)











