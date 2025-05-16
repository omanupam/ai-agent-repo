from pathlib import Path
 
from agno.agent import Agent
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector
from agno.tools.reasoning import ReasoningTools
from agno.embedder.google import GeminiEmbedder
from sqlalchemy import create_engine, MetaData, Table
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.groq import Groq

import os

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
csvKnowledgeBase = CSVKnowledgeBase(
    path=Path("data"),
    vector_db=PgVector(
        db_url=db_url,
        table_name="cvs_document",
        embedder=GeminiEmbedder(),
    ),
    num_documents=5
)
 
pdfKnowledgeBase = PDFKnowledgeBase(
    path=Path("pdf"),
    vector_db=PgVector(
        db_url=db_url,
        table_name="pdf_document",
        embedder=GeminiEmbedder(),
    ),
    num_documents=5
)
engine = create_engine(db_url)

metadata = MetaData()
table = Table('pdf_document', metadata, autoload_with=engine)

#table.drop(engine, checkfirst=True)
#print("Dropped existing pdf_document table.")

csvKnowledgeBase.load(recreate=False)
pdfKnowledgeBase.load(recreate=False)

csv_agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    name="CSV Agent",
    description="You are an assistant that can help with general questions and tasks.",
    knowledge=csvKnowledgeBase,
    search_knowledge=True
)

yfinance_tools=[
    ReasoningTools(add_instructions=True),
    YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True),
],

web_agent=Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Search the web for information",
    name="Web Agent",
    description="You are an assistant that can help with general questions and tasks.",
    tools=[GoogleSearchTools()],
    markdown=True,
    instructions="Always include the sources of your information.",
    show_tool_calls=True,
)

finance_agent = Agent(
    tools=[yfinance_tools],
    role="Investment Analyst",
    name="Finance Agent",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)
 
pdf_agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    name="PDF Agent",
    description="You are an assistant that can help with general questions and tasks.",
    knowledge=pdfKnowledgeBase,
    search_knowledge=True
)

agent_team=Agent(
    team=[pdf_agent, csv_agent, finance_agent, web_agent],
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    name="Agent Team",
    instructions=[
        "For anime-related queries (titles, episodes, characters), use the CSV Agent.",
        "For cooking-related queries (recipes, ingredients), use the PDF Agent.",
        "For financial queries (stocks, fundamentals), use Finance Agent.",
        "For real-time information or unanswered questions, use Web Agent.",
        "Always try to answer first using specialized agents before falling back to web search.",
        "When using Web Agent, include [Web Result] in response.",
        "If a query is unclear, use Web Agent as default."
    ],
    show_tool_calls=True,
    markdown=True,
    search_knowledge=True
)
    
def main():
    print("Agent is ready! Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            agent_team.print_response(user_input, markdown=True, stream=True, show_full_reasoning=True, stream_intermediate_steps=True)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
 
if __name__ == "__main__":
    main()