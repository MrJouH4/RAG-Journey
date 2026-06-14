import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient

app = FastAPI(title="Agentic RAG Production Server")

# Initialize Tavily and Groq clients using environment variables
# This ensures zero local model downloads
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key or not tavily_api_key:
    raise RuntimeError("Missing required environment variables: GROQ_API_KEY or TAVILY_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)
tavily_client = TavilyClient(api_key=tavily_api_key)

# 1. Define Request/Response Schemas
class QueryRequest(BaseModel):
    question: str

class AgentResponse(BaseModel):
    route_chosen: str
    answer: str

# 2. Pydantic Schema for Router Decisions
class RouteDecision(BaseModel):
    destination: str = "Either 'web_search' for real-time/current/general events or 'internal_knowledge' for definitions/static concepts."

parser = JsonOutputParser(pydantic_object=RouteDecision)

# 3. Router Prompt Setup
router_prompt = ChatPromptTemplate.from_template(
    "You are an expert query router. Determine if the user's question requires real-time/current data from a web search "
    "or if it can be answered accurately using general static knowledge.\n"
    "Format Instructions:\n{format_instructions}\n"
    "User Question: {question}\n"
    "Decision JSON:"
)

router_chain = router_prompt | llm | parser

@app.get("/")
def read_root():
    return {"status": "Agentic RAG Server is running online"}

@app.post("/agentic-rag", response_model=AgentResponse)
async def run_agentic_rag(request: QueryRequest):
    try:
        # Step 1: Run the LLM Router
        decision = router_chain.invoke({
            "question": request.question,
            "format_instructions": parser.get_format_instructions()
        })
        
        destination = decision.get("destination", "internal_knowledge")
        
        # Step 2: Execute Routing Logic Based on Tool Selection
        if "web_search" in destination:
            # Trigger the Tavily web search tool dynamically
            search_result = tavily_client.search(query=request.question, max_results=2)
            
            # Synthesize search results with Groq
            synthesis_prompt = f"Synthesize a clear answer to the question using these search results:\n{search_result}\nQuestion: {request.question}"
            answer = llm.invoke(synthesis_prompt).content
            return AgentResponse(route_chosen="Tavily Web Search", answer=answer)
            
        else:
            # Fallback to pure LLM foundational knowledge
            answer = llm.invoke(request.question).content
            return AgentResponse(route_chosen="LLM Foundational Knowledge", answer=answer)
            
    except Exception as e:
        raise HTTPException(status_index=500, detail=str(e))