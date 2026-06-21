import os
import random
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Optional, Dict, Any

# 1. LangChain Core (메시지, 툴, 프롬프트, 파서)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

# 2. LangChain Components (LLM, 임베딩, 벡터스토어, 문서 로더)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_tavily import TavilySearch

# 3. LangGraph (그래프 구조, 상태 및 메시지 관리, 메모리)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- 환경 변수 로드 ---
load_dotenv()

# ==========================================
# 1. 초기 1회 문서 인덱싱 (Chroma 벡터 스토어 빌드)
# ==========================================
def init_vector_store():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write(
                "도쿄 가이드: 신주쿠는 맛집이 많으며 교통의 요지입니다. 도쿄 타워 근처 야경을 추천합니다.\n"
                "파리 가이드: 에펠탑 근처 피크닉하기 좋은 장소가 많습니다. 루브르 박물관은 사전 예약이 필수입니다.\n"
                "뉴욕 가이드: 타임스퀘어 야경과 센트럴 파크 산책을 추천합니다.\n"
                "방콕 가이드: 왓 아룬 사원의 야경이 아름답고 카오산 로드에서 길거리 음식을 즐기기 좋습니다.\n"
                "오사카 가이드: 도톤보리의 길거리 음식과 유니버셜 스튜디오 재팬이 유명합니다.\n"
                "런던 가이드: 대영박물관은 무료이며, 템즈강 주변 런던아이 탑승을 추천합니다."
            )
    
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"'{data_dir}' 폴더 안에 로드할 .txt 파일이 없습니다.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

vectorstore = init_vector_store()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

search_weather = TavilySearch(
    max_results=3,
    topic="general",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    search_depth="basic"
)

# ==========================================
# 2. 상태(State) 정의
# ==========================================
class TravelAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # RAG 검색 결과를 저장하기 위한 상태 추가
    rag_context: Optional[str]

# ==========================================
# 3. 외부 툴(Tool) 연동 함수 (RAG 제외)
# ==========================================
@tool  
def fetch_weather_info(destination: str) -> str:
    """특정 도시의 현재 날씨 및 기온 정보를 검색합니다."""
    try:
        search_query = f"{destination} 현재 날씨 기온"
        result_weather = search_weather.invoke(search_query)
        if not result_weather['answer'] or len(result_weather['results']) < 1:
            return f"'{destination}'의 날씨 정보를 찾을 수 없습니다."
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer = llm.invoke(f"{result_weather['answer']} 를 설명없이 한문장으로 번역해줘.")
        return f"{destination} 날씨 정보:\n{answer.content}\n출처: {result_weather['results'][0]['url']}"
    except Exception as e:
        return f"날씨 정보 검색 중 오류가 발생했습니다: {str(e)}"

@tool 
def fetch_exchange_rate(destination: str) -> str:
    """여행 목적지 국가의 통화에 맞춘 현재 환율(KRW 기준) 정보를 가져옵니다."""
    try:
        search_query = f"{destination} 현재 환율"
        result_currency = search_weather.invoke(search_query)
        if not result_currency['answer'] or len(result_currency['results']) < 1:
            return f"'{destination}' 의 현재 환율 정보를 찾을 수 없습니다."
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer = llm.invoke(f"{result_currency['answer']} 설명없이 한문장으로 번역해줘.")
        return f"{destination} 환율 정보:\n{answer.content}\n출처: {result_currency['results'][0]['url']}"
    except Exception as e:
        return f"환율 정보 검색 중 오류가 발생했습니다: {str(e)}"

# RAG 툴이 노드로 변경되었으므로 툴 리스트에는 날씨와 환율만 포함됩니다.
tools = [fetch_weather_info, fetch_exchange_rate]
tool_map = {tool.name: tool for tool in tools}


# ==========================================
# 4. 노드(Node) 함수 정의
# ==========================================

# 👈 1. 새롭게 추가된 RAG 검색 노드
def rag_search_node(state: TravelAgentState):
    """사용자의 입력을 바탕으로 RAG 검색을 우선 수행합니다."""
    messages = state.get("messages", [])
    # 가장 마지막에 입력된 사용자 질문 가져오기
    user_query = messages[-1].content
    
    docs = retriever.invoke(user_query)
    results = []
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "출처 알 수 없음"))
        results.append(f"[출처: {file_name}] {doc.page_content}")
    
    rag_info = "\n- ".join(results) if results else "가이드 정보가 없습니다."
    
    # 검색된 정보를 State의 rag_context에 저장합니다.
    return {"rag_context": rag_info}


def agent_node(state: TravelAgentState):
    """사전 검색된 RAG 정보와 외부 도구를 결합하여 답변을 생성합니다."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # State에 저장된 RAG 컨텍스트 가져오기 (없으면 기본 메시지)
    rag_context = state.get("rag_context", "가이드 정보를 찾을 수 없습니다.")
    
    # 시스템 프롬프트에 RAG 결과물 주입
    system_prompt = SystemMessage(content=f"""
        당신은 스마트 여행 비서입니다. 항상 한글로 대답하세요.
        
        [사전 검색된 로컬 가이드 정보]
        {rag_context}
        
        사용자가 여행지에 대해 질문하면, 먼저 위 [사전 검색된 로컬 가이드 정보]를 확인하세요.
        추가로 필요한 '날씨', '환율' 정보는 제공된 도구(Tool)를 활용하여 검색하세요.
        필요한 정보를 모두 수집한 후, 반드시 아래 양식에 맞게 최종 답변을 작성해야 합니다.

        [출력 형식]
        1. 요청 요약: (사용자의 요청 목적지 요약)
        2. 추천 일정: ([사전 검색된 로컬 가이드 정보] 기반, 출처를 반드시 기술, 없으면 없다고 할 것)
        3. 예산 및 환율: (도구로 제공된 환율 정보만 기재)
        4. 날씨와 준비물: (도구로 제공된 날씨 정보만 기재)
        5. 주의사항: (모르는 내용은 출발 전 재확인이 필요하다고 안내할 것)
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def tool_execution_node(state: TravelAgentState):
    """LLM이 요청한 도구들을 실행하고 결과를 반환합니다."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name))
            except Exception as e:
                tool_messages.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call_id, name=tool_name))
        else:
            tool_messages.append(ToolMessage(content=f"Error: Unknown tool", tool_call_id=tool_call_id, name=tool_name))
            
    return {"messages": tool_messages}


def route_next_step(state: TravelAgentState):
    """마지막 메시지를 확인하여 툴 호출 여부에 따라 분기합니다."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "execute_tools"
    return "end"

# ==========================================
# 5. 워크플로우 그래프 조립
# ==========================================
workflow = StateGraph(TravelAgentState) 

# 1) 노드 등록
workflow.add_node("rag_search", rag_search_node)  # 👈 RAG 노드 등록
workflow.add_node("agent", agent_node)
workflow.add_node("execute_tools", tool_execution_node)

# 2) 엣지(Edge) 연결
# 시작하면 무조건 RAG 검색을 먼저 실행합니다.
workflow.add_edge(START, "rag_search")
# RAG 검색이 끝나면 에이전트로 넘어갑니다.
workflow.add_edge("rag_search", "agent")

# 3) 조건부 엣지(Conditional Edge) 연결
workflow.add_conditional_edges(
    "agent",
    route_next_step,
    {
        "execute_tools": "execute_tools",
        "end": END
    }
)

# 4) 툴 실행이 끝나면 다시 에이전트로 복귀
workflow.add_edge("execute_tools", "agent")

# 5) 그래프 컴파일
app = workflow.compile()

# ==========================================
# 6. 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("Agentic 스마트 여행 비서 LangGraph 구동 (RAG Node 분리형)")
    print("==================================================")
    
    user_message = "방콕 여행정보 알려줘"
    initial_input = {
        "messages": [HumanMessage(content=user_message)]
    }
    
    for event in app.stream(initial_input, stream_mode="values"):
        # RAG 검색이 완료되어 상태가 업데이트 되었는지 확인 가능 (원하면 출력 추가)
        
        last_message = event["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"\n⚙️ 도구 실행 준비: {[tool['name'] for tool in last_message.tool_calls]}")
        
        elif last_message.content and not hasattr(last_message, 'tool_calls'):
            print("\n[ 최종 생성된 답변 ]\n")
            print(last_message.content)