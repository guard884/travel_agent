import os
import random
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Optional, Dict, Any

# 1. LangChain Core (메시지, 툴, 프롬프트, 파서)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

# 2. LangChain Components (LLM, 임베딩, 벡터스토어, 문서 로더)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_tavily import TavilySearch

# 3. LangGraph (그래프 구조, 상태 및 메시지 관리, 메모리, 사전구축 모듈)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import ToolMessage

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
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

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
    # LLM이 도구 호출 및 대화 기록을 관리할 수 있도록 messages 상태만 유지합니다.
    messages: Annotated[List[BaseMessage], add_messages]

# ==========================================
# 3. 외부 툴(Tool) 연동 함수 (그대로 유지)
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

@tool 
def retrieve_travel_guide(destination: str) -> str:
    """여행 목적지의 관광 가이드 정보와 출처를 RAG DB에서 검색합니다."""
    docs = retriever.invoke(destination)
    results = []
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "출처 알 수 없음"))
        results.append(f"[출처: {file_name}] {doc.page_content}")
    return "\n- ".join(results) if results else "가이드 정보가 없습니다."

# 툴 리스트 생성
tools = [fetch_weather_info, fetch_exchange_rate, retrieve_travel_guide]

# ==========================================
# 4. 에이전트 노드 정의 (분석 및 생성 통합)
# ==========================================
def agent_node(state: TravelAgentState):
    """LLM이 문맥을 파악하여 도구를 사용할지, 아니면 최종 답변을 출력할지 결정합니다."""
    
    # 툴이 바인딩된 LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # 시스템 프롬프트 (최종 답변 양식 및 역할 부여)
    system_prompt = SystemMessage(content="""
        당신은 스마트 여행 비서입니다. 항상 한글로 대답하세요.
        사용자가 여행지에 대해 질문하면 제공된 도구(Tool)를 활용하여 '날씨', '환율', '가이드 정보'를 모두 검색하세요.
        필요한 도구를 모두 사용하여 정보를 수집한 후, 반드시 아래 양식에 맞게 최종 답변을 작성해야 합니다.

        [출력 형식]
        1. 요청 요약: (사용자의 요청 목적지 요약)
        2. 추천 일정: (가이드 추천 정보 기반, 출처를 반드시 기술, 없으면 없다고 할 것)
        3. 예산 및 환율: (제공된 환율 정보만 기재)
        4. 날씨와 준비물: (제공된 날씨 정보만 기재)
        5. 주의사항: (모르는 내용은 출발 전 재확인이 필요하다고 안내할 것)
    """)
    
    # 시스템 프롬프트와 현재까지의 메시지 내역을 LLM에 전달
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}



# 툴 리스트 생성 (기존과 동일)
tools = [fetch_weather_info, fetch_exchange_rate, retrieve_travel_guide]
# 이름으로 툴을 빠르게 찾기 위한 딕셔너리
tool_map = {tool.name: tool for tool in tools}

def tool_execution_node(state: TravelAgentState):
    """LLM이 요청한 도구(Tool)들을 수동으로 파싱하여 실행하고 결과를 반환합니다."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    tool_messages = []
    
    # LLM이 호출을 요청한 툴 목록(tool_calls) 순회
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # 요청한 툴이 유효한지 확인 후 실행
        if tool_name in tool_map:
            try:
                # 툴 실행
                result = tool_map[tool_name].invoke(tool_args)
                
                # 결과를 ToolMessage 객체로 감싸서 리스트에 추가
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                )
            except Exception as e:
                # 에러 발생 시 에러 메시지를 컨텐츠로 반환
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                )
        else:
            # 알 수 없는 툴을 호출했을 경우
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Unknown tool '{tool_name}'",
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
            )
            
    # 실행된 툴의 결과 메시지들을 상태에 추가
    return {"messages": tool_messages}

def route_next_step(state: TravelAgentState):
    """마지막 메시지를 확인하여 툴 호출 여부에 따라 다음 노드를 결정합니다."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # 마지막 메시지에 tool_calls가 존재하면 툴 실행 노드로 이동
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "execute_tools"
    
    # 툴 호출이 없다면 최종 답변 생성이 완료된 것이므로 그래프 종료
    return "end"

# ==========================================
# 5. 워크플로우 그래프 조립
# ==========================================
workflow = StateGraph(TravelAgentState) 

# 1) 노드 등록
workflow.add_node("agent", agent_node)
workflow.add_node("execute_tools", tool_execution_node)  # 👈 직접 작성한 툴 노드 등록

# 2) 엣지(Edge) 연결
workflow.add_edge(START, "agent")

# 3) 조건부 엣지(Conditional Edge) 연결: 직접 작성한 라우팅 로직 사용
workflow.add_conditional_edges(
    "agent",
    route_next_step,
    {
        "execute_tools": "execute_tools", # 툴 호출이 필요할 때
        "end": END                        # 최종 답변이 완료되었을 때
    }
)

# 4) 툴 실행이 끝나면 다시 에이전트로 돌아가서 툴의 결과를 보고 답변을 생성하도록 연결
workflow.add_edge("execute_tools", "agent")

# 5) 그래프 컴파일
app = workflow.compile()

# ==========================================
# 6. 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("Agentic 스마트 여행 비서 LangGraph 구동")
    print("==================================================")
    
    user_message = "방콕 여행정보 알려줘"
    initial_input = {
        "messages": [HumanMessage(content=user_message)]
    }
    
    # 과정(Stream) 출력 기능 활성화: 에이전트가 어떤 툴을 부르는지 눈으로 확인할 수 있습니다.
    for event in app.stream(initial_input, stream_mode="values"):
        last_message = event["messages"][-1]
        
        # LLM이 도구 호출을 결정했을 때
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"\n⚙️ 도구 실행 준비: {[tool['name'] for tool in last_message.tool_calls]}")
        
        # 최종 답변을 뱉었을 때 (content가 있고 도구 호출이 없을 때)
        elif last_message.content and not hasattr(last_message, 'tool_calls'):
            print("\n[ 최종 생성된 답변 ]\n")
            print(last_message.content)