import os
import random
import json
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
    print(">>>>>>>>>> fetch_weather_info")
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
    print(">>>>>>>>>> fetch_exchange_rate")
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
def analyze_input_node(state: TravelAgentState):
    """LLM을 사용하여 사용자의 입력을 분석하고 목적지를 추출합니다."""
    user_input = state.get("user_input", "")
  
    ANALYZE_SYSTEM_PROMPT = """
        당신은 여행 비서 에이전트의 요청 분석 노드입니다.
        사용자 입력에서 여행 목적지를 추출해 JSON 객체로 반환하세요.

        반드시 아래의 JSON 키를 포함해야 합니다:
        - "destination": 도시 또는 지역명 (예: 도쿄, 파리). 알 수 없으면 "알 수 없음"으로 설정.
        - "parsed_intent": 여행 정보 요청이면 "need_all_info", 단순 인사면 "greeting"으로 설정.
        사용자 입력: {user_input}
"""

    # JSON 출력을 강제하는 LLM 세팅
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
        response_format={"type": "json_object"}
    )
    parser = JsonOutputParser()
    prompt = PromptTemplate(template=ANALYZE_SYSTEM_PROMPT, input_variables=["user_input"])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"user_input": user_input})
        destination = result.get("destination", "알 수 없음")
        
    except Exception as e:
        print(f"파싱 에러: {e}")
        destination = "알 수 없음"
       

    # 목적지를 찾지 못했을 경우 바로 종료 처리
    if destination == "알 수 없음":
        return {
            "destination": "",
            "messages": [AIMessage(content="어느 도시로 여행을 떠나고 싶으신가요?")],
            "next_step": "end" # 툴 실행 없이 종료
        }
    
    result={
        "destination": destination,
        "next_step": "rag_search" # 목적지를 찾았으므로 툴 실행 노드로 이동
    }
   
    return result
def analyze_input_node(state: TravelAgentState):
    """사용자 입력에서 여행 목적지를 추출합니다."""
    messages = state.get("messages", [])
    user_input = messages[-1].content
    
    # JSON 출력을 강제하여 목적지 이름만 깔끔하게 추출
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
        response_format={"type": "json_object"}
    )
    
    prompt = f"""
    당신은 여행 비서입니다. 다음 사용자 입력에서 여행 목적지(도시, 지역, 국가명 등)를 추출하세요.
    목적지를 찾을 수 없다면 "알 수 없음"으로 설정하세요.
    반드시 아래 JSON 형식으로 반환해야 합니다:
    {{"destination": "추출된 목적지"}}
    
    사용자 입력: {user_input}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        result = json.loads(response.content)
        destination = result.get("destination", "알 수 없음")
    except Exception as e:
        print(f"목적지 파싱 에러: {e}")
        destination = "알 수 없음"
        
    # 콘솔에서 확인하기 위한 출력
    print(f"\n📍 [분석 완료] 추출된 목적지: {destination}")
    
    return {"destination": destination}


# 👈 1. 새롭게 추가된 RAG 검색 노드
def rag_search_node(state: TravelAgentState):
    print(">>>>>>>>>> rag_search_node")
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
    
    rag_context = state.get("rag_context", "가이드 정보를 찾을 수 없습니다.")
    destination = state.get("destination", "알 수 없음")
    
    system_prompt = SystemMessage(content=f"""
        당신은 스마트 여행 비서입니다. 항상 한글로 대답하세요.
        현재 사용자의 여행 목적지는 '{destination}' 입니다.
        
        [사전 검색된 로컬 가이드 정보]
        {rag_context}
        
        먼저 위 [사전 검색된 로컬 가이드 정보]를 확인하세요.
        추가로 필요한 '{destination}'의 '날씨', '환율' 정보는 제공된 도구(Tool)를 활용하여 검색하세요.
        필요한 정보를 모두 수집한 후, 반드시 아래 양식에 맞게 최종 답변을 작성해야 합니다.

   

        [출력 형식]
        1. 요청 요약: (사용자의 요청 목적지 요약)
        2. 추천 일정: (반드시 '가이드 추천 정보'에 있는 장소와 내용만 나열할 것. 출처를 반드시 기술. 없으면 없다고 할 것)
        3. 예산 및 환율: (제공된 환율 정보만 기재 출처를 반드시 기술)
        4. 날씨와 준비물: (제공된 날씨 정보만 기재 출처를 반드시 기술)
        5. 주의사항: (모르는 내용은 출발 전 재확인이 필요하다고 안내할 것)
    """)
    
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_execution_node(state: TravelAgentState):
    
    """LLM이 요청한 도구들을 실행하고 결과를 반환합니다."""
    print(">>>>>>>>>> agent_node")
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
    print(">>>>>>>>>> ",tool_messages)        
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
workflow.add_node("analyze", analyze_input_node)    # 👈 새로 추가된 분석 노드
workflow.add_node("rag_search", rag_search_node)  # 👈 RAG 노드 등록
workflow.add_node("agent", agent_node)
workflow.add_node("execute_tools", tool_execution_node)

# 2) 엣지(Edge) 연결
workflow.add_edge(START, "analyze")                 # 시작 -> 분석
workflow.add_edge("analyze", "rag_search")          # 분석 -> RAG 검색
workflow.add_edge("rag_search", "agent")            # RAG 검색 -> 에이전트

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
memory = MemorySaver()
workflow.add_edge("execute_tools", "agent")
#app = workflow.compile(checkpointer=memory)
# 5) 그래프 컴파일
app = workflow.compile()

# ==========================================
# 6. 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("Agentic 스마트 여행 비서 LangGraph 구동")
    print("(대화를 종료하려면 '종료', 'quit', 'q' 중 하나를 입력하세요.)")
    print("==================================================")
    
    # Thread ID를 지정하여 대화 세션을 유지 (메모리가 연결되어 있다면 이전 대화를 기억합니다)
    config = {"configurable": {"thread_id": "travel_session_1"}}
    
    while True:
        # 1. 사용자로부터 계속 입력을 받음
        user_message = input("\n👤 : ")
        
        # 2. 종료 조건 설정
        if user_message.strip().lower() in ['종료', 'quit', 'exit', 'q']:
            print("\n👋 스마트 여행 비서를 종료합니다. 즐거운 여행 되세요!")
            break
            
        # 빈 입력 무시
        if not user_message.strip():
            continue
            
        # 3. LangGraph 상태 업데이트용 입력 데이터 구성
        # (이전 코드에서 user_input 키를 사용했다면 함께 넘겨줍니다)
        initial_input = {
            "user_input": user_message, 
            "messages": [HumanMessage(content=user_message)]
        }
        
        # 4. 그래프 실행 및 답변 출력
        try:
            # app.invoke를 통해 그래프 실행 (중간 로그 없이 최종 결과만 받음)
            final_output = app.invoke(initial_input, config=config)
            
            print("\n🤖 [ 스마트 여행 비서 ]")
            print(final_output["messages"][-1].content)
            
        except Exception as e:
            print(f"\n⚠️ 오류가 발생했습니다: {e}")