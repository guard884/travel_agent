import os
import random
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
# 1. LangChain Core (메시지, 툴, 프롬프트, 파서)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 2. LangChain Components (LLM, 임베딩, 벡터스토어, 문서 로더)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 3. LangGraph (그래프 구조, 상태 및 메시지 관리, 메모리)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- 환경 변수 로드 (.env 파일에서 API Key 등을 불러옴) ---
load_dotenv()

# ==========================================
# 1. 초기 1회 문서 인덱싱 (Chroma 벡터 스토어 빌드)
# ==========================================
def init_vector_store():
    data_dir = "data"
    
    # 1. data 폴더가 없거나 폴더 내에 텍스트 파일이 없는 경우를 대비한 초기 세팅
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        # 테스트용 샘플 파일 생성
        with open(os.path.join(data_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write(
                "도쿄 가이드: 신주쿠는 맛집이 많으며 교통의 요지입니다. 도쿄 타워 근처 야경을 추천합니다.\n"
                "파리 가이드: 에펠탑 근처 피크닉하기 좋은 장소가 많습니다. 루브르 박물관은 사전 예약이 필수입니다.\n"
                "뉴욕 가이드: 타임스퀘어 야경과 센트럴 파크 산책을 추천합니다.\n"
                "오사카 가이드: 도톤보리의 길거리 음식과 유니버셜 스튜디오 재팬이 유명합니다.\n"
                "런던 가이드: 대영박물관은 무료이며, 템즈강 주변 런던아이 탑승을 추천합니다."
            )
    # 2. DirectoryLoader를 사용하여 data 폴더 안의 모든 .txt 파일 로드
    # glob="*.txt"를 통해 txt 확장자만 필터링하고, 내부적으로 TextLoader를 사용하도록 지정합니다.
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()
    
    # 3. 로드된 문서가 없을 경우의 안전장치
    if not documents:
        raise ValueError(f"'{data_dir}' 폴더 안에 로드할 .txt 파일이 없습니다. 파일을 추가해주세요.")

    # 4. 문서 분할 및 벡터 DB 저장
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")


vectorstore = init_vector_store()
# 검색기 설정 (관련도가 높은 문서 2개를 가져오도록 설정)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})




from langchain_tavily import TavilySearch

search_weather = TavilySearch(
    max_results=3,
    topic="general",               # 또는 "news", "finance" 등
    include_answer=True,           # 답변 포함 여부
    include_raw_content=False,     # 원본 내용 포함 여부
    include_images=False,          # 이미지 포함 여부
    search_depth="basic",          # "basic" 또는 "advanced"
    # include_domains=[
    #     "https://weather.daum.net/",
    #     "https://www.weatheri.co.kr/" ,
    #     "https://finance.daum.net/exchanges"
    # ],
    exclude_domains=None            # 필요하면 제외 도메인 지정 가능
)
# ==========================================
# 2. 상태(State) 정의
# ==========================================

class TravelAgentState(TypedDict):
    # 1. 대화 맥락 및 툴 호출 결과 추적
    # add_messages 리듀서를 사용하여 기존 메시지 목록에 새 메시지가 자동으로 병합되도록 합니다.
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. 사용자 입력
    # 현재 처리 중인 사용자의 원본 질문이나 요청을 저장합니다.
    user_input: str
    
    # 3. 중간 프로세스 데이터 (추출된 여행 정보)
    destination: Optional[str]
    parsed_intent: Optional[str]
    
    # 4. 외부 툴 연동 결과 데이터 통합
    # 날씨, 환율, RAG 검색 결과 등 각 노드에서 수집된 컨텍스트 데이터를 저장합니다.
    weather_data:  Optional[str]
    exchange_rate: Optional[str]
    rag_context:   Optional[List[str]]
   
    
    # 5. 제어 흐름 상태 (조건부 에지 라우팅용)
    next_step: Optional[str]


# ==========================================
# 3. 외부 툴(Tool) 연동 함수
# ==========================================

@tool  
def fetch_weather_info(destination: str) -> str:
    """특정 도시의 현재 날씨 정보를 검색합니다."""
    try:
        search_query = f"{destination} 현재 날씨 기온"
        result_weather = search_weather.invoke(search_query)
        
        if not result_weather['answer'] or len(result_weather['results']) < 1:
            return f"'{destination}'의 날씨 정보를 찾을 수 없습니다."
        
        # 첫 번째 결과에서 날씨 정보 추출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
        answer = llm.invoke(f"""
                        {result_weather['answer']} 를 설명없이 한문장으로 번역해줘.

                                """)
        weather_info = f"""
        {destination} 날씨 정보:
        {answer.content}

        출처: {result_weather['results'][0]['url']}
        """
        return weather_info
        
    except Exception as e:
        return f"날씨 정보 검색 중 오류가 발생했습니다: {str(e)}"
@tool 
def fetch_exchange_rate(destination: str) ->str:
    """여행 목적지 국가의 통화에 맞춘 현재 환율(KRW 기준) 정보를 가져옵니다."""  # 👈 추가된 부분
    try:
        search_query = f"{destination} 현재 환율"
        result_currency = search_weather.invoke(search_query)
    
        if not result_currency['answer'] or len(result_currency['results']) < 1:
            return f"'{destination}' 의 현재 환율 정보를 찾을 수 없습니다."
        
        # 첫 번째 결과에서 날씨 정보 추출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
        answer = llm.invoke(f"""
                            {result_currency['answer']} 설명없이 한문장으로 번역해줘.
                                """)
        currency_info = f"""
        {destination} 환율 정보:
        {answer.content}

        출처: {result_currency['results'][0]['url']}
        """
        
        return currency_info
    except Exception as e:
        return f"날씨 정보 검색 중 오류가 발생했습니다: {str(e)}"

@tool 
def retrieve_travel_guide(destination: str) -> List[str]:
    """초기화한 RAG Retriever를 활용하여 여행 목적지의 가이드 정보와 출처를 검색합니다."""
    query = f"{destination}"
    docs = retriever.invoke(query)
    
    results = []
    for doc in docs:
        # 1. Document 객체의 metadata에서 'source' (파일 경로)를 가져옵니다.
        source_path = doc.metadata.get("source", "출처 알 수 없음")
        
        # 2. 긴 경로(예: C:/data/tokyo_guide.txt) 대신 파일명(tokyo_guide.txt)만 추출합니다.
        file_name = os.path.basename(source_path)
        
        # 3. 출처와 텍스트 내용을 알아보기 쉽게 결합합니다.
        formatted_info = f"[출처: {file_name}] {doc.page_content}"
        results.append(formatted_info)
    print("?????????",results)    
    return results

# ==========================================
# 4. 노드(Node) 함수 구현
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
        parsed_intent = result.get("parsed_intent", "greeting")
    except Exception as e:
        print(f"파싱 에러: {e}")
        destination = "알 수 없음"
        parsed_intent = "greeting"

    # 목적지를 찾지 못했을 경우 바로 종료 처리
    if destination == "알 수 없음":
        return {
            "destination": "",
            "parsed_intent": parsed_intent,
            "messages": [AIMessage(content="어느 도시로 여행을 떠나고 싶으신가요?")],
            "next_step": "end" # 툴 실행 없이 종료
        }
    
    result={
        "destination": destination,
        "parsed_intent": parsed_intent,
        "next_step": "run_tools" # 목적지를 찾았으므로 툴 실행 노드로 이동
    }
   
    return result


def rag_search_node(state: TravelAgentState):
    print("-------------------->")
    """추출된 목적지를 바탕으로 RAG 검색을 수행합니다."""
    destination = state.get("destination", "알 수 없음")
    
    # 목적지를 모르면 검색 생략
    if destination == "알 수 없음":
        return {"rag_context": "목적지를 파악할 수 없어 가이드 정보를 검색하지 못했습니다."}
    
    # 사용자 원본 문장 대신 추출된 'destination' 키워드로 문서 검색
    docs = retriever.invoke(destination)
    results = []
    for doc in docs:
        # 1. Document 객체의 metadata에서 'source' (파일 경로)를 가져옵니다.
        source_path = doc.metadata.get("source", "출처 알 수 없음")
        
        # 2. 긴 경로(예: C:/data/tokyo_guide.txt) 대신 파일명(tokyo_guide.txt)만 추출합니다.
        file_name = os.path.basename(source_path)
        
        # 3. 출처와 텍스트 내용을 알아보기 쉽게 결합합니다.
        formatted_info = f"[출처: {file_name}] {doc.page_content}"
        results.append(formatted_info)
    
    rag_info = "\n- ".join(results) if results else "가이드 정보가 없습니다."
    
    return {"rag_context": rag_info}


def tool_execution_node(state: TravelAgentState):
    """추출된 목적지를 바탕으로 외부 도구(@tool)들을 실행하고 결과를 State에 추가합니다."""
    destination = state.get("destination")
    
    # 1. 외부 툴(@tool) 실행 및 결과 가져오기
    weather = fetch_weather_info.invoke({"destination": destination})
    exchange = fetch_exchange_rate.invoke({"destination": destination})
    guide = retrieve_travel_guide.invoke({"destination": destination}) 
    
    # 2. 기존 State에 있던 tool_context를 가져옵니다. (없으면 빈 딕셔너리)
    current_tool_context = state.get("tool_context") or {}
    
    # 3. 기존 툴 컨텍스트를 유지하면서 새로운 데이터를 안전하게 누적(추가)합니다.
    result = {
        **state,
        "weather_data": weather,
        "exchange_rate": exchange,
        "rag_context": guide,
        "next_node": "generate_response"  # 앞서 정의한 제어 흐름 키 이름(next_node)으로 통일
    }

   
    # 4. AgentState의 키 구조와 정확히 일치하는 딕셔너리를 반환하여 State를 업데이트합니다.
    return result




def generate_response_node(state: TravelAgentState):
    """수집된 툴 데이터(날씨, 환율, RAG)를 바탕으로 LLM을 통해 최종 답변을 생성합니다."""
    
    # 1. State에서 필요한 데이터 추출
    destination = state.get("destination", "알 수 없는 목적지")
    weather = state.get("weather_data")
    exchange = state.get("exchange_rate")
    rag = state.get("rag_context")
    
    # 2. 데이터 전처리 (None 방지 및 포맷팅)
    # 날씨
    if weather:
       
        weather_str =weather

    else:
        weather_str = "날씨 정보를 가져오지 못했습니다."
        
    # 환율
    if exchange:
       
        exchange_str = exchange
    else:
        exchange_str = "환율 정보를 가져오지 못했습니다."

    # RAG (리스트 형태인 경우 문자열로 합침)
    if isinstance(rag, list) and len(rag) > 0:
        rag_info = "\n- ".join(rag)
    elif isinstance(rag, str):
        rag_info = rag
    else:
        rag_info = "가이드 정보가 없습니다."

    # 3. LLM 프롬프트 정의
    # 3. LLM 프롬프트 정의 (엄격한 통제 적용)
    PLAN_SYSTEM_PROMPT = """당신은 제공된 데이터만 기반으로 답변을 정리하는 엄격한 여행 요약 비서입니다.

⚠️ [절대 준수 규칙]
1. 반드시 아래 [수집된 정보]에 포함된 내용만 사용하여 답변을 작성하세요.
2. 당신이 기존에 학습한 사전 지식은 절대 섞지 마세요.
3. [수집된 정보]의 내용이 부족하여 특정 항목을 채울 수 없다면, 절대 임의로 지어내지 말고 "수집된 정보에 해당 내용이 부족합니다."라고 명확히 기재하세요.
4. '여행 계획을 세워라'가 아닌 '수집된 정보를 주어진 양식에 맞게 요약하라'는 것이 당신의 임무입니다.
5. 모든정보에 출처가 있으만 반드시 출처를 기술하세요.

[수집된 정보]
- 목적지: {destination}
- 날씨: {weather}
- 환율: {exchange}
- 가이드 추천 정보: {rag_info}

[출력 형식]
1. 요청 요약: (사용자의 요청 목적지 요약)
2. 추천 일정: (반드시 '가이드 추천 정보'에 있는 장소와 내용만 나열할 것. 출처를 반드시기술할것. 없으면 없다고 할 것)
3. 예산 및 환율: (제공된 환율 정보만 기재)
4. 날씨와 준비물: (제공된 날씨 정보만 기재)
5. 주의사항: (모르는 내용은 출발 전 재확인이 필요하다고 안내할 것)
"""

    # 4. LLM 체인 구성 및 실행
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template=PLAN_SYSTEM_PROMPT, 
        # 프롬프트에 구멍 뚫어놓은 변수들을 명시
        input_variables=["destination", "weather", "exchange", "rag_info"] 
    )
    chain = prompt | llm | parser

    # invoke에 딕셔너리 형태로 변수들을 매핑해서 전달
    try:
        final_text = chain.invoke({
            "destination": destination,
            "weather": weather_str,
            "exchange": exchange_str,
            "rag_info": rag_info
        })
    except Exception as e:
        print(f"LLM 답변 생성 중 에러 발생: {e}")
        final_text = "죄송합니다. 답변을 생성하는 과정에서 오류가 발생했습니다."

    # 5. 최종 결과를 State에 반영 (messages 리스트에 AIMessage 추가)
    result = {
        "messages": [AIMessage(content=final_text)],
        "next_node": "END" # 마지막 노드이므로 그래프 종료를 알림
    }
    
  
    return result
# ==========================================
# 5. 조건부 에지 (라우팅 로직)
# ==========================================
def route_next_step(state: TravelAgentState):
    """상태(state)의 'next_step' 값을 확인하여 다음 이동할 노드를 결정합니다."""
    return state.get("next_step", "end")

# ==========================================
# 6. 워크플로우 그래프 설계 조립
# ==========================================
workflow = StateGraph(TravelAgentState) 

# 1) 노드 등록
workflow.add_node("analyze", analyze_input_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("execute_tools", tool_execution_node)
workflow.add_node("generate", generate_response_node)

# 2) 에지(Edge) 연결
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "rag_search")
workflow.add_edge("rag_search", "generate")
workflow.add_conditional_edges(
    "generate",
    route_next_step,
    {
        "run_tools": "execute_tools",
        
        "end": END  # 👈 추가: 알 수 없는 목적지일 경우 그래프 종료
    }
)

workflow.add_edge("execute_tools", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
#app = workflow.compile(checkpointer=memory)
app = workflow.compile()


# ==========================================
# 7. 그래프 실행 트리거
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("스마트 여행 비서 LangGraph 구동")
    print("==================================================")
    
    # Thread ID를 지정하여 대화 세션을 유지할 수 있도록 설정
    config = {"configurable": {"thread_id": "travel_session_1"}}
    
    user_message = "방콕 여행정보알려줘.."
    initial_input = {
        "user_input": user_message,
        "messages": [HumanMessage(content=user_message)]
    }
    
    final_output = app.invoke(initial_input, config=config)
    
    print("\n[ 최종 생성된 답변 ]\n")
    print(final_output["messages"][-1].content)