# HyDE-Planner 프로젝트 개발 현황 요약

## 1. 핵심 기능 구현 완료 (초기 상태)

- 연구 계획서에 제안된 3단계(가상 문서 생성, 계획 역설계, 실행 및 검증) 아키텍처의 핵심 로직이 `run_hyde_planner.py`와 `prompts.py`를 통해 완벽하게 구현됨.
- Streamlit 기반의 웹 데모를 통해 각 단계의 결과를 시각적으로 확인할 수 있는 프로토타입 완성.

## 2. 실행 로그 저장 기능 추가

- **요청 사항:** 매 실행마다 추론 및 실행 기록을 JSON 파일로 저장하여 나중에 분석할 수 있도록 요청.
- **작업 내용:**
    - `run_hyde_planner.py`를 수정하여, 실행 시마다 타임스탬프가 포함된 이름의 JSON 로그 파일을 생성하도록 기능 추가.
    - 로그 파일은 `Hydeplanner/demo/logs/` 디렉토리에 저장됨.
    - 로그에는 사용자 쿼리, 가상 문서, 실행 계획, 수집된 근거, 최종 답변 등 전 과정의 데이터가 포함됨.

## 3. 금융 뉴스 검색 기능 추가 (`yfinance`)

- **요청 사항:** 금융 관련 질문의 정확도를 높이기 위해 `yfinance` 라이브러리를 이용한 뉴스 검색 기능 추가 요청.
- **작업 내용:**
    - `yfinance` 라이브러리 설치 및 `requirements.txt`에 의존성 추가.
    - 주식 티커(ticker)로 뉴스를 검색하는 `yfinance_search.py` 모듈 생성.
    - `prompts.py`를 수정하여 LLM 플래너가 `google_search` (기존 `bing_search`)와 `yfinance_news` 두 가지 도구를 인지하고, 상황에 맞는 도구를 선택하도록 프롬프트 엔지니어링 수행.
    - `run_hyde_planner.py`의 실행 로직을 수정하여, 계획(JSON)에 명시된 `tool`에 따라 적절한 검색 함수를 동적으로 호출하도록 변경.

## 4. API 키 관리 및 환경 변수 설정

- **문제점:** 기존 코드에 API 키(LLM, 검색 API)를 직접 설정하거나 로드하는 기능이 없어 하드코딩되거나 누락된 상태였음.
- **작업 내용:**
    - `.env` 파일에서 설정을 안전하게 로드하기 위해 `python-dotenv` 라이브러리 추가.
    - `settings.py`의 `Environment` 클래스를 리팩토링하여, 외부에서 API 키를 주입받아 각 API 클라이언트를 초기화하도록 구조 변경.
    - `run_hyde_planner.py`의 시작점에서 `.env` 파일을 로드하고, 필요한 키(LLM 키, 검색 API 키 등)를 `Environment` 클래스에 전달하도록 수정.

## 5. Bing Search API를 Google Search API로 교체

- **문제점:** Bing Search API가 지원 종료됨에 따라 대안 필요.
- **작업 내용:**
    - `google-api-python-client` 라이브러리 설치 및 `requirements.txt` 업데이트.
    - Google Custom Search API를 사용하는 `google_search.py` 모듈 신규 생성.
    - 기존의 불필요한 `bing_search.py` 모듈 삭제.
    - `settings.py`와 `run_hyde_planner.py`를 수정하여 Bing 관련 키 대신 Google API 키와 CSE ID를 사용하도록 변경.
    - `prompts.py`를 수정하여 LLM 플래너에게 `bing_search` 대신 `google_search` 도구를 사용하도록 지시.
    - 사용자가 `.env` 파일에 `GOOGLE_API_KEY`와 `GOOGLE_CSE_ID`를 설정하도록 최종 안내 완료.

## 6. 멀티-LLM 지원 기능 추가 (OpenAI & Google Gemini)

- **요청 사항:** OpenAI의 `gpt-5-mini-2025-08-07` 모델과 Google의 `gemini-2.5-pro` 모델 중 사용자가 선택하여 사용할 수 있도록 기능 추가 요청.
- **작업 내용:**
    - `google-generativeai` 라이브러리 설치 및 `requirements.txt`에 추가.
    - Streamlit UI에 두 모델을 선택할 수 있는 드롭다운 메뉴를 추가.
    - `.env` 파일에서 `OPENAI_API_KEY`와 `GEMINI_API_KEY`를 모두 읽어오도록 `run_hyde_planner.py`의 `initialize_environment` 함수 수정.
    - `settings.py`를 수정하여 `gemini_api_key` 필드를 추가하고, 클라이언트 초기화 로직을 `run_hyde_planner.py`로 이전.
    - `run_hyde_planner.py`의 `call_llm` 함수를 리팩토링하여, 사용자가 선택한 모델 이름에 따라 OpenAI 또는 Gemini API를 동적으로 호출하는 분기 로직 구현.
    - 전체 실행 흐름을 `asyncio`를 사용하여 비동기적으로 처리하도록 수정.

## 7. 다양한 연구 방법론 지원 및 비교 기능 추가

- **요청 사항:** HyDE-Planner의 성능을 다른 접근 방식과 비교하고, 사용자 쿼리의 특성에 따라 가장 적합한 방법론을 선택할 수 있는 기능 추가 요청.
- **작업 내용:**
    - `run_hyde_planner.py`를 대대적으로 리팩토링하여, 여러 연구 방법론을 독립적으로 실행하고 그 결과를 비교할 수 있는 프레임워크를 구축.
    - **추가된 방법론:**
        - `HyDE-Planner`: 기존의 핵심 3단계(가설 생성 → 계획 수립 → 검증) 방법론.
        - `Priority-HyDE-Planner`: 계획(plan)에 포함된 검증 항목(claim)의 우선순위(`high`, `medium`, `low`)에 따라 실행 순서를 정하는 버전.
        - `2-Step HyDE-Planner`: 초기 검증 결과를 바탕으로 중간 요약을 생성하고, 이를 기반으로 가설 문서를 다시 생성하여 계획을 정교화하는 심화 버전.
        - `Direct Search`: 사용자의 쿼리를 추가적인 처리 없이 직접 검색 엔진에 입력하는 가장 기본적인 베이스라인.
        - `Query Decomposition Search`: 복잡한 쿼리를 여러 개의 하위 쿼리로 분해하여 각각 검색하고 결과를 종합하는 방법론.
        - `Sequential-Reflection Search`: 검색을 한 번에 끝내지 않고, 이전 검색 결과를 바탕으로 "생각(reflection)" 단계를 거쳐 다음 검색 쿼리를 동적으로 생성하는 반복적 방법론.
    - Streamlit UI에 각 방법론을 선택하거나 "Compare All" 옵션으로 모든 방법론을 순차적으로 실행할 수 있는 멀티-셀렉트 드롭다운 메뉴 추가.
    - 각 방법론의 실행 로직을 별도의 `async` 함수(예: `run_hyde_planner`, `run_direct_search` 등)로 모듈화하여 코드 구조 개선.

## 8. Jina AI API 연동을 통한 콘텐츠 추출 기능 강화

- **문제점:** `requests`와 `BeautifulSoup`을 이용한 기본적인 웹페이지 스크래핑 방식은 동적 콘텐츠나 복잡한 구조의 사이트에서 원하는 정보를 정확히 추출하는 데 한계가 있었음.
- **작업 내용:**
    - `hyde_search_module.py`에 Jina AI의 `Reader API`를 연동하는 로직 추가.
    - 사용자가 UI에서 "Use Jina AI API" 옵션을 선택하면, URL의 메인 콘텐츠를 Markdown 형식으로 변환하여 반환받음으로써 정보의 정확성과 가독성을 크게 향상시킴.
    - Jina AI API 사용량 제어를 위해 분당 120회로 호출을 제한하는 `RateLimiter` 클래스를 구현.
    - `.env` 파일에 `JINA_API_KEY`를 추가하고, `settings.py`와 `run_hyde_planner.py`에서 이를 로드하여 사용하도록 설정.

## 9. JSON 응답 오류 수정 기능 추가 (`json_repair`)

- **문제점:** LLM이 생성하는 실행 계획(plan)이 때때로 사소한 JSON 형식 오류(예: 마지막 쉼표 누락)를 포함하여 `json.loads()`에서 파싱 에러를 발생시키는 경우가 있었음.
- **작업 내용:**
    - `json_repair` 라이브러리를 설치하고 `requirements.txt`에 추가.
    - `run_hyde_planner.py`의 `phase_2_reverse_engineer_plan` 함수에서 `json.loads()` 대신 `json_repair.loads()`를 사용하도록 변경.
    - 이를 통해 LLM이 생성한 JSON의 사소한 문법적 오류를 자동으로 수정하여, 시스템의 안정성과 성공률을 높임.

## 10. 비용 추정 및 성능 추적 기능 추가

- **요청 사항:** 각 LLM API 호출에 따른 예상 비용과 총 실행 시간을 추적하여 사용자에게 보여주는 기능 추가 요청.
- **작업 내용:**
    - `run_hyde_planner.py`에 `calculate_cost` 함수를 추가하여, 사용된 모델과 입/출력 토큰 수를 기반으로 API 호출 비용을 추정.
    - 각 방법론 실행 함수 내에 시작 및 종료 시간을 기록하여 총 실행 시간을 계산.
    - 실행 완료 후, Streamlit UI에 "Execution Time", "Total Tokens", "Estimated Cost" 정보를 요약하여 표시하는 `display_tracking_info` 함수를 구현 및 적용.
