# prompts.py for HyDE-Planner

def get_hyde_generation_prompt(query: str, language: str = "English") -> str:
    """
    [Improved Prompt] Generates a prompt for Phase 1: Hypothetical Document Generation (HyDE).
    
    This prompt asks the LLM to:
    1. Generate an expert-level hypothetical answer to the user's query.
    2. Brainstorm several "competing strategic hypotheses" for answering the query,
       rather than just a single, ideal answer.
    3. This provides richer material for the Phase 2 (Planning) model to critique and
       explore in greater depth.
    """
    
    return f"""
You are a **panel of senior strategy analysts** from McKinsey, BCG, and Bloomberg Intelligence.
Your task is to brainstorm **several key hypotheses** or **strategic mechanisms** from different perspectives to answer the following complex query. You must then write a detailed, structured document based on these hypotheses.

This document will be used by an AI planner to create a fact-checking and research plan. Therefore, it is crucial that each hypothesis is rich with specific, verifiable claims, key concepts, and related entities, even if they are speculative.

**Query: "{query}"**

---
Now, please generate the detailed hypothetical document based on the following instructions.

1.  **Present Diverse Hypotheses:** Present 2-3 distinct key hypotheses that could answer the query. Each hypothesis should represent a unique angle (e.g., a legal/regulatory angle, an economic/competitive angle, an operational/technical angle).
2.  **Detail Each Hypothesis:** For each hypothesis, elaborate on why it is plausible and what specific (speculative) facts, data points, or entities support it.
3.  **Structure Clearly:** Structure the document with a clear title and distinct sections.

**[Example Document Structure]**

# Analysis of Query: [Main Topic of Query]

## Hypothesis 1: [The First Key Hypothesis/Mechanism]
(A detailed explanation of this hypothesis. Include specific, verifiable claims)

## Hypothesis 2: [The Second Key Hypothesis/Mechanism]
(A detailed explanation from a different perspective than Hypothesis 1. Include specific, verifiable claims)

## Hypothesis 3: [The Third Key Hypothesis/Mechanism (e.g., The Non-Obvious 'Dark Horse' Angle)]
(The most unexpected or counter-intuitive hypothesis that, if true, would have a significant impact. Provide specific rationale and claims for it...)

Please write the document in {language}.
"""

def get_plan_reverse_engineering_prompt(hypothetical_document: str, use_arxiv: bool = False, use_finance: bool = False, use_local_search: bool = False, language: str = "English") -> str:
    """
    [개선된 프롬프트] 2단계: 계획 역설계 (PRE)를 위한 프롬프트를 생성합니다.
    
    이 프롬프트는 기획자 LLM에게 다음을 요청합니다:
    1. 1단계의 가상 문서를 비판적으로 분석하여 핵심적인 '가정(assumption)'과 '정보 격차(gap)'를 식별합니다.
    2. 기존 주장을 '검증'하는 쿼리를 생성합니다.
    3. 1단계 문서가 놓쳤을 가능성이 높은 '비직관적 통찰'이나 '대안적 가설'을 찾기 위한
       새로운 '탐색적 쿼리(exploratory queries)'를 생성합니다.
    """
    
    available_tools = ["google_search"]
    if use_arxiv:
        available_tools.append("arxiv_search")
    if use_finance:
        available_tools.append("yahoo_finance")
    if use_local_search:
        available_tools.append("local_search")
    tools_string = ", ".join(f'`{tool}`' for tool in available_tools)

    finance_claim_example = ""
    if use_finance:
        finance_claim_example = """
    {{
      "claim_id": 3,
      "claim_text": "A financial claim about a specific company.",
      "priority": "medium",
      "tool": "yahoo_finance",
      "verification_query": "CATL",
      "depends_on": []
    }}"""

    return f"""
You are a **Skeptical Chief of Research and Expert Research Planner**.
Your task is twofold:
1.  **Deconstruct & Verify:** Deconstruct the following document to create a plan to verify its core claims.
2.  **Critique & Expand:** Critically analyze what this document might be **missing** or what core assumption it relies on. Identify the most significant **non-obvious angle, alternative hypothesis, or strategic mechanism** (e.g., a specific legal loophole, a unique policy, a key government statement) that the document overlooks. Then, add **new exploratory search queries** to investigate this missing angle.

You have the following tools: {tools_string}.
Your plan must be a single JSON object.
Only use the tools listed above. For financial queries, use the company's stock ticker symbol (e.g., "TSLA", "AAPL") as the query for `yahoo_finance`. For academic papers, use `arxiv_search`.

**JSON Structure:**
{{
  "main_topic": "A brief summary of the document's main topic.",
  "critique_of_hypothesis": "A brief critique of the document's core assumption and what non-obvious strategic angle it is likely missing. (e.g., 'The document focuses on generic 'country of origin' rules, but misses the more specific, high-impact mechanism of a direct 'tariff exemption' policy for strategic investments.')",
  "claims_to_verify": [
    {{
      "claim_id": 1,
      "claim_text": "A core claim from the document that needs verification.",
      "priority": "medium",
      "tool": "google_search",
      "verification_query": "latest advancements in perovskite solar cell efficiency 2024",
      "depends_on": []
    }},
    {{
      "claim_id": 2,
      "claim_text": "A technical claim about a specific scientific breakthrough.",
      "priority": "high",
      "tool": "arxiv_search",
      "verification_query": "graphene battery energy density",
      "depends_on": []
    }}{finance_claim_example}
  ],
  "exploratory_queries": [
    {{
      "claim_id": 101,
      "claim_text": "(Exploratory) Investigate the missing non-obvious angle identified in the critique.",
      "priority": "high",
      "tool": "google_search",
      "verification_query": "economic impact of green hydrogen subsidies Europe"
    }}
  ]
}}

Here is the document to analyze:
---
{hypothetical_document}
---

Now, generate the JSON research plan. Please write the JSON research plan in {language}.
"""

def get_verification_and_synthesis_prompt(original_query: str, research_plan: dict, evidence: dict, language: str = "English") -> str:
    """
    [개선된 프롬프트] 3단계: 검증 및 종합을 위한 프롬프트를 생성합니다.
    
    이 프롬프트는 LLM에게 다음을 요청합니다:
    1. 수집된 '증거'에만 기반하여 최종적이고 검증된 답변을 종합합니다.
    2. 'exploratory_queries'(탐색적 쿼리)에서 나온 증거를 핵심적이고
       비직관적인 통찰로 특별히 주목하도록 합니다.
    3. 단순한 사실 나열이 아닌, 발견한 내용의 '전략적 함의(So What?)'를 중심으로
       답변을 구조화하도록 합니다.
       
    *참고: 이 함수의 시그니처는 'claims: list' 대신 'research_plan: dict'를 받도록
     변경되었습니다.
    """
    
    evidence_str = ""
    
    # 2단계의 '비판' 내용을 컨텍스트로 추가하여, 종합(3단계) AI가
    # '왜' 탐색적 쿼리가 실행되었는지 알도록 합니다.
    if "critique_of_hypothesis" in research_plan:
        evidence_str += f"--- Initial Critique from Planner (Phase 2) ---\n"
        evidence_str += f"{research_plan['critique_of_hypothesis']}\n\n"
    
    # 1. 검증 쿼리(claims_to_verify)에 대한 증거 처리
    evidence_str += "--- EVIDENCE FOR VERIFICATION CLAIMS ---\n"
    if "claims_to_verify" in research_plan:
        for claim in research_plan["claims_to_verify"]:
            claim_id = claim['claim_id']
            claim_text = claim['claim_text']
            # .get()을 사용하여 증거가 없는 경우를 안전하게 처리
            claim_evidence = evidence.get(claim_id, "No evidence found.")
            evidence_str += f"Claim #{claim_id}: {claim_text}\n"
            evidence_str += f"Evidence Found:\n---\n{claim_evidence}\n---\n\n"
    
    # 2. 탐색적 쿼리(exploratory_queries)에 대한 증거 처리 (더 중요)
    evidence_str += "--- EVIDENCE FOR EXPLORATORY QUERIES (High Priority Insight) ---\n"
    if "exploratory_queries" in research_plan:
        for claim in research_plan["exploratory_queries"]:
            claim_id = claim['claim_id']
            claim_text = claim['claim_text']
            claim_evidence = evidence.get(claim_id, "No evidence found.")
            evidence_str += f"Query #{claim_id}: {claim_text}\n"
            evidence_str += f"Evidence Found:\n---\n{claim_evidence}\n---\n\n"

    # 프롬프트 문자열 반환
    return f"""
You are a **Lead Strategist and Senior Research Analyst** tasked with producing a final, comprehensive **analysis** for a senior executive.
Your task is to provide a clear, data-driven answer to the user's original query based *only* on the evidence collected.
**Pay special attention to the "EVIDENCE FOR EXPLORATORY QUERIES,"** as this was designed to find the core, non-obvious insights that the initial hypothesis (Phase 1) may have missed. The "Initial Critique" provides context on what we were looking for.

Your report must adhere to the following rules:
1.  **Structure:** The report must include:
    * A clear **Title** (e.g., "Strategic Imperatives for X: An Analysis of Y").
    * An **Executive Summary** (1-2 paragraphs) that directly answers the query, summarizing the key strategic insights, *especially those from the exploratory research*.
    * A detailed **Main Body / In-Depth Analysis** that elaborates on the summary.

2.  **Evidence-Based Synthesis:** Your analysis must be **grounded *only* in the provided 'Evidence Found'.**
    * Synthesize information from all evidence sources (both verification and exploratory) to build a cohesive narrative.
    * **Do not use information from the initial claims (Phase 1) if it is not supported or verified by the evidence.**
    * **Do not invent facts, data, or sources** not present in the provided evidence.

3.  **Prioritize the "So What?" (Strategic Implication):**
    * Do not just list facts. Your primary value is to **connect the dots** and answer **"So What?"**
    * **Identify the core strategic mechanism or insight** (likely found in the 'Exploratory' evidence, e.g., a "tariff exemption mechanism").
    * Analyze the **competitive advantage, risk, or strategic benefit** of this finding (e.g., "This mechanism gives TSMC a 'Competitive Positioning Benefit' because..."). This analysis is the most critical part of your report.

4.  **Citation:** For each key point, synthesized claim, or inference, you **must cite the source(s)** found within the evidence. Use a simple citation format like `[Source: title of the search result]`.

5.  **Handle Nuance and Gaps:**
    * If evidence presents **conflicting information, you must highlight this nuance** rather than ignoring one side.
    * If the evidence is insufficient to answer a key part of the query (even after the exploratory search), explicitly state what information is missing in a concluding "Information Gaps" or "Areas for Further Research" section.

6.  **Professional Tone:** The language must be objective, professional, analytical, and confident.

Original User Query: "{original_query}"

---
CLAIMS, QUERIES, AND EVIDENCE ---
{evidence_str}
---
END OF CLAIMS, QUERIES, AND EVIDENCE ---

Now, please synthesize the final, verified analysis based *only* on the provided evidence, following all the rules above. Please write the final response in {language}.
"""

def get_query_decomposition_prompt(query: str, use_arxiv: bool = False, use_finance: bool = False, use_local_search: bool = False, language: str = "English") -> str:
    """
    Generates a prompt to decompose a complex query into several simpler sub-queries.
    """
    available_tools = ["google_search"]
    if use_arxiv:
        available_tools.append("arxiv_search")
    if use_finance:
        available_tools.append("yahoo_finance")
    if use_local_search:
        available_tools.append("local_search")
    tools_string = ", ".join(f'`{tool}`' for tool in available_tools)

    return f"""
You are an expert at breaking down complex questions into smaller, manageable, and searchable sub-queries.
Your task is to decompose the following user query into specific questions that can be independently searched to gather comprehensive information.
For each sub-query, you must specify which tool to use from the available list.

You have the following tools: {tools_string}.
- Use `yahoo_finance` for queries about a specific company's financial data (use the ticker symbol).
- Use `arxiv_search` for scientific or technical topics that might be in academic papers.
- Use `google_search` for all other general queries.

The output MUST be a single JSON object.
Example format:
{{
  "sub_queries": [
    {{
      "tool": "google_search",
      "query": "What are the main components of a lithium-ion battery?"
    }},
    {{
      "tool": "yahoo_finance",
      "query": "TSLA"
    }},
    {{
      "tool": "arxiv_search",
      "query": "graphene battery energy density"
    }}
  ]
}}

User Query: "{query}"

Now, generate the JSON object with the decomposed sub-queries. Please write the JSON object in {language}.
"""

def get_reflection_prompt(query: str, conversation_history: str, use_arxiv: bool = False, use_finance: bool = False, use_local_search: bool = False, language: str = "English") -> str:
    """
    Generates a prompt for the reflection step in a sequential search process.
    """
    available_tools = ["google_search"]
    if use_arxiv:
        available_tools.append("arxiv_search")
    if use_finance:
        available_tools.append("yahoo_finance")
    if use_local_search:
        available_tools.append("local_search")
    tools_string = ", ".join(f'`{tool}`' for tool in available_tools)

    return f"""
You are an autonomous research agent. Your goal is to answer the user's original query by sequentially searching for information.
You will be given the original query and the history of your previous searches and their results.
Your task is to reflect on the information gathered so far and decide on the next step.

You have the following tools: {tools_string}.
- Use `yahoo_finance` for queries about a specific company's financial data (use the ticker symbol).
- Use `arxiv_search` for scientific or technical topics that might be in academic papers.
- Use `google_search` for all other general queries.

1.  **Reflection:** Briefly explain your thought process. What have you learned? What information is still missing?
2.  **Next Action:** Decide whether you need to search for more information or if you have enough to answer the query.
    - If you need more information, provide the next specific search query and the tool to use.
    - If you are finished, set `is_final_answer` to `true` and `next_query` to `null`.

The output MUST be a single JSON object with the following structure:
{{
  "reflection": "Your brief thought process here.",
  "next_query": {{
    "tool": "google_search",
    "query": "Your next search query here."
  }},
  "is_final_answer": false
}}

**Example:**
{{
  "reflection": "I have found general information about the proposed tariffs. Now I need to find TSMC's latest financial results to assess the impact.",
  "next_query": {{
    "tool": "yahoo_finance",
    "query": "TSLA"
  }},
  "is_final_answer": false
}}

---
Original User Query: "{query}"
---
Conversation History:
{conversation_history}
---

Now, provide your reflection and the next action in the specified JSON format. Please write the JSON object in {language}.
"""

def get_synthesis_from_conversation_prompt(query: str, conversation_history: str, language: str = "English") -> str:
    """
    Generates a prompt to synthesize a final answer from a conversation history.
    """
    return f"""
You are a **Lead Strategist and Senior Research Analyst** tasked with producing a final, comprehensive **analysis** for a senior executive. The executive needs to understand not just *what* the facts are, but ***why they matter*** **and** ***how they connect***. Your task is to provide a clear, data-driven answer to the user's original query based *only* on the evidence found within the provided conversation history.

Your report must adhere to the following rules:
1.  **Structure:** The report must include:
    *   A clear **Title**.
    *   An **Executive Summary** (1-2 paragraphs) that directly answers the query, summarizing the key strategic insights.
    *   A detailed **Main Body / In-Depth Analysis** that elaborates on the summary.
2.  **Evidence-Based Synthesis:** Your analysis must be **grounded in the *evidence* (e.g., search results, verified facts) found within the provided 'Conversation History'.**
    *   **Synthesize information from multiple evidence sources** to build a cohesive narrative.
    *   **Logical inferences are permitted, but *only if* they are directly supported by the text of the evidence.** (e.g., If Source A states X and Source B states Y, you can infer Z, citing both).
    *   **Do not invent facts, data, or sources** not present in the provided history.
3.  **Citation:** For each key point, synthesized claim, or inference, you **must cite the specific source (e.g., search result title) mentioned *within* the conversation history**. Use a simple citation format like `[Source: title of the search result]`.
4.  **Analytical Depth & Nuance:**
    *   Do not just list facts. **Connect the dots** between different pieces of information to provide a complete picture (e.g., "This policy [Source A] directly addresses the market gap identified in [Source B].").
    *   If sources within the history present **conflicting information or different perspectives, you must highlight this nuance** rather than ignoring one side.
5.  **Clarity on Gaps:** If the evidence in the conversation history does not contain enough information to answer a part of the query, explicitly state what information is missing in a concluding "Information Gaps" or "Areas for Further Research" section.
6.  **Professional Tone:** The language should be objective, professional, analytical, and confident.

Original User Query: "{query}"
---
Conversation History:
{conversation_history}
---

Now, please synthesize the final, verified analysis based only on the evidence found within the provided conversation history, following all the rules above. Please write the final response in {language}.
"""

def get_synthesis_from_search_results_prompt(original_query: str, search_results: dict, language: str = "English") -> str:
    """
    Generates a prompt to synthesize a final answer from a collection of search results.
    """
    return f"""
You are a **Lead Strategist and Senior Research Analyst** tasked with producing a final, comprehensive **analysis** for a senior executive. The executive needs to understand not just *what* the facts are, but ***why they matter*** **and** ***how they connect***. Your task is to provide a clear, data-driven answer to the user's original query based *only* on the provided search results.

Your report must adhere to the following rules:
1.  **Structure:** The report must include:
    *   A clear **Title**.
    *   An **Executive Summary** (1-2 paragraphs) that directly answers the query, summarizing the key strategic insights.
    *   A detailed **Main Body / In-Depth Analysis** that elaborates on the summary.
2.  **Evidence-Based Synthesis:** Your analysis must be **grounded in the provided 'Search Results'.**
    *   **Synthesize information from multiple evidence sources** to build a cohesive narrative.
    *   **Logical inferences are permitted, but *only if* they are directly supported by the text.** (e.g., If Source A states X and Source B states Y, you can infer Z, citing both).
    *   **Do not invent facts, data, or sources** not present in the provided results.
3.  **Citation:** For each key point, synthesized claim, or inference, you **must cite the source(s)**. Use a simple citation format like `[Source: title of the search result]`.
4.  **Analytical Depth & Nuance:**
    *   Do not just list facts. **Connect the dots** between different pieces of information to provide a complete picture (e.g., "This policy [Source A] directly addresses the market gap identified in [Source B].").
    *   If sources present **conflicting information or different perspectives, you must highlight this nuance** rather than ignoring one side.
5.  **Clarity on Gaps:** If the provided search results do not contain enough information to answer a part of the query, explicitly state what information is missing in a concluding "Information Gaps" or "Areas for Further Research" section.
6.  **Professional Tone:** The language should be objective, professional, analytical, and confident.

Original User Query: "{original_query}"

--- SEARCH RESULTS ---
{search_results}
---
END OF SEARCH RESULTS ---

Now, please synthesize the final, verified analysis based only on the provided search results, following all the rules above. Please write the final response in {language}.
"""

def get_information_gap_prompt(original_query: str, first_pass_answer: str, previous_queries: list, use_arxiv: bool = False, use_finance: bool = False, use_local_search: bool = False, language: str = "English") -> str:
    """
    Generates a prompt to identify information gaps in a first-pass answer and create a search query.
    """
    available_tools = ["google_search"]
    if use_arxiv:
        available_tools.append("arxiv_search")
    if use_finance:
        available_tools.append("yahoo_finance")
    if use_local_search:
        available_tools.append("local_search")
    tools_string = ", ".join(f'`{tool}`' for tool in available_tools)
    previous_queries_str = "\n".join(f"- {q}" for q in previous_queries)

    return f"""
You are a critical analyst and researcher. You have been given an initial answer to a query and a list of the search queries used to generate it.
Your task is to identify the most important "information gaps" in the answer.
Based on those gaps, generate a list of specific search queries to find the missing information.
These new queries must be different from the ones already used.

You have the following tools: {tools_string}.
- Use `yahoo_finance` for queries about a specific company's financial data (use the ticker symbol).
- Use `arxiv_search` for scientific or technical topics that might be in academic papers.
- Use `google_search` for all other general queries.

The output MUST be a single JSON object with the following structure:
{{
  "information_gap_analysis": "A brief analysis of what key information is missing from the first-pass answer.",
  "search_queries_for_gap": [
    {{
      "tool": "google_search",
      "query": "The first specific, keyword-based search query."
    }},
    {{
      "tool": "yahoo_finance",
      "query": "GOOG"
    }}
  ]
}}

Original Query: "{original_query}"

Previously Used Search Queries:
---
{previous_queries_str}
---

First-Pass Answer:
---
{first_pass_answer}
---

Now, generate the JSON object identifying the information gap and a new list of search queries. Please write the JSON object in {language}.
"""

def get_synthesis_with_reflection_prompt(original_query: str, first_pass_answer: str, gap_search_results: dict, language: str = "English") -> str:
    """
    Generates a prompt to synthesize a final answer using an initial answer and new evidence found to fill an information gap.
    """
    return f"""
You are a **Lead Strategist and Senior Research Analyst**. You have already produced an initial report.
After reviewing it, you identified an information gap and conducted further research.
Your task is to create a new, final, and more comprehensive report by integrating the new evidence into your original analysis.

Your final report must adhere to the following rules:
1.  **Structure:** The report must include:
    *   A clear **Title**.
    *   An **Executive Summary** (1-2 paragraphs) that directly answers the query, summarizing the key strategic insights from the combined information.
    *   A detailed **Main Body / In-Depth Analysis** that elaborates on the summary, now enriched with the new evidence.
2.  **Evidence-Based Synthesis:** Your analysis must be **grounded in both the original report ('First-Pass Answer') and the new 'Evidence for Information Gap'.**
    *   **Integrate, Don't Just Append:** Seamlessly weave the new information into the original analysis to enhance and complete it.
    *   **Synthesize information from all available sources** to build a cohesive narrative.
    *   **Do not invent facts, data, or sources.**
3.  **Citation:** For each key point, especially those supported by the new evidence, you **must cite the source(s)**. Use a simple citation format like `[Source: title of the search result]`.
4.  **Analytical Depth & Nuance:**
    *   Do not just list facts. **Connect the dots** between different pieces of information to provide a complete picture.
    *   If sources present **conflicting information or different perspectives, you must highlight this nuance** rather than ignoring one side.
5.  **Clarity on Gaps:** If the combined evidence is still insufficient, explicitly state what information is missing in a concluding "Information Gaps" or "Areas for Further Research" section.
6.  **Professional Tone:** The language should be objective, professional, analytical, and confident.

Original User Query: "{original_query}"

--- FIRST-PASS ANSWER ---
{first_pass_answer}
---
END OF FIRST-PASS ANSWER ---

--- EVIDENCE FOR INFORMATION GAP ---
{gap_search_results}
---
END OF EVIDENCE FOR INFORMATION GAP ---

Now, please synthesize the final, comprehensive analysis by integrating the new evidence into the first-pass answer, following all the rules above. Please write the final response in {language}.
"""
