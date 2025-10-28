```mermaid
graph TD
    subgraph "Traditional Methods"
        direction LR
        
        subgraph "Method 1: Reflection-based (ReAct) - 'The Explorer'"
            direction TB
            A_Start["User Query"] --> A_T1["Thought: What should I do first?"]
            A_T1 --> A_A1["Action: Search for X"]
            A_A1 --> A_O1["Observation: Found Y"]
            A_O1 --> A_T2["Thought: Okay, now what?"]
            A_T2 --> A_Loop[...]
            A_Loop --> A_End("Final Answer")
            style A_Loop fill:#f9f,stroke:#333,stroke-width:2px
        end

        subgraph "Method 2: Query Decomposition - 'The List Maker'"
            direction TB
            B_Start["User Query"] --> B_Decompose{"Decompose into Sub-Questions"}
            B_Decompose --> B_Q1["Sub-Q1"]
            B_Decompose --> B_Q2["Sub-Q2"]
            B_Decompose --> B_Q3["Sub-Q3"]
            
            B_Q1 --> B_S1["Search 1"]
            B_Q2 --> B_S2["Search 2"]
            B_Q3 --> B_S3["Search 3"]

            subgraph "Parallel & Independent Execution"
                direction LR
                B_S1
                B_S2
                B_S3
            end

            B_S1 & B_S2 & B_S3 --> B_Synthesize["Synthesize Results"]
            B_Synthesize --> B_End("Final Answer")
        end
    end

    subgraph "Proposed Method: HyDE-Planner - 'The Navigator / Architect'"
        direction TB
        C_Start["User Query"] --> C_HyDE["1. Generate Hypothetical Document (The Destination)"]
        C_HyDE --> C_Plan{"2. Create Structured Plan (The Blueprint)"}
        
        subgraph "Structured & Prioritized Plan"
            direction TB
            C_Plan --> C_P1["Priority 1: Verify Core Claim A"]
            C_P1 --> C_P2["Priority 2: Analyze Factor B (Depends on A)"]
            C_P2 --> C_P3["Priority 3: Find Supporting Evidence C"]
        end

        C_P1 --> C_E1["Execute & Verify P1"]
        C_E1 --> C_P2
        C_P2 --> C_E2["Execute & Verify P2"]
        C_E2 --> C_P3
        C_P3 --> C_E3["Execute & Verify P3"]
        
        C_E1 & C_E2 & C_E3 --> C_Synthesize["3. Synthesize Verified Insights"]
        C_Synthesize --> C_End("Actionable Insight")
    end

    style C_End fill:#cff,stroke:#333,stroke-width:2px
```

### 다이어그램 설명

이 다이어그램은 세 가지 방법론의 핵심적인 차이를 시각적으로 보여줍니다.

1.  **ReAct (탐험가):** `Thought -> Action -> Observation`의 순환적인 구조로 표현하여, 명확한 목표 없이 계속해서 다음 단계를 탐색하는 모습을 보여줍니다. 이는 사용자가 설명한 "어디로 가야 할까?"를 계속 고민하는 탐험가와 같습니다.

2.  **Query Decomposition (리스트 메이커):** 하나의 질문을 여러 개의 독립적인 하위 질문으로 나누고, 이를 병렬적으로 처리하는 모습을 보여줍니다. 각 질문 간의 관계나 우선순위가 없는 단순 '분해'의 특징을 나타냅니다.

3.  **HyDE-Planner (네비게이터/설계자):**
    *   **가설 문서 생성 (목적지 설정):** 가장 먼저 최종 목표의 청사진을 그립니다.
    *   **구조화된 계획 (청사진):** 단순히 질문을 나누는 것을 넘어, **우선순위**와 **의존 관계**(A를 알아야 B를 검증)를 가진 구조적인 계획을 수립합니다.
    *   **체계적인 실행:** 계획에 따라 순차적이고 체계적으로 검증을 수행하여 최종적으로 '실행 가능한 인사이트'를 도출합니다. 이는 방황을 최소화하는 네비게이터이자, 구조를 설계하는 설계자의 모습과 일치합니다.