# run_hyde_planner.py
import streamlit as st
import json
import asyncio
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
import google.generativeai as genai
import pandas as pd
from typing import Tuple

from prompts import (
    get_hyde_generation_prompt,
    get_plan_reverse_engineering_prompt,
    get_verification_and_synthesis_prompt,
    get_query_decomposition_prompt,
    get_reflection_prompt,
    get_synthesis_from_conversation_prompt,
    get_synthesis_from_search_results_prompt,
    get_information_gap_prompt,
    get_synthesis_with_reflection_prompt
)
from hyde_search_module import google_search
from arxiv_search_module import arxiv_search
from finance_search_module import finance_search
from local_search_module import LocalSearch
from pdf_processing_module import process_pdf_with_embeddings
from settings import Environment
from ingest import run_ingestion
import json_repair

# --- Helper Functions ---

@st.cache_resource
def initialize_environment():
    """Initializes and caches the application environment."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path, override=True)
    return Environment(
        main_api_key=os.getenv("OPENAI_API_KEY"),
        api_base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model_name=os.getenv("GEMINI_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        jina_api_key=os.getenv("JINA_API_KEY")
    )

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimates the cost of an LLM call based on token usage."""
    pricing = {
        "gemini-2.5-pro": {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000},
        "gemini-2.5-flash": {"prompt": 0.30 / 1_000_000, "completion": 2.50 / 1_000_000},
    }
    model_pricing = pricing.get(model_name)
    if not model_pricing:
        return 0.0
    return (prompt_tokens * model_pricing["prompt"]) + (completion_tokens * model_pricing["completion"])

def display_tracking_info(info: dict):
    """Displays the tracking info in the Streamlit UI."""
    st.info(
        f"- **Execution Time:** {info.get('duration', 0):.2f} seconds\n"
        f"- **Total Tokens:** {info.get('total_tokens', 0)}\n"
        f"- **Estimated Cost:** ${info.get('total_cost', 0):.6f}"
    )

async def call_llm(env: Environment, model_name: str, prompt: str) -> Tuple[str, int, int]:
    """
    Calls the appropriate LLM and returns (text, prompt_tokens, completion_tokens).
    """
    text, p_tokens, c_tokens = "", 0, 0
    try:
        if "gpt" in model_name:
            client = AsyncOpenAI(api_key=env.main_api_key, base_url=env.api_base_url)
            response = await client.completions.create(
                model=model_name, max_tokens=4096, prompt=prompt, timeout=3600
            )
            text = response.choices[0].text
            usage = response.usage
            p_tokens = usage.prompt_tokens
            c_tokens = usage.completion_tokens
        elif "gemini" in model_name:
            genai.configure(api_key=env.gemini_api_key)
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async(prompt)
            text = response.text
            usage = response.usage_metadata
            p_tokens = usage.prompt_token_count
            c_tokens = usage.candidates_token_count
        else:
            text = f"Error: Unknown model provider for '{model_name}'."
    except Exception as e:
        st.error(f"An error occurred during the LLM call for {model_name}: {e}")
        text = f"Error: {e}"
    return text, p_tokens, c_tokens

# --- Methodology-Specific Runners ---

async def run_hyde_planner(env: Environment, model: str, query: str, time_period: str, search_depth: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, log_data: dict):
    st.header("HyDE-Planner")
    with st.spinner("Running HyDE-Planner... This involves multiple LLM calls and searches."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        doc, p, c = await phase_1_generate_hypothetical_document(env, model, query)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1_hypothetical_document"] = doc
        with st.expander("Phase 1: Hypothetical Document", expanded=False): st.markdown(doc)

        plan, p, c = await phase_2_reverse_engineer_plan(env, model, doc, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2_research_plan"] = plan
        if plan:
            with st.expander("Phase 2: Research Plan", expanded=False): st.json(plan)
            evidence = phase_3_execute_plan_and_verify(env, plan, time_period, search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
            log_data["phases"]["3_collected_evidence"] = evidence
            if evidence:
                with st.expander("Phase 3: Execution & Verification", expanded=False): st.json(evidence)
                final_answer, p, c = await phase_4_synthesize_final_answer(env, model, query, plan, evidence)
                tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
                log_data["phases"]["4_final_answer"] = final_answer
                if final_answer:
                    with st.expander("Phase 4: Final Answer", expanded=True): 
                        st.markdown(final_answer)
                        if st.button('Copy report text', key='hyde_planner_copy'):
                            st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("HyDE-Planner finished.")
        display_tracking_info(tracking)

async def run_direct_search(env: Environment, model: str, query: str, time_period: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_local_search: bool, log_data: dict):
    st.header("Direct Search")
    with st.spinner("Running Direct Search..."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        search_results = google_search(env, query, time_period=time_period, use_jina_api=use_jina_api, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
        if use_local_search:
            try:
                local_search_tool = LocalSearch()
                local_results = local_search_tool.search(query)
                search_results["local_search"] = local_results
            except FileNotFoundError as e:
                st.warning(f"Local search index not found. Skipping. Error: {e}")
            except Exception as e:
                st.error(f"An error occurred during local search: {e}")

        with st.expander("Search Results", expanded=False):
            st.json(search_results)
            log_data["phases"]["direct_search_results"] = search_results

        if search_results:
            final_answer, p, c = await phase_5_synthesize_answer_from_evidence(env, model, query, search_results)
            tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
            log_data["phases"]["final_answer"] = final_answer
            if final_answer:
                with st.expander("Final Answer", expanded=True):
                    st.markdown(final_answer)
                    if st.button('Copy report text', key='direct_search_copy'):
                        st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("Direct Search finished.")
        display_tracking_info(tracking)

async def run_query_decomposition_search(env: Environment, model: str, query: str, time_period: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_local_search: bool, log_data: dict):
    st.header("Query Decomposition Search")
    with st.spinner("Running Query Decomposition..."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        prompt = get_query_decomposition_prompt(query, use_local_search=use_local_search)
        response_text, p, c = await call_llm(env, model, prompt)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        sub_queries = json_repair.loads(response_text).get("sub_queries", [])
        
        if not sub_queries:
            st.warning("Could not decompose the query."); return
        log_data["phases"]["1_decomposed_queries"] = sub_queries
        with st.expander("Decomposed Sub-Queries", expanded=False): st.json(sub_queries)

        evidence = {}
        local_search_tool = LocalSearch() if use_local_search else None

        for sub_query in sub_queries:
            tool = sub_query.get("tool", "google_search")
            query_text = sub_query.get("query")

            if not query_text:
                continue

            st.write(f"  - Executing: `{query_text}` with tool `{tool}`")

            search_results = None
            if tool == "google_search":
                search_results = google_search(env, query_text, time_period=time_period, use_jina_api=use_jina_api, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "arxiv_search":
                search_results = arxiv_search(query_text)
            elif tool == "yahoo_finance":
                search_results = finance_search(query_text)
            elif tool == "local_search" and local_search_tool:
                try:
                    search_results = local_search_tool.search(query_text)
                except Exception as e:
                    st.error(f"Local search failed for '{query_text}': {e}")
                    search_results = {"error": str(e)}
            else:
                st.warning(f"Tool '{tool}' is not recognized or enabled. Skipping.")
                search_results = {"error": f"Tool '{tool}' is not recognized or enabled."}
            
            evidence[json.dumps(sub_query)] = search_results

        with st.expander("Collected Evidence", expanded=False):
            st.json(evidence)
            log_data["phases"]["2_collected_evidence"] = evidence

        if evidence:
            final_answer, p, c = await phase_5_synthesize_answer_from_evidence(env, model, query, evidence)
            tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
            log_data["phases"]["final_answer"] = final_answer
            if final_answer:
                with st.expander("Final Answer", expanded=True):
                    st.markdown(final_answer)
                    if st.button('Copy report text', key='query_decomposition_copy'):
                        st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("Query Decomposition finished.")
        display_tracking_info(tracking)

async def run_sequential_reflection_search(env: Environment, model: str, query: str, time_period: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, log_data: dict):
    st.header("Sequential-Reflection Search")
    with st.spinner("Running Sequential-Reflection Search... This may take several steps."):
        start_time = time.time()
        log_data["phases"] = {"conversation_history": []}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}
        conversation_history = ""
        local_search_tool = LocalSearch() if use_local_search else None

        for i in range(5): # Limit steps
            prompt = get_reflection_prompt(query, conversation_history, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
            response_text, p, c = await call_llm(env, model, prompt)
            tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
            action = json_repair.loads(response_text)
            
            with st.expander(f"Step {i+1}: Thought Process", expanded=False): st.json(action)
            
            if action.get("is_final_answer", False) or not action.get("next_query"):
                break
            
            next_query = action["next_query"]
            tool = action.get("tool", "google_search")
            search_results = {}

            if tool == "google_search":
                search_results = google_search(env, next_query, time_period=time_period, use_jina_api=use_jina_api, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "local_search" and local_search_tool:
                 try:
                    search_results = local_search_tool.search(next_query)
                 except Exception as e:
                    st.error(f"Local search failed for '{next_query}': {e}")
                    search_results = {"error": str(e)}
            
            step_log = {"step": i + 1, "reflection": action.get("reflection"), "query": next_query, "search_results": search_results}
            log_data["phases"]["conversation_history"].append(step_log)
            conversation_history += f"\n\n--- Step {i+1} ---\nQuery: {next_query}\nResults: {json.dumps(search_results)}"

        prompt = get_synthesis_from_conversation_prompt(query, conversation_history)
        final_answer, p, c = await call_llm(env, model, prompt)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["final_answer"] = final_answer
        with st.expander("Final Answer", expanded=True):
            st.markdown(final_answer)
            if st.button('Copy report text', key='sequential_reflection_copy'):
                st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("Sequential-Reflection Search finished.")
        display_tracking_info(tracking)

async def run_priority_hyde_planner(env: Environment, model: str, query: str, time_period: str, search_depth: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, log_data: dict):
    st.header("Priority-HyDE-Planner")
    with st.spinner("Running Priority-HyDE-Planner... Claims will be executed in priority order."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        doc, p, c = await phase_1_generate_hypothetical_document(env, model, query)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1_hypothetical_document"] = doc
        with st.expander("Phase 1: Hypothetical Document", expanded=False): st.markdown(doc)

        plan, p, c = await phase_2_reverse_engineer_plan(env, model, doc, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2_research_plan"] = plan
        if plan:
            with st.expander("Phase 2: Research Plan (Sorted by Priority)", expanded=False): st.json(plan)
            
            # phase_3 automatically sorts claims now
            evidence = phase_3_execute_plan_and_verify(env, plan, time_period, search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
            log_data["phases"]["3_collected_evidence"] = evidence
            if evidence:
                with st.expander("Phase 3: Execution & Verification (Priority Order)", expanded=False): st.json(evidence)
                
                final_answer, p, c = await phase_4_synthesize_final_answer(env, model, query, plan, evidence)
                tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
                log_data["phases"]["4_final_answer"] = final_answer
                if final_answer:
                    with st.expander("Phase 4: Final Answer", expanded=True):
                        st.markdown(final_answer)
                        if st.button('Copy report text', key='priority_hyde_planner_copy'):
                            st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("Priority-HyDE-Planner finished.")
        display_tracking_info(tracking)

async def run_2step_hyde_planner(env: Environment, model: str, query: str, time_period: str, replan_depth: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, log_data: dict):
    st.header("2-Step HyDE-Planner")
    with st.spinner(f"Running 2-Step HyDE-Planner with re-plan after {replan_depth}..."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        # --- STEP 1: Initial Run ---
        st.subheader("Step 1: Initial Hypothesis and Verification")
        doc, p, c = await phase_1_generate_hypothetical_document(env, model, query)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        with st.expander("Step 1.1: Initial Hypothetical Document", expanded=False): st.markdown(doc)

        plan, p, c = await phase_2_reverse_engineer_plan(env, model, doc, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1.2_initial_plan"] = plan
        if not plan:
            st.error("Failed to generate an initial plan.")
            return

        with st.expander("Step 1.2: Initial Research Plan", expanded=False): st.json(plan)
        
        depth_map = {
            "Re-plan after High priority": "High priority only",
            "Re-plan after Medium priority": "Medium priority & higher",
            "Run all then re-plan": "All priorities"
        }
        search_depth = depth_map.get(replan_depth, "All priorities")

        evidence = phase_3_execute_plan_and_verify(env, plan, time_period, search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
        log_data["phases"]["1.3_initial_evidence"] = evidence
        with st.expander(f"Step 1.3: Initial Evidence ({search_depth})", expanded=False): st.json(evidence)

        # --- STEP 2: Re-planning based on initial evidence ---
        st.subheader("Step 2: Synthesis, Re-planning, and Final Verification")
        
        initial_synthesis, p, c = await phase_4_synthesize_final_answer(env, model, query, plan, evidence)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2.1_intermediate_synthesis"] = initial_synthesis
        with st.expander("Step 2.1: Intermediate Synthesis", expanded=False): st.markdown(initial_synthesis)

        st.info("Generating a new, improved hypothetical document based on initial findings...")
        refined_query = f"Original Query: {query}\n\nBased on initial research, the following is known:\n{initial_synthesis}\n\nPlease provide an updated and more accurate hypothetical answer to the original query based on this new context."
        doc2, p, c = await phase_1_generate_hypothetical_document(env, model, refined_query)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2.2_refined_document"] = doc2
        with st.expander("Step 2.2: Refined Hypothetical Document", expanded=False): st.markdown(doc2)

        plan2, p, c = await phase_2_reverse_engineer_plan(env, model, doc2, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2.3_refined_plan"] = plan2
        if not plan2:
            st.error("Failed to generate a refined plan. Synthesizing based on initial evidence.")
            final_answer = initial_synthesis
        else:
            with st.expander("Step 2.3: Refined Research Plan", expanded=False): st.json(plan2)
            
            evidence2 = phase_3_execute_plan_and_verify(env, plan2, time_period, "All priorities", use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
            log_data["phases"]["2.4_final_evidence"] = evidence2
            with st.expander("Step 2.4: Final Evidence Collection", expanded=False): st.json(evidence2)

            final_answer, p, c = await phase_4_synthesize_final_answer(env, model, query, plan2, evidence2)
            tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        
        log_data["phases"]["3_final_answer"] = final_answer
        with st.expander("Final Answer", expanded=True):
            st.markdown(final_answer)
            if st.button('Copy report text', key='2step_hyde_planner_copy'):
                st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("2-Step HyDE-Planner finished.")
        display_tracking_info(tracking)

async def run_hyde_planner_with_reflection(env: Environment, model: str, query: str, time_period: str, search_depth: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, log_data: dict):
    st.header("HyDE-Planner with 2nd Reflection")
    with st.spinner("Running HyDE-Planner with 2nd Reflection..."):
        start_time = time.time()
        log_data["phases"] = {}
        tracking = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        # --- STEP 1: Initial Run (Standard HyDE-Planner) ---
        st.subheader("Step 1: Initial HyDE-Planner Run")
        doc, p, c = await phase_1_generate_hypothetical_document(env, model, query)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1.1_initial_document"] = doc
        with st.expander("Step 1.1: Initial Hypothetical Document", expanded=False): st.markdown(doc)

        plan, p, c = await phase_2_reverse_engineer_plan(env, model, doc, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1.2_initial_plan"] = plan
        if not plan:
            st.error("Failed to generate an initial plan.")
            return
        with st.expander("Step 1.2: Initial Research Plan", expanded=False): st.json(plan)

        evidence = phase_3_execute_plan_and_verify(env, plan, time_period, search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
        log_data["phases"]["1.3_initial_evidence"] = evidence
        with st.expander("Step 1.3: Initial Evidence Collection", expanded=False): st.json(evidence)

        first_pass_answer, p, c = await phase_4_synthesize_final_answer(env, model, query, plan, evidence)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["1.4_first_pass_answer"] = first_pass_answer
        with st.expander("Step 1.4: First-Pass Answer", expanded=False): st.markdown(first_pass_answer)

        # --- STEP 2: Reflection and Refinement ---
        st.subheader("Step 2: Reflection and Refinement")
        gap_analysis, gap_evidence, p, c = await phase_6_reflection_and_research(env, model, query, first_pass_answer, plan, time_period, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_local_search=use_local_search)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["2.1_gap_analysis"] = gap_analysis
        log_data["phases"]["2.2_gap_evidence"] = gap_evidence
        with st.expander("Step 2.1: Information Gap Analysis", expanded=False): st.json(gap_analysis)
        with st.expander("Step 2.2: Evidence for Information Gap", expanded=False): st.json(gap_evidence)

        final_answer, p, c = await phase_7_synthesize_with_reflection(env, model, query, first_pass_answer, gap_evidence)
        tracking["prompt_tokens"] += p; tracking["completion_tokens"] += c
        log_data["phases"]["3_final_answer"] = final_answer
        with st.expander("Final Answer", expanded=True):
            st.markdown(final_answer)
            if st.button('Copy report text', key='hyde_planner_with_reflection_copy'):
                st.code(final_answer)

        tracking["duration"] = time.time() - start_time
        tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
        tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
        log_data["performance_summary"] = tracking
        st.success("HyDE-Planner with 2nd Reflection finished.")
        display_tracking_info(tracking)

# --- Core HyDE Phases ---

async def phase_1_generate_hypothetical_document(env: Environment, model: str, query: str) -> Tuple[str, int, int]:
    st.info("Phase 1: Generating Hypothetical Document...")
    prompt = get_hyde_generation_prompt(query)
    doc, p_tokens, c_tokens = await call_llm(env, model, prompt)
    if "Error:" in doc:
        st.error(f"Failed to generate hypothetical document: {doc}")
        return None, 0, 0
    return doc, p_tokens, c_tokens
async def phase_2_reverse_engineer_plan(env: Environment, model: str, doc: str, use_arxiv: bool, use_finance: bool, use_local_search: bool) -> Tuple[dict, int, int]:
    st.info("Phase 2: Reverse-Engineering Research Plan...")
    prompt = get_plan_reverse_engineering_prompt(doc, use_arxiv=use_arxiv, use_finance=use_local_search)
    response_text, p_tokens, c_tokens = await call_llm(env, model, prompt)
    
    if "Error:" in response_text:
        st.error(f"Failed to generate research plan: {response_text}")
        return None, p_tokens, c_tokens
        
    try:
        # Clean up the response to ensure it's valid JSON
        clean_response = response_text.strip()
        # Use json_repair to handle potential malformations
        plan = json_repair.loads(clean_response)
        return plan, p_tokens, c_tokens
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from the research plan: {e}")
        st.text_area("LLM Output that failed parsing:", response_text, height=200)
        return None, p_tokens, c_tokens

def phase_3_execute_plan_and_verify(env: Environment, plan: dict, time_period: str, search_depth: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool) -> dict:
    st.info("Phase 3: Executing Plan and Verifying Claims...")
    evidence = {}
    local_search_tool = None
    if use_local_search:
        try:
            local_search_tool = LocalSearch()
        except FileNotFoundError as e:
            st.warning(f"Could not initialize local search: {e}. Skipping local searches.")
        except Exception as e:
            st.error(f"An unexpected error occurred initializing local search: {e}")

    claims_to_verify = plan.get("claims_to_verify", [])
    exploratory_queries = plan.get("exploratory_queries", [])
    all_claims = claims_to_verify + exploratory_queries

    if not all_claims:
        st.warning("No claims or exploratory queries were found in the research plan.")
        return evidence

    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_claims.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
    
    priorities_to_run = []
    if search_depth == "High priority only":
        priorities_to_run = ["high"]
    elif search_depth == "Medium priority & higher":
        priorities_to_run = ["high", "medium"]
    else: # "All priorities"
        priorities_to_run = ["high", "medium", "low"]
        
    claims_to_run = [c for c in all_claims if c.get("priority") in priorities_to_run]

    for claim in claims_to_run:
        claim_id = claim.get("claim_id")
        query = claim.get("verification_query")
        claim_text = claim.get("claim_text", "N/A")
        tool = claim.get("tool", "google_search")
        
        if not query or not claim_id:
            continue
            
        st.write(f"  - Verifying Claim #{claim_id} ({claim.get('priority', 'N/A')}): `{claim_text}` -> `{query}` with tool `{tool}`")
        
        search_results = None
        if tool == "google_search":
            search_results = google_search(env, query, time_period=time_period, use_jina_api=use_jina_api, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
        elif tool == "arxiv_search" and use_arxiv:
            search_results = arxiv_search(query)
        elif tool == "yahoo_finance" and use_finance:
            search_results = finance_search(query)
        elif tool == "local_search" and use_local_search and local_search_tool:
            try:
                search_results = local_search_tool.search(query)
            except Exception as e:
                st.error(f"Local search failed for query '{query}': {e}")
                search_results = {"error": str(e)}
        else:
            st.warning(f"Tool '{tool}' is not enabled or not recognized. Skipping claim.")
            search_results = {"error": f"Tool '{tool}' is not enabled or not recognized."}

        evidence[claim_id] = search_results
        
    return evidence
async def phase_4_synthesize_final_answer(env: Environment, model: str, original_query: str, plan: dict, evidence: dict) -> Tuple[str, int, int]:
    st.info("Phase 4: Synthesizing Final Answer...")
    prompt = get_verification_and_synthesis_prompt(
        original_query=original_query,
        research_plan=plan,
        evidence=evidence
    )
    
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)
    
    if "Error:" in answer:
        st.error(f"Failed to synthesize the final answer: {answer}")
        return None, p_tokens, c_tokens
        
    return answer, p_tokens, c_tokens
async def phase_5_synthesize_answer_from_evidence(env: Environment, model: str, query: str, evidence: dict) -> Tuple[str, int, int]:
    st.info("Synthesizing answer from search results...")
    prompt = get_synthesis_from_search_results_prompt(query, evidence)
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)
    
    if "Error:" in answer:
        st.error(f"Failed to synthesize answer from search results: {answer}")
        return None, p_tokens, c_tokens
        
    return answer, p_tokens, c_tokens

async def phase_6_reflection_and_research(env: Environment, model: str, original_query: str, first_pass_answer: str, plan: dict, time_period: str, use_jina_api: bool, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_local_search: bool) -> Tuple[dict, dict, int, int]:
    st.info("Phase 6: Reflecting on First-Pass Answer and Researching Gaps...")
    
    previous_queries = [claim.get("verification_query", "") for claim in plan.get("claims_to_verify", [])]
    
    prompt = get_information_gap_prompt(original_query, first_pass_answer, previous_queries, use_local_search=use_local_search)
    response_text, p_tokens, c_tokens = await call_llm(env, model, prompt)

    if "Error:" in response_text:
        st.error(f"Failed to generate information gap analysis: {response_text}")
        return None, None, p_tokens, c_tokens

    try:
        gap_analysis = json_repair.loads(response_text)
        gap_queries = gap_analysis.get("search_queries_for_gap", [])
        if not gap_queries:
            st.warning("No search queries were generated for the information gap.")
            return gap_analysis, {}, p_tokens, c_tokens

        gap_evidence = {}
        for i, gap_query in enumerate(gap_queries):
            st.write(f"  - Researching Information Gap Query #{i+1}: `{gap_query}`")
            search_results = google_search(env, gap_query, time_period=time_period, use_jina_api=use_jina_api, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            gap_evidence[f"gap_query_{i+1}_{gap_query}"] = search_results
            
        return gap_analysis, gap_evidence, p_tokens, c_tokens

    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from the information gap analysis: {e}")
        st.text_area("LLM Output that failed parsing:", response_text, height=200)
        return None, None, p_tokens, c_tokens

async def phase_7_synthesize_with_reflection(env: Environment, model: str, original_query: str, first_pass_answer: str, gap_evidence: dict) -> Tuple[str, int, int]:
    st.info("Phase 7: Synthesizing Final Answer with Reflection...")
    prompt = get_synthesis_with_reflection_prompt(original_query, first_pass_answer, gap_evidence)
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)

    if "Error:" in answer:
        st.error(f"Failed to synthesize the final answer with reflection: {answer}")
        return None, p_tokens, c_tokens

    return answer, p_tokens, c_tokens

def main():
    """Defines the Streamlit UI and triggers the async runner."""
    st.set_page_config(page_title="HyDE Planner Demo", layout="wide")
    st.title("HyDE Planner Demo")

    env = initialize_environment()

    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Select LLM",
        options=["gemini-2.5-pro", "gemini-2.5-flash"],
        index=0
    )
    
    st.sidebar.subheader("Search Time Period")
    time_period_option = st.sidebar.selectbox(
        "Select a time period",
        options=["Any time", "Past year", "Past month", "Past week", "Custom range"],
        index=0
    )
    
    time_period = ""
    if time_period_option == "Custom range":
        start_date = st.sidebar.date_input("Start date")
        end_date = st.sidebar.date_input("End date")
        if start_date and end_date and start_date <= end_date:
            time_period = f"{start_date.strftime('%Y%m%d')}..{end_date.strftime('%Y%m%d')}"
    else:
        time_period = time_period_option

    log_to_file = st.sidebar.checkbox("Save run log to file", value=True)
    use_jina_api = st.sidebar.checkbox("Use Jina AI API for content extraction", value=True)
    search_pdfs = st.sidebar.checkbox("Include PDF results in search", value=False)
    use_arxiv = st.sidebar.checkbox("Search ArXiv", value=False)
    use_finance = st.sidebar.checkbox("Search yfinance_statements", value=False)
    use_local_search = st.sidebar.checkbox("Search Local Documents", value=False)

    if use_local_search:
        st.sidebar.subheader("Local Document Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF, TXT, or MD files", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md']
        )
        if uploaded_files:
            save_path = os.path.join(os.path.dirname(__file__), "local_docs")
            os.makedirs(save_path, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.sidebar.status("Processing documents...") as status:
                st.write("Saving files...")
                st.write("Running ingestion to update vector store...")
                try:
                    run_ingestion()
                    status.update(label="Document processing complete!", state="complete", expanded=False)
                    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded and indexed successfully!")
                except Exception as e:
                    status.update(label="Error processing documents", state="error")
                    st.sidebar.error(f"An error occurred during ingestion: {e}")

    
    pdf_processing_method = st.sidebar.selectbox(
        "PDF Processing Method",
        options=["Keyword Match (Fast)", "Embedding Search (Accurate)"],
        index=0
    )

    save_pdf_embeddings = False
    if pdf_processing_method == "Embedding Search (Accurate)":
        save_pdf_embeddings = st.sidebar.checkbox("Save PDF embeddings locally", value=True)


    st.sidebar.header("Query")
    query = st.sidebar.text_area(
        "Enter your research query:",
        "What were the key factors contributing to the decline of the woolly mammoth population?",
        height=150
    )

    st.sidebar.header("Methodology")
    
    available_methods = [
        "HyDE-Planner",
        "Priority-HyDE-Planner",
        "2-Step HyDE-Planner",
        "HyDE-Planner with 2nd Reflection",
        "Direct Search",
        "Query Decomposition Search",
        "Sequential-Reflection Search"
    ]
    
    selected_methods = st.sidebar.multiselect(
        "Choose research methodologies (run sequentially):",
        options=["Compare All"] + available_methods,
    )

    # --- Methodology-specific options ---
    search_depth = "All priorities"
    if "HyDE-Planner" in selected_methods or "Priority-HyDE-Planner" in selected_methods or "Compare All" in selected_methods or "HyDE-Planner with 2nd Reflection" in selected_methods:
        st.sidebar.subheader("HyDE Planner Settings")
        search_depth = st.sidebar.select_slider(
            "Search Depth",
            options=["High priority only", "Medium priority & higher", "All priorities"],
            value="All priorities"
        )

    replan_depth = ""
    if "2-Step HyDE-Planner" in selected_methods or "Compare All" in selected_methods:
        st.sidebar.subheader("2-Step HyDE Planner Settings")
        replan_depth = st.sidebar.selectbox(
            "Re-plan trigger",
            options=["Re-plan after High priority", "Re-plan after Medium priority", "Run all then re-plan"]
        )

    if st.sidebar.button("Run", use_container_width=True):
        if not query:
            st.sidebar.warning("Please enter a query.")
            return
        if not selected_methods:
            st.sidebar.warning("Please select at least one methodology.")
            return
        if time_period_option == "Custom range" and not time_period:
            st.sidebar.warning("Please select a valid custom date range.")
            return

        methods_to_run = available_methods if "Compare All" in selected_methods else selected_methods
        
        for methodology in methods_to_run:
            st.markdown(f"<hr style='margin: 2rem 0; border-top: 2px solid #bbb;'>", unsafe_allow_html=True)
            st.header(f"Running: {methodology}")

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "methodology": methodology,
                "model": model_name,
                "settings": {
                    "time_period": time_period, 
                    "search_depth": search_depth, 
                    "replan_depth": replan_depth, 
                    "use_jina_api": use_jina_api, 
                    "search_pdfs": search_pdfs,
                    "pdf_processing_method": pdf_processing_method,
                    "save_pdf_embeddings": save_pdf_embeddings,
                    "use_arxiv": use_arxiv,
                    "use_finance": use_finance,
                    "use_local_search": use_local_search
                }
            }

            runner = None
            args = [env, model_name, query, time_period]
            if methodology == "HyDE-Planner":
                runner = run_hyde_planner
                args.extend([search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, log_data])
            elif methodology == "Priority-HyDE-Planner":
                runner = run_priority_hyde_planner
                args.extend([search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, log_data])
            elif methodology == "2-Step HyDE-Planner":
                runner = run_2step_hyde_planner
                args.extend([replan_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, log_data])
            elif methodology == "HyDE-Planner with 2nd Reflection":
                runner = run_hyde_planner_with_reflection
                args.extend([search_depth, use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, log_data])
            elif methodology == "Direct Search":
                runner = run_direct_search
                args.extend([use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_local_search, log_data])
            elif methodology == "Query Decomposition Search":
                runner = run_query_decomposition_search
                args.extend([use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_local_search, log_data])
            elif methodology == "Sequential-Reflection Search":
                runner = run_sequential_reflection_search
                args.extend([use_jina_api, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, log_data])

            if runner:
                asyncio.run(runner(*args))

            if log_to_file:
                log_dir = os.path.join(os.path.dirname(__file__), "logs")
                os.makedirs(log_dir, exist_ok=True)
                safe_method_name = methodology.lower().replace(" ", "_")
                filename = f"run_{safe_method_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
                with open(os.path.join(log_dir, filename), "w") as f:
                    json.dump(log_data, f, indent=2)
                st.sidebar.success(f"Log saved to {filename}")


if __name__ == "__main__":
    main()
