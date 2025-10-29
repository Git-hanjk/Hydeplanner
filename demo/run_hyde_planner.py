# run_hyde_planner.py
import streamlit as st
import json
import asyncio
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Tuple, Dict, Any, List
import requests

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
import google.generativeai as genai
from prompts import (
    get_hyde_generation_prompt, get_plan_reverse_engineering_prompt,
    get_verification_and_synthesis_prompt, get_query_decomposition_prompt,
    get_reflection_prompt, get_synthesis_from_conversation_prompt,
    get_synthesis_from_search_results_prompt, get_information_gap_prompt,
    get_synthesis_with_reflection_prompt
)
from hyde_search_module import google_search
from arxiv_search_module import arxiv_search
from finance_search_module import finance_search
from local_search_module import LocalSearch
from settings import Environment
from ingest import run_ingestion
import json_repair
from PIL import Image, ImageDraw, ImageFont
import markdown
import io

# --- Font Management ---
def download_font(language: str = "English") -> str:
    """
    Checks for a font file in the static directory based on the selected language,
    downloads it if not present (for non-local fonts), and returns the path.
    """
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    font_map = {
        "Korean": {
            "filename": "NotoSansKR-VariableFont_wght.ttf",
            "local": True
        },
        "Japanese": {
            "filename": "NotoSansJP-VariableFont_wght.ttf",
            "local": True
        },
        "Chinese": {
            "filename": "NotoSansSC-VariableFont_wght.ttf",
            "local": True
        },
        "English": {
            "filename": "DejaVuSans.ttf",
            "url": "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf",
            "local": False
        },
        "Spanish": {
            "filename": "DejaVuSans.ttf",
            "url": "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf",
            "local": False
        },
        "French": {
            "filename": "DejaVuSans.ttf",
            "url": "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf",
            "local": False
        }
    }

    font_info = font_map.get(language, font_map["English"])
    font_filename = font_info["filename"]
    font_path = os.path.join(static_dir, font_filename)

    if font_info.get("local", False):
        if not os.path.exists(font_path):
            st.error(f"Local font file not found: {font_path}. Please make sure it exists.")
            return None
        return font_path

    # Download logic for non-local fonts
    if not os.path.exists(font_path):
        st.info(f"Downloading required font: {font_filename} for {language}...")
        try:
            font_url = font_info["url"]
            response = requests.get(font_url, stream=True)
            response.raise_for_status()
            with open(font_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Font for {language} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading font: {e}")
            return None
    return font_path

# --- UI Helper Functions ---

def save_report_as_image(report_text, key, font_path):
    try:
        width, height, margin, font_size, line_spacing = 800, 1000, 20, 14, 5
        text = report_text.replace('*', '')
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            if not font_path:
                 st.warning("Using default font for image. Non-ASCII characters may not render correctly.")
        except IOError:
            st.warning(f"Failed to load font at {font_path}. Non-ASCII characters may not render correctly in the image.")
            font = ImageFont.load_default()
        
        y_text, lines = margin, []
        words = text.split(' ')
        line = ''
        for word in words:
            if '\n' in word:
                parts = word.split('\n')
                for i, part in enumerate(parts):
                    line_to_add = f"{line} {part}" if line else part
                    if draw.textbbox((0, 0), line_to_add, font=font)[2] < width - margin * 2:
                        line = line_to_add
                    else:
                        lines.append(line); line = part
                    if i < len(parts) - 1:
                        lines.append(line); line = ''
                continue
            if draw.textbbox((0, 0), f"{line} {word}" if line else word, font=font)[2] < width - margin * 2:
                line = f"{line} {word}" if line else word
            else:
                lines.append(line); line = word
        lines.append(line)

        for l in lines:
            draw.text((margin, y_text), l.strip(), font=font, fill='black')
            y_text += font_size + line_spacing
            if y_text > height - margin: break
        
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        st.download_button(label="Download Report as Image", data=buf.getvalue(), file_name=f"report_{key}.png", mime="image/png")
    except Exception as e:
        st.error(f"Failed to create image from report: {e}")

def save_report_as_document(report_text, key):
    try:
        st.download_button(label="Download Report as Document", data=report_text.encode('utf-8'), file_name=f"report_{key}.txt", mime="text/plain")
    except Exception as e:
        st.error(f"Failed to create download link for document: {e}")

def save_report_as_pdf(report_text, key, font_path):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        try:
            if font_path:
                pdf.add_font('NotoSans', '', font_path, uni=True)
                pdf.set_font('NotoSans', '', 12)
            else:
                raise RuntimeError("Font file not available.")
        except RuntimeError:
            st.warning("Selected font not found or failed to load. PDF content may not render non-ASCII characters correctly.")
            pdf.set_font('Arial', '', 12)
        pdf.write_html(markdown.markdown(report_text))
        pdf_output = pdf.output(dest='S').encode('utf-8')
        st.download_button(label="Download Report as PDF", data=pdf_output, file_name=f"report_{key}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to create PDF from report: {e}")

# --- Core Logic Helper Functions ---

@st.cache_resource
def initialize_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
    return Environment(
        main_api_key=os.getenv("OPENAI_API_KEY"), api_base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"), gemini_model_name=os.getenv("GEMINI_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"), google_cse_id=os.getenv("GOOGLE_CSE_ID")
    )

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = {"gemini-2.5-pro": {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000}, "gemini-2.5-flash": {"prompt": 0.30 / 1_000_000, "completion": 2.50 / 1_000_000}}
    model_pricing = pricing.get(model_name)
    return ((prompt_tokens * model_pricing["prompt"]) + (completion_tokens * model_pricing["completion"])) if model_pricing else 0.0

async def call_llm(env: Environment, model_name: str, prompt: str) -> Tuple[str, int, int]:
    text, p_tokens, c_tokens = "", 0, 0
    try:
        if "gemini" in model_name:
            genai.configure(api_key=env.gemini_api_key)
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async(prompt)
            text, usage = response.text, response.usage_metadata
            p_tokens, c_tokens = usage.prompt_token_count, usage.candidates_token_count
        else:
            text = f"Error: Unknown model provider for '{model_name}'."
    except Exception as e:
        text = f"Error: LLM call failed for {model_name}: {e}"
    return text, p_tokens, c_tokens

# --- Core Phase Functions (Data-in, Data-out) ---

async def phase_1_generate_hypothetical_document(env: Environment, model: str, query: str, language: str) -> Dict[str, Any]:
    prompt = get_hyde_generation_prompt(query, language=language)
    doc, p_tokens, c_tokens = await call_llm(env, model, prompt)
    result = {"doc": doc, "p_tokens": p_tokens, "c_tokens": c_tokens}
    if "Error:" in doc:
        result["error"] = f"Failed to generate hypothetical document: {doc}"
    return result

async def phase_2_reverse_engineer_plan(env: Environment, model: str, doc: str, use_arxiv: bool, use_finance: bool, use_local_search: bool, language: str) -> Dict[str, Any]:
    prompt = get_plan_reverse_engineering_prompt(doc, use_arxiv=use_arxiv, use_finance=use_finance, use_local_search=use_local_search, language=language)
    response_text, p_tokens, c_tokens = await call_llm(env, model, prompt)
    result = {"plan": None, "p_tokens": p_tokens, "c_tokens": c_tokens, "raw_text": response_text}
    if "Error:" in response_text:
        result["error"] = f"Failed to generate research plan: {response_text}"
        return result
    try:
        result["plan"] = json_repair.loads(response_text.strip())
    except json.JSONDecodeError as e:
        result["error"] = f"Failed to decode JSON from the research plan: {e}"
    return result

def phase_3_execute_plan_and_verify(env: Environment, plan: dict, time_period: str, search_depth: str, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool) -> Dict[str, Any]:
    evidence, logs = {}, []
    try:
        local_search_tool = LocalSearch() if use_local_search else None
    except Exception as e:
        logs.append(f"Warning: Could not initialize local search: {e}. Skipping local searches.")
        local_search_tool = None

    all_claims = plan.get("claims_to_verify", []) + plan.get("exploratory_queries", [])
    if not all_claims:
        logs.append("Warning: No claims or exploratory queries found in the research plan.")
        return {"evidence": evidence, "logs": logs}

    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_claims.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
    
    depth_map = {"High priority only": ["high"], "Medium priority & higher": ["high", "medium"], "All priorities": ["high", "medium", "low"]}
    priorities_to_run = depth_map.get(search_depth, ["high", "medium", "low"])
    claims_to_run = [c for c in all_claims if c.get("priority") in priorities_to_run]

    for claim in claims_to_run:
        claim_id, query, tool = claim.get("claim_id"), claim.get("verification_query"), claim.get("tool", "google_search")
        if not query or not claim_id: continue
        
        logs.append(f"Verifying Claim #{claim_id} ({claim.get('priority', 'N/A')}): `{claim.get('claim_text', 'N/A')}` -> `{query}` with tool `{tool}`")
        
        search_results = None
        try:
            if tool == "google_search":
                search_results = google_search(env, query, time_period=time_period, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "arxiv_search" and use_arxiv: search_results = arxiv_search(query)
            elif tool == "yahoo_finance" and use_finance: search_results = finance_search(query)
            elif tool == "local_search" and local_search_tool: search_results = local_search_tool.search(query)
            else:
                logs.append(f"Warning: Tool '{tool}' is not enabled or recognized. Skipping claim.")
                search_results = {"error": f"Tool '{tool}' is not enabled or recognized."}
        except Exception as e:
            logs.append(f"Error: Search failed for query '{query}' with tool '{tool}': {e}")
            search_results = {"error": str(e)}
        evidence[claim_id] = search_results
    return {"evidence": evidence, "logs": logs}

async def phase_4_synthesize_final_answer(env: Environment, model: str, original_query: str, plan: dict, evidence: dict, language: str) -> Dict[str, Any]:
    prompt = get_verification_and_synthesis_prompt(original_query=original_query, research_plan=plan, evidence=evidence, language=language)
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)
    result = {"answer": answer, "p_tokens": p_tokens, "c_tokens": c_tokens}
    if "Error:" in answer:
        result["error"] = f"Failed to synthesize the final answer: {answer}"
    return result

# ... (Other phase functions would be refactored similarly) ...

# --- Methodology Runners (Refactored to return log dict) ---

async def run_hyde_planner(env: Environment, model: str, query: str, time_period: str, search_depth: str, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, language: str) -> Dict[str, Any]:
    log_data = {"methodology": "HyDE-Planner", "phases": {}, "status_updates": [], "performance_summary": {}}
    start_time = time.time()
    tracking = {"prompt_tokens": 0, "completion_tokens": 0}

    log_data["status_updates"].append("Phase 1: Generating Hypothetical Document...")
    p1_res = await phase_1_generate_hypothetical_document(env, model, query, language)
    tracking["prompt_tokens"] += p1_res["p_tokens"]; tracking["completion_tokens"] += p1_res["c_tokens"]
    if p1_res.get("error"):
        log_data["status_updates"].append(f"Error in Phase 1: {p1_res['error']}")
        return log_data
    log_data["phases"]["1_hypothetical_document"] = p1_res["doc"]

    log_data["status_updates"].append("Phase 2: Reverse-Engineering Research Plan...")
    p2_res = await phase_2_reverse_engineer_plan(env, model, p1_res["doc"], use_arxiv, use_finance, use_local_search, language)
    tracking["prompt_tokens"] += p2_res["p_tokens"]; tracking["completion_tokens"] += p2_res["c_tokens"]
    if p2_res.get("error"):
        log_data["status_updates"].append(f"Error in Phase 2: {p2_res['error']}")
        log_data["phases"]["2_raw_plan_output"] = p2_res.get("raw_text")
        return log_data
    log_data["phases"]["2_research_plan"] = p2_res["plan"]

    log_data["status_updates"].append("Phase 3: Executing Plan and Verifying Claims...")
    p3_res = phase_3_execute_plan_and_verify(env, p2_res["plan"], time_period, search_depth, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search)
    log_data["status_updates"].extend(p3_res["logs"])
    log_data["phases"]["3_collected_evidence"] = p3_res["evidence"]

    log_data["status_updates"].append("Phase 4: Synthesizing Final Answer...")
    p4_res = await phase_4_synthesize_final_answer(env, model, query, p2_res["plan"], p3_res["evidence"], language)
    tracking["prompt_tokens"] += p4_res["p_tokens"]; tracking["completion_tokens"] += p4_res["c_tokens"]
    if p4_res.get("error"):
        log_data["status_updates"].append(f"Error in Phase 4: {p4_res['error']}")
        return log_data
    log_data["phases"]["final_answer"] = p4_res["answer"]

    tracking["duration"] = time.time() - start_time
    tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
    tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
    log_data["performance_summary"] = tracking
    return log_data

async def run_hyde_planner_with_reflection(env: Environment, model: str, query: str, time_period: str, search_depth: str, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, language: str) -> Dict[str, Any]:
    log_data = {"methodology": "HyDE-Planner with 2nd Reflection", "phases": {}, "status_updates": [], "performance_summary": {}}
    start_time = time.time()
    tracking = {"prompt_tokens": 0, "completion_tokens": 0}

    # --- Run original HyDE Planner ---
    initial_hyde_log = await run_hyde_planner(env, model, query, time_period, search_depth, search_pdfs, pdf_processing_method, save_pdf_embeddings, use_arxiv, use_finance, use_local_search, language)
    
    # --- Merge initial log data ---
    log_data["status_updates"].extend(initial_hyde_log.get("status_updates", []))
    log_data["phases"] = initial_hyde_log.get("phases", {})
    if "performance_summary" in initial_hyde_log:
        tracking["prompt_tokens"] += initial_hyde_log["performance_summary"].get("prompt_tokens", 0)
        tracking["completion_tokens"] += initial_hyde_log["performance_summary"].get("completion_tokens", 0)

    first_pass_answer = log_data.get("phases", {}).get("final_answer")
    if not first_pass_answer or "Error:" in first_pass_answer:
        log_data["status_updates"].append("Skipping reflection due to error or no initial answer.")
        return log_data

    # --- Phase 5: Identify Information Gaps ---
    log_data["status_updates"].append("Phase 5: Reflecting on the first-pass answer to find information gaps...")
    previous_queries = [c.get("verification_query", "") for c in log_data.get("phases", {}).get("2_research_plan", {}).get("claims_to_verify", [])]
    gap_prompt = get_information_gap_prompt(query, first_pass_answer, previous_queries, use_arxiv, use_finance, use_local_search, language)
    gap_response, p_tokens, c_tokens = await call_llm(env, model, gap_prompt)
    tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
    
    try:
        gap_plan = json_repair.loads(gap_response)
        log_data["phases"]["5_gap_analysis"] = gap_plan
    except json.JSONDecodeError as e:
        log_data["status_updates"].append(f"Error in Phase 5: Failed to decode JSON from gap analysis: {e}")
        log_data["phases"]["5_raw_gap_output"] = gap_response
        return log_data

    # --- Phase 6: Execute Gap-Filling Search ---
    log_data["status_updates"].append("Phase 6: Executing gap-filling search...")
    gap_evidence, gap_logs = {}, []
    for claim in gap_plan.get("search_queries_for_gap", []):
        claim_id, gap_query, tool = f"gap_{len(gap_evidence)+1}", claim.get("query"), claim.get("tool", "google_search")
        if not gap_query: continue
        
        gap_logs.append(f"Executing Gap Query: `{gap_query}` with tool `{tool}`")
        search_results = None
        try:
            if tool == "google_search": search_results = google_search(env, gap_query, time_period=time_period, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "arxiv_search" and use_arxiv: search_results = arxiv_search(gap_query)
            elif tool == "yahoo_finance" and use_finance: search_results = finance_search(gap_query)
            elif tool == "local_search" and use_local_search: search_results = LocalSearch().search(gap_query)
            else: search_results = {"error": f"Tool '{tool}' not enabled or recognized."}
        except Exception as e: search_results = {"error": str(e)}
        gap_evidence[claim_id] = search_results
    log_data["status_updates"].extend(gap_logs)
    log_data["phases"]["6_gap_evidence"] = gap_evidence

    # --- Phase 7: Final Synthesis with Reflection ---
    log_data["status_updates"].append("Phase 7: Synthesizing final answer with new evidence...")
    final_synth_prompt = get_synthesis_with_reflection_prompt(query, first_pass_answer, gap_evidence, language)
    final_answer, p_tokens, c_tokens = await call_llm(env, model, final_synth_prompt)
    tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
    log_data["phases"]["final_answer"] = final_answer # Overwrite the old final answer

    tracking["duration"] = time.time() - start_time
    tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
    tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
    log_data["performance_summary"] = tracking
    return log_data

async def run_query_decomposition_search(env: Environment, model: str, query: str, time_period: str, search_depth: str, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, language: str) -> Dict[str, Any]:
    log_data = {"methodology": "Query Decomposition Search", "phases": {}, "status_updates": [], "performance_summary": {}}
    start_time = time.time()
    tracking = {"prompt_tokens": 0, "completion_tokens": 0}

    # Phase 1: Decompose Query
    log_data["status_updates"].append("Phase 1: Decomposing query...")
    prompt = get_query_decomposition_prompt(query, use_arxiv, use_finance, use_local_search, language)
    response_text, p_tokens, c_tokens = await call_llm(env, model, prompt)
    tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
    
    try:
        decomposed_plan = json_repair.loads(response_text)
        log_data["phases"]["1_decomposed_plan"] = decomposed_plan
    except json.JSONDecodeError as e:
        log_data["status_updates"].append(f"Error in Phase 1: Failed to decode JSON from decomposition: {e}")
        log_data["phases"]["1_raw_decomposition_output"] = response_text
        return log_data

    # Phase 2: Execute All Sub-Queries
    log_data["status_updates"].append("Phase 2: Executing sub-queries...")
    evidence, logs = {}, []
    local_search_tool = LocalSearch() if use_local_search else None
    for i, sub_query_data in enumerate(decomposed_plan.get("sub_queries", [])):
        sub_query, tool = sub_query_data.get("query"), sub_query_data.get("tool", "google_search")
        if not sub_query: continue
        
        logs.append(f"Executing Sub-Query #{i+1}: `{sub_query}` with tool `{tool}`")
        search_results = None
        try:
            if tool == "google_search": search_results = google_search(env, sub_query, time_period=time_period, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "arxiv_search" and use_arxiv: search_results = arxiv_search(sub_query)
            elif tool == "yahoo_finance" and use_finance: search_results = finance_search(sub_query)
            elif tool == "local_search" and local_search_tool: search_results = local_search_tool.search(sub_query)
            else: search_results = {"error": f"Tool '{tool}' not enabled or recognized."}
        except Exception as e: search_results = {"error": str(e)}
        evidence[f"sub_query_{i+1}"] = {"query": sub_query, "results": search_results}
    log_data["status_updates"].extend(logs)
    log_data["phases"]["2_collected_evidence"] = evidence

    # Phase 3: Synthesize Final Answer
    log_data["status_updates"].append("Phase 3: Synthesizing final answer from all results...")
    prompt = get_synthesis_from_search_results_prompt(query, evidence, language)
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)
    tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
    log_data["phases"]["final_answer"] = answer

    tracking["duration"] = time.time() - start_time
    tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
    tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
    log_data["performance_summary"] = tracking
    return log_data

async def run_sequential_reflection_search(env: Environment, model: str, query: str, time_period: str, search_depth: str, search_pdfs: bool, pdf_processing_method: str, save_pdf_embeddings: bool, use_arxiv: bool, use_finance: bool, use_local_search: bool, language: str, max_turns: int = 5) -> Dict[str, Any]:
    log_data = {"methodology": "Sequential-Reflection Search", "phases": {}, "status_updates": [], "performance_summary": {}}
    start_time = time.time()
    tracking = {"prompt_tokens": 0, "completion_tokens": 0}
    conversation_history = ""
    local_search_tool = LocalSearch() if use_local_search else None

    for turn in range(1, max_turns + 1):
        log_data["status_updates"].append(f"Turn {turn}: Reflecting on progress...")
        
        # Phase 1: Reflection and Planning
        prompt = get_reflection_prompt(query, conversation_history, use_arxiv, use_finance, use_local_search, language)
        response_text, p_tokens, c_tokens = await call_llm(env, model, prompt)
        tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
        
        try:
            reflection_plan = json_repair.loads(response_text)
            log_data["phases"][f"turn_{turn}_reflection"] = reflection_plan
            conversation_history += f"\nTurn {turn} Reflection: {reflection_plan.get('reflection', '')}"
        except json.JSONDecodeError as e:
            log_data["status_updates"].append(f"Error in Turn {turn}: Failed to decode reflection JSON: {e}")
            log_data["phases"][f"turn_{turn}_raw_reflection_output"] = response_text
            break

        if reflection_plan.get("is_final_answer"):
            log_data["status_updates"].append("Agent decided to synthesize the final answer.")
            break

        # Phase 2: Execution
        next_query_data = reflection_plan.get("next_query")
        if not next_query_data or not next_query_data.get("query"):
            log_data["status_updates"].append("Agent did not provide a next query. Ending run.")
            break
        
        search_query, tool = next_query_data["query"], next_query_data.get("tool", "google_search")
        log_data["status_updates"].append(f"Turn {turn}: Executing query `{search_query}` with tool `{tool}`")
        
        search_results = None
        try:
            if tool == "google_search": search_results = google_search(env, search_query, time_period=time_period, search_pdfs=search_pdfs, pdf_processing_method=pdf_processing_method, save_pdf_embeddings=save_pdf_embeddings)
            elif tool == "arxiv_search" and use_arxiv: search_results = arxiv_search(search_query)
            elif tool == "yahoo_finance" and use_finance: search_results = finance_search(search_query)
            elif tool == "local_search" and local_search_tool: search_results = local_search_tool.search(search_query)
            else: search_results = {"error": f"Tool '{tool}' not enabled or recognized."}
        except Exception as e: search_results = {"error": str(e)}
        
        log_data["phases"][f"turn_{turn}_evidence"] = search_results
        conversation_history += f"\nExecuted Query: '{search_query}'\nResults:\n{json.dumps(search_results, indent=2)}"

    # Final Synthesis
    log_data["status_updates"].append("Final Phase: Synthesizing answer from conversation history...")
    prompt = get_synthesis_from_conversation_prompt(query, conversation_history, language)
    answer, p_tokens, c_tokens = await call_llm(env, model, prompt)
    tracking["prompt_tokens"] += p_tokens; tracking["completion_tokens"] += c_tokens
    log_data["phases"]["final_answer"] = answer

    tracking["duration"] = time.time() - start_time
    tracking["total_tokens"] = tracking["prompt_tokens"] + tracking["completion_tokens"]
    tracking["total_cost"] = calculate_cost(model, tracking["prompt_tokens"], tracking["completion_tokens"])
    log_data["performance_summary"] = tracking
    return log_data

# ... (Other runner functions would be refactored similarly) ...

# --- Display Functions ---

def display_tracking_info(info: dict):
    st.info(f"- **Execution Time:** {info.get('duration', 0):.2f} seconds\n"
            f"- **Total Tokens:** {info.get('total_tokens', 0)}\n"
            f"- **Estimated Cost:** ${info.get('total_cost', 0):.6f}")

def display_run_results(log_data: Dict[str, Any], font_path: str):
    methodology = log_data.get("methodology", "Unknown")
    st.markdown(f"<hr style='margin: 2rem 0; border-top: 2px solid #bbb;'>", unsafe_allow_html=True)
    st.header(f"Results for: {methodology}")

    for status in log_data.get("status_updates", []):
        if status.startswith("Warning:"): st.warning(status)
        elif status.startswith("Error:"): st.error(status)
        else: st.info(status)

    phases = log_data.get("phases", {})
    key_suffix = methodology.lower().replace(" ", "_")

    # Display logic for different methodologies can be added here
    if "1_hypothetical_document" in phases:
        with st.expander("Phase 1: Hypothetical Document", expanded=False):
            st.markdown(phases["1_hypothetical_document"] or "No document generated.")
    if "2_research_plan" in phases:
        with st.expander("Phase 2: Research Plan", expanded=False):
            st.json(phases["2_research_plan"] or "No plan generated.")
    if "2_raw_plan_output" in phases:
        with st.expander("Phase 2: Raw Plan Output (JSON Error)", expanded=True):
            st.text(phases["2_raw_plan_output"] or "No raw output.")
    if "3_collected_evidence" in phases:
        with st.expander("Phase 3: Execution & Verification", expanded=False):
            st.json(phases["3_collected_evidence"] or "No evidence collected.")
    
    final_answer = phases.get("final_answer")
    if final_answer:
        with st.expander("Final Answer", expanded=True):
            st.markdown(final_answer)
            if st.button('Copy report text', key=f'{key_suffix}_copy'):
                st.code(final_answer)
            save_report_as_image(final_answer, f"{key_suffix}_report", font_path)
            save_report_as_document(final_answer, f"{key_suffix}_report")
            save_report_as_pdf(final_answer, f"{key_suffix}_report", font_path)

    if log_data.get("performance_summary"):
        st.success(f"{methodology} finished.")
        display_tracking_info(log_data["performance_summary"])

# --- Main App ---

def main():
    st.set_page_config(page_title="HyDE Planner Demo", layout="wide")
    st.title("HyDE Planner Demo")

    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []
    if "font_path" not in st.session_state:
        st.session_state.font_path = None

    env = initialize_environment()

    # --- Sidebar Setup ---
    st.sidebar.header("Settings")
    language_options = {"English": "English", "Korean": "Korean", "Chinese": "Chinese", "Japanese": "Japanese", "Spanish": "Spanish", "French": "French"}
    selected_language = language_options[st.sidebar.selectbox("Select Language", options=list(language_options.keys()))]
    
    # Download font based on language and store it in session state
    st.session_state.font_path = download_font(selected_language)

    model_name = st.sidebar.selectbox("Select LLM", options=["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
    
    time_period_option = st.sidebar.selectbox("Search Time Period", options=["Any time", "Past year", "Past month", "Past week", "Custom range"], index=0)
    time_period = ""
    if time_period_option == "Custom range":
        start_date, end_date = st.sidebar.date_input("Start date"), st.sidebar.date_input("End date")
        if start_date and end_date and start_date <= end_date:
            time_period = f"{start_date.strftime('%Y%m%d')}..{end_date.strftime('%Y%m%d')}"
    else:
        time_period = time_period_option

    log_to_file = st.sidebar.checkbox("Save run log to file", value=True)
    search_pdfs = st.sidebar.checkbox("Include PDF results in search", value=False)
    use_arxiv = st.sidebar.checkbox("Search ArXiv", value=False)
    use_finance = st.sidebar.checkbox("Search yfinance_statements", value=False)
    use_local_search = st.sidebar.checkbox("Search Local Documents", value=False)
    
    # ... (rest of the sidebar is similar) ...
    query = st.sidebar.text_area("Enter your research query:", "What were the key factors contributing to the decline of the woolly mammoth population?", height=150)
    
    available_methods = ["HyDE-Planner", "HyDE-Planner with 2nd Reflection", "Query Decomposition Search", "Sequential-Reflection Search"]
    selected_methods = st.sidebar.multiselect("Choose research methodologies:", options=available_methods, default=[available_methods[0]])
    search_depth = st.sidebar.select_slider("Search Depth", options=["High priority only", "Medium priority & higher", "All priorities"], value="All priorities")

    # --- Run Button and Logic ---
    if st.sidebar.button("Run", use_container_width=True):
        if not query or not selected_methods or (time_period_option == "Custom range" and not time_period):
            st.sidebar.warning("Please provide a query, select a method, and ensure date range is valid.")
        else:
            st.session_state.run_logs = []
            
            for methodology in selected_methods:
                run_log = None
                runner_args = (env, model_name, query, time_period, search_depth, search_pdfs, "Keyword Match (Fast)", False, use_arxiv, use_finance, use_local_search, selected_language)
                
                with st.spinner(f"Running {methodology}..."):
                    if methodology == "HyDE-Planner":
                        run_log = asyncio.run(run_hyde_planner(*runner_args))
                    elif methodology == "HyDE-Planner with 2nd Reflection":
                        run_log = asyncio.run(run_hyde_planner_with_reflection(*runner_args))
                    elif methodology == "Query Decomposition Search":
                        run_log = asyncio.run(run_query_decomposition_search(*runner_args[:-5], selected_language)) # Adjust args
                    elif methodology == "Sequential-Reflection Search":
                        run_log = asyncio.run(run_sequential_reflection_search(*runner_args))

                if run_log:
                    st.session_state.run_logs.append(run_log)
                    if log_to_file:
                        log_dir = os.path.join(os.path.dirname(__file__), "logs")
                        os.makedirs(log_dir, exist_ok=True)
                        filename = f"run_{methodology.lower().replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json"
                        with open(os.path.join(log_dir, filename), "w") as f:
                            json.dump(run_log, f, indent=2)
                        st.sidebar.success(f"Log saved to {filename}")

    # --- Display Area (runs on every script rerun) ---
    if st.session_state.run_logs:
        for log in st.session_state.run_logs:
            display_run_results(log, st.session_state.font_path)

if __name__ == "__main__":
    main()
