import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time
from collections import defaultdict
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List
import aiohttp
from bs4 import BeautifulSoup


def extract_answer_fn(output, mode='qa', extract_answer=False):
    extracted_text = ''
    pattern_info = "**Final Information"
    if "</think>\n" in output:
        extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
        if mode == 'infogen':
            extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
    elif pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
        if mode == 'infogen':
            extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
    else:
        # extracted_text = "No helpful information found."
        extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
    if mode == 'research':
        extracted_text = extracted_text[:6000]
    else:
        extracted_text = extracted_text[:2500]
    return extracted_text


def extract_snippet_with_context(full_content, snippet, context_chars=500):
    """
    Extracts a snippet from the full content with surrounding context.

    Args:
        full_content (str): The entire text content.
        snippet (str): The snippet to find within the full content.
        context_chars (int): The number of characters of context to include 
                             before and after the snippet.

    Returns:
        tuple: A tuple containing:
               - bool: True if the snippet was found, False otherwise.
               - str: The snippet with context, or the original full_content 
                      if the snippet was not found.
    """
    # Clean up the snippet by removing punctuation and making it lowercase
    # for a more robust search.
    cleaned_snippet = snippet.strip(string.punctuation).lower()
    
    # Find the starting position of the cleaned snippet in the lowercase full content.
    start_index = full_content.lower().find(cleaned_snippet)
    
    if start_index != -1:
        # Calculate the start and end points for the context extraction.
        context_start = max(0, start_index - context_chars)
        context_end = min(len(full_content), start_index + len(cleaned_snippet) + context_chars)
        
        # Extract the snippet with context.
        context_snippet = full_content[context_start:context_end]
        return True, context_snippet
    else:
        # If the snippet is not found, return False and the original content.
        return False, full_content


async def bing_web_search_async(query, subscription_key, endpoint):
    """
    Performs an asynchronous Bing web search.
    """
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint, headers=headers, params=params, timeout=10) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"An error occurred during Bing search: {e}")
            return {}

async def fetch_page_content_async(urls, keep_links=False):
    """
    Asynchronously fetches content from a list of URLs.
    """
    async def fetch(session, url):
        try:
            async with session.get(url, timeout=10, ssl=False) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                if not keep_links:
                    for a in soup.find_all('a'):
                        a.decompose()
                
                return url, soup.get_text(separator=' ', strip=True)
        except Exception as e:
            return url, f"Error fetching content: {e}"

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return {url: content for url, content in results}

def extract_relevant_info(results):
    """
    Extracts relevant information (title, URL, snippet) from Bing search results.
    """
    relevant_info = []
    if 'webPages' in results and 'value' in results['webPages']:
        for item in results['webPages']['value']:
            relevant_info.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", "")
            })
    return relevant_info

