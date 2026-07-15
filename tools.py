from langchain_core.tools import tool
from datetime import datetime
import warnings

# Suppress the duckduckgo_search rename warning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

from duckduckgo_search import DDGS


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: 25*4+100
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)


@tool
def current_time(dummy: str = "") -> str:
    """
    Returns current date and time.
    """
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


@tool
def read_file(filename: str) -> str:
    """
    Read a local text file.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for general information.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        answer = ""

        for r in results:
            answer += f"Title: {r['title']}\n"
            answer += f"Body: {r['body']}\n"
            answer += f"Link: {r['href']}\n\n"

        return answer

    except Exception as e:
        return str(e)