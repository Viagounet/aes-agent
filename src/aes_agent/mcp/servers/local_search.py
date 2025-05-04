# basic import
from fastmcp import FastMCP, Context
from typing import Any
from loguru import logger
import math

# instantiate an MCP server client
mcp = FastMCP("LocalSearch Server")

# DEFINE TOOLS

@mcp.tool()
async def number_of_words(pdf_path: str, ctx: Context) -> str:
    """Returns to the user the number of words contained inside a PDF document"""
    await ctx.info(f"Processing {pdf_path}...")
    data = await ctx.read_resource(f"pdf://{pdf_path}")
    return f"{pdf_path} contains {data[0].content.count(' ')} words"

@mcp.tool()
async def read_full_pdf(pdf_path: str, ctx: Context) -> str:
    """Returns to the user the full content of a PDF file."""
    await ctx.info(f"Processing {pdf_path}...")
    data = await ctx.read_resource(f"pdf://{pdf_path}")
    return data[0].content

@mcp.tool()
async def read_specific_page(pdf_path: str, pdf_page:int, ctx: Context) -> str:
    """Returns to the user the content of a specific page in a PDF file."""
    await ctx.info(f"Processing {pdf_path}...")
    data = await ctx.read_resource(f"page://{pdf_path}/{pdf_page}")
    return data[0].content

@mcp.tool()
def final_answer(answer: Any) -> str:
    """provides the user with your final answer, ends the conversation."""
    return str(answer)



# Dynamic resource template
@mcp.resource("pdf://{file_path*}")
def read_pdf(file_path: str):
    import fitz
    texte_complet = ""
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texte_page = page.get_text("text") # ou simplement page.get_text()
        texte_complet += texte_page + "\n" # Ajouter un saut de ligne entre les pages

    doc.close()
    return texte_complet

# Dynamic resource template
@mcp.resource("page://{file_path*}/{requested_page}")
def read_pdf_page(file_path: str, requested_page: int):
    import fitz
    texte_complet = ""
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texte_page = page.get_text("text") # ou simplement page.get_text()
        if page_num == requested_page:
            doc.close()
            return texte_page
    doc.close()
    return "Couldn't find content from said page."

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# execute and return the stdio output
if __name__ == "__main__":
    mcp.run(transport="stdio")
