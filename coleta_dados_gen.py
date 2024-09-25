import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import fitz

from src.gpt_utils import call_gpt, truncate_prompt

prompt = """You are an assistant helping researchers looking for insights from publications.

## PUBLICATION
{paper_text}

Now your task is to create a summary of the given publication.
Your summary must:
- Be concise, objective and brief
- Highlight the main ideas and contributions disclosed by the authors

Write the summary with the title and the publication date (if available) inside a json, like the example:
{
    "title": "THE TITLE OF THE PAPER",
    "publication": "DATE in the format yyyy-mm-dd" | "" if not available,
    "arxiv_id": "the arxiv id of the paper" | "" if not avaliable,
    "summary": "the summary requested"
}

## YOUR ANSWER"""


def get_pdf_text(pdf_path):
    """
    Pega nome do pdf e linha do arXiv
    """
    doc = fitz.open(pdf_path)
    textos_paginas = []
    for pagina in doc:
        textos_paginas.append(pagina.get_text())

    return "\n\n".join(textos_paginas)


def generation_pipeline(paper_path):
    paper_path = Path(paper_path)
    paper_hash = hashlib.md5(paper_path.read_bytes()).hexdigest()

    extraction_path = Path(f"./dados_extraidos/{paper_hash}.json")
    if not extraction_path.exists():
        texto = get_pdf_text(paper_path.as_posix())
        p = prompt.replace("{paper_text}", texto)
        p = truncate_prompt(p)
        resposta = call_gpt(p)
        extraction_path.write_text(resposta, encoding="utf8")


if __name__ == "__main__":
    base = pd.read_excel("dados.xlsx", engine="openpyxl")
    for _, row in tqdm(base.iterrows(), total=len(base)):
        generation_pipeline(row.pdf_path)
