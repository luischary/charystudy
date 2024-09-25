import re
from pathlib import Path

import pandas as pd
import fitz

NUM_CLASSES = 3


def get_arxiv_id(text):
    """Gets the id from arxiv, eg. arXiv:1904.10509v1"""
    pattern = r"(?<=:)\d{4}\.\d{5}"
    lista = re.findall(pattern, text)
    if len(lista) > 0:
        return lista[0]
    else:
        return ""


def get_arxiv_date(texto):
    stonum = {
        "Jan": "1",
        "Feb": "2",
        "Mar": "3",
        "Apr": "4",
        "May": "5",
        "Jun": "6",
        "Jul": "7",
        "Aug": "8",
        "Sep": "9",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    pattern = r"\d{1,2}\s[Jan|Feb|Mar|Abr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec]+\s\d{4}"
    lista = re.findall(pattern, texto)
    if len(lista) > 0:
        date_str = lista[0]
        # substitui o mes pelo numero
        for mes in stonum.keys():
            if date_str.find(mes) >= 0:
                date_str = date_str.replace(mes, stonum[mes])
                date_str = date_str.replace(" ", "-")
                break
        return date_str
    return ""


def get_linha_dados(texto):
    """
    Na linha que tem os dados do arxiv temos id e data
    """
    # linhas = texto.split("\n")
    # for l in linhas:
    #     if l.find("arXiv:") >= 0:
    #         return l
    patterns = [
        r"arXiv:\d+\.\d+[v\d{1,2}]*\s{2}\[.*\]\s{2}\d{1,2}\s[Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec]+\s\d{4}",
        r"arXiv:\d+\.\d+",
    ]
    for p in patterns:
        lista = re.findall(p, texto)
        if len(lista) > 0:
            return lista[0]
    return None


def extrai_dados_basicos(pdf_path):
    """
    Pega nome do pdf e linha do arXiv
    """
    doc = fitz.open(pdf_path)
    textos_paginas = []
    for pagina in doc:
        textos_paginas.append(pagina.get_text())

    dados = {}
    dados["name"] = Path(pdf_path).stem
    dados["arxiv_line"] = get_linha_dados("\n".join(textos_paginas))
    return dados


def mapeia_repo(root_path):
    """
    Pega os dados dos pdfs e tambem ja guarda a extrutura que esta organizado
    para aproveitar a hierarquia de classificacao dos dados
    """
    dados_coletados = {"pdf_path": [], "pdf_name": [], "arxiv_line": []}
    pastas = [Path(root_path)]
    while len(pastas) > 0:
        p = pastas.pop(0)
        print(p.stem)
        for f in p.iterdir():
            if f.is_dir():
                pastas.append(f)
            elif f.suffix.lower() == ".pdf":
                dados_pdf = extrai_dados_basicos(f.as_posix())
                dados_coletados["pdf_path"].append(f.as_posix())
                dados_coletados["arxiv_line"].append(dados_pdf["arxiv_line"])
                dados_coletados["pdf_name"].append(dados_pdf["name"])

    dados = pd.DataFrame(dados_coletados)
    dados.to_excel("extracao.xlsx", index=False)


def extrai_classificacao(row):
    # faz a classificacao em cima do path
    p = row.pdf_path
    nome = row.pdf_name
    campos = p.split("/")
    classes_paper = []
    achou_papers = False
    indice = 0
    while not achou_papers:
        campo = campos[indice]
        if campo == "Papers":
            achou_papers = True
        indice += 1

    for i in range(NUM_CLASSES):
        j = indice + i
        if j < len(campos):
            if campos[j].find(nome) < 0:
                classes_paper.append(campos[j])
            else:
                classes_paper.append("")
        else:
            classes_paper.append("")
    return classes_paper


def faz_campos(df_path):
    df = pd.read_excel(df_path, engine="openpyxl")
    df = df.fillna("")
    arxiv_id = []
    arxiv_date = []
    classifica = []
    for _, row in df.iterrows():
        dados_arxiv = row.arxiv_line
        if dados_arxiv != "":
            id_arxiv = get_arxiv_id(dados_arxiv)
            data_arxiv = get_arxiv_date(dados_arxiv)
        else:
            id_arxiv = ""
            data_arxiv = ""

        arxiv_id.append(id_arxiv)
        arxiv_date.append(data_arxiv)

        classifica.append(extrai_classificacao(row))

    df["arxiv_id"] = arxiv_id
    df["arxiv_date"] = arxiv_date

    for i in range(1, NUM_CLASSES + 1):
        df[f"classe_{i}"] = [c[i - 1] for c in classifica]

    df.to_excel("dados.xlsx", index=False)


if __name__ == "__main__":
    mapeia_repo(r"C:\Users\Luis Felipe Chary\Downloads\papers\Papers")
    faz_campos("extracao.xlsx")
