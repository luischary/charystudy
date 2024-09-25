from pathlib import Path
import json
import datetime

import pandas as pd

root_path = "./catalogo"


def create_md_path(node_name: str, node_parents: list[str] = []):
    p = Path(root_path)
    while len(node_parents) > 0:
        p = p / node_parents.pop(0)

    p = p / node_name
    p = p / "readme.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Paper:
    def __init__(self, name, arxiv_id, arxiv_date, description=""):
        self.name = name
        self.id = arxiv_id
        self.date = arxiv_date
        self.description = description


class No:
    def __init__(self, name, parents: list[str] = []):
        self.name = name
        self.papers = []
        self.children = []
        self.parents = parents
        # para montar o markdown
        self.template_title = """## {topic}\n\n"""
        self.template_name = """### {name}"""
        self.template_date = """**{date}**"""
        self.template_url = """https://arxiv.org/pdf/{arxiv_id}"""
        self.template_description = """\n\n{description}"""

    def create_markdown(self):
        # precisa ordenar pela data antes, o que nao tiver data vai ser ordenado
        # por ordem alfabetica
        com_data = []
        sem_data = []
        for p in self.papers:
            if p.date is not None:
                # ja adiciona na ordem
                indice_insere = 0
                for i in range(len(com_data)):
                    if com_data[i].date > p.date:
                        break
                    indice_insere += 1
                com_data.insert(indice_insere, p)
            else:
                sem_data.append(p)
        # agora precisa so ordenar o cara sem data
        sem_data.sort(key=lambda x: x.name)

        # agora comeca o markdown
        md = self.template_title.format(topic=" -> ".join(self.parents + [self.name]))
        # adiciona os caras sem data primeiro
        for p in sem_data:
            md += "\n\n" + self.template_name.format(name=p.name)
            if p.id != "":
                md += "\n\n" + self.template_url.format(arxiv_id=p.id)
            if p.description != "":
                md += self.template_description.format(description=p.description)
            md += "\n\n---"
        # agora os caras ordenados pela data
        for p in com_data:
            md += "\n\n" + self.template_name.format(name=p.name)
            md += "\n\n" + self.template_date.format(date=p.date.strftime("%Y-%m-%d"))
            if p.id != "":
                md += "\n\n" + self.template_url.format(arxiv_id=p.id)
            if p.description != "":
                md += self.template_description.format(description=p.description)
            md += "\n\n---"

        file_path = create_md_path(node_name=self.name, node_parents=self.parents)
        file_path.write_text(md, encoding="utf8")
        return md

    def __str__(self):
        texto = "## "
        texto += " -> ".join(self.parents + [self.name]) + "\n\n"
        texto += "\n".join([f"{p.name}\t{p.date}\t{p.id}" for p in self.papers])
        texto += "\n\n" + "*" * 50
        return texto

    def has_child(self, name):
        tem = [c.name == name for c in self.children]
        return any(tem)


class Arvore:
    def __init__(self, root_name: str):
        self.root = No(root_name)

    def add_node(self, node: No, parents: list[str] = []):
        curr_node = self.root
        while len(parents) > 0:
            procura = parents.pop(0)
            achou = False
            for c in curr_node.children:
                if c.name == procura:
                    curr_node = c
                    achou = True
                    break
            if not achou:
                return False
        if not curr_node.has_child(node.name):
            curr_node.children.append(node)
        return True

    def add_paper(self, paper: Paper, parents: list[str] = []):
        curr_node = self.root
        while len(parents) > 0:
            procura = parents.pop(0)
            achou = False
            for c in curr_node.children:
                if c.name == procura:
                    curr_node = c
                    achou = True
                    break
            if not achou:
                return False
        curr_node.papers.append(paper)
        return True

    def traverse(self):
        curr_node = self.root
        print(curr_node.create_markdown())
        queue = curr_node.children
        while len(queue) > 0:
            curr_node = queue.pop(-1)
            print(curr_node.create_markdown())
            queue += curr_node.children


# estrutura
def faz_nlp(root=Path("./Papers")):
    dados = pd.read_excel("dados.xlsx", engine="openpyxl")
    dados = dados.fillna("")
    dados = dados[dados.classe_1 != ""].reset_index(drop=True)
    dados["arxiv_date"] = pd.to_datetime(dados["arxiv_date"])
    arvore = Arvore("raiz")
    # para remover duplicidades
    hashes_adicionados = set()
    for _, row in dados.iterrows():
        paper_hash = row.pdf_hash
        if paper_hash in hashes_adicionados:
            continue
        else:
            hashes_adicionados.add(paper_hash)
        extracted_path = Path(f"./dados_extraidos/{paper_hash}.json")
        nome = row.pdf_name
        arxiv_id = row.arxiv_id
        data = row.arxiv_date if not pd.isna(row.arxiv_date) else None
        summary = ""
        if extracted_path.exists():
            try:
                dados = json.loads(extracted_path.read_text(encoding="utf8"))
                nome = dados["title"]
                if dados["arxiv_id"] != "":
                    arxiv_id = dados["arxiv_id"]
                if dados["publication"] != "":
                    data = datetime.datetime.strptime(dados["publication"], "%Y-%m-%d")
                summary = dados["summary"]
            except:
                print("sem dados extraidos para o paper")

        p = Paper(nome, arxiv_id, data, summary)

        tipo1 = row.classe_1
        tipo2 = row.classe_2
        tipo3 = row.classe_3

        if tipo1 != "":
            arvore.add_node(No(tipo1))

        if tipo2 != "":
            arvore.add_node(No(tipo2, [tipo1]), [tipo1])

        if tipo3 != "":
            arvore.add_node(No(tipo3, [tipo1, tipo2]), [tipo1, tipo2])

        if tipo3 != "":
            arvore.add_paper(p, parents=[tipo1, tipo2, tipo3])
        elif tipo2 != "":
            arvore.add_paper(p, parents=[tipo1, tipo2])
        elif tipo1 != "":
            arvore.add_paper(p, parents=[tipo1])
        else:
            arvore.add_paper(p)

    arvore.traverse()


if __name__ == "__main__":
    faz_nlp()
