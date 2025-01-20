from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import fitz

from src.gpt_utils import call_gpt, truncate_prompt

main_prompt = """Você é um escritor e resumidor de artigos científicos.
Seu trabalho é ler e consolidar informações sobre artigos publicados.

Você deve resumir os artigos seguindo a seguinte estrutura:

1. Overview sobre o que se trata o trabalho.
2. Quais foram as principais contribuições do trabalho.
3. Descrição técnica.
4. Resultados obtidos.
5. Aplicações ou possíveis aplicações práticas.

Para elaboraçao do resumo faça antes um rascunho bastante breve sobre cada um dos tópicos que o resumo deve ter com base no que foi lido.

Além do resumo você também deve dar 3 classificações sobre o trabalho:

1. Macrotópico: NLP, Processamento de imagens, otimização, reinforcement learning, GAN, processamento de áudio, dataset, outros.
2. Tipo de modelo envolvido: geral (deep learning), transformer, redes convolucionais, autoencoder, outros.
3. Tipo de aplicação: modelo generativo, modelo de classificação, modelo de regressão, otimizador, outros.

Justifique a classificação escolhida e em seguida selecione apenas uma opção de cada tipo para colocar na sua resposta.

Por último seria muito importante seria muito importante termos uma nota de uso prático do trabalho em questão.
Utilize os seguintes critérios para dar uma nota de 1 a 5 para a publicação:

1. Trabalho totalmente teórico, sem testes em casos práticos.
2. Trabalho possui aplicação prática, porém, muito nichada ou restrita aos dados dos autores.
3. Trabalho possui aplicação prática, porém, os autores não conseguiram realizar todos os testes que os demonstrem.
4. Trabalho possui aplicação prática e bons resultados foram demonstrados, porém necessitam de hardware ou software muito específicos para serem replicados.
5. Trabalho possui aplicação prática, bons resultados foram demonstrados e seria relativamente fácil o emprego da técnica por outras pessoas. 

Justifique a sua escolha em em seguida dê a sua nota.

Após o rascunho do resumo e suas justificativas das classificações e nota, escreva sua resposta. Ela deve ser um json no seguinte formato:
{
    "resumo": "SEU RESUMO",
    "macrotopico": "OPCAO DO MACROTOPICO ESCOLHIDA",
    "modelo": "OPCAO DO MODELO ESCOLHIDA",
    "aplicacao": "OPCAO DA APLICACAO ESCOLHIDA",
    "nota_pratica": "SUA NOTA DE AVALIACAO PRATICA"
}

## TEXTO DA PUBLICAÇÃO
{publicacao}
"""

main_prompt = """You are a writter and summarizer of scientific papers.
Your job is to read and consolidate info about published papers.

You must summarize the papers following the structure:

1. Overview about the paper.
2. What are its main contributions.
3. A brief description of the proposed technique.
4. Main results.
5. Applications or possible practical applications.

To elaborate the summary make a brief sketch before about the content that each topic must have.

Besides the summary you also must classify the paper in three different ways:

1. Macrotopic: NLP, image processing, optimization, reinforcement learning, GAN, audio processing, dataset, others.
2. Type of model employed: general (deep learning), transformer, convolutional networks, autoencoder, recurren networks, state-space models, others.
3. Type of application: generative model, classification model, regression model, optimizer, others.

If you think you can come up with a better option to classify, feel free to add more options, just make sure it's the right move.
First justify your classification and then select only one option of each kind for your response.

At least it is very important that we have a grade for the practical application of the paper.
Use the following criteria to grade the paper with grades from 1 to 5:

1. Totally theoric. No practical cases were tested
2. The paper has some practical application, but very niched or restricted to the author's data.
3. The paper has practical applications but the authors could not perform the tests that prove it or did not surpass the state of the art.
4. The paper has practical application and good results were proved, but it is necessary very specialized hardware (eg. clusters of gpus) or software (custom kernels or libs) to be replicated.
5. The paper has practical application, good results were shown and it whould be relativelly easy to use the same technique or idea by other people. Or the work is so innovative and/or outstanding that everyone should read it.

Justify you choice before giving your grade.

After the sketch of the summary and the classification and grade justification it's time to write your response.
It must be a json with the following structure:
{
    "summary": "YOUR SUMMARY",
    "macrotopic": "MACROTOPIC OPTION",
    "model": "MODEL OPTION",
    "application": "APPLICATION OPTION",
    "grade": "YOUR GRADE"
}

## TEXT OF THE PAPER
{publicacao}
"""


def get_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    textos_paginas = []
    for pagina in doc:
        textos_paginas.append(pagina.get_text())

    return "\n\n".join(textos_paginas)


def faz_resumos():
    root_path = Path(r"C:\Users\Luis Felipe Chary\Downloads\neurips_papers")
    for planilha in ["fotos", "workshops"]:
        base = pd.read_excel(
            r"C:\Users\Luis Felipe Chary\OneDrive\Documents\anotacoes_neurips.xlsx",
            engine="openpyxl",
            sheet_name=planilha,
        )
        for row in tqdm(base.itertuples(), total=len(base)):
            idx = row.id
            # procura o paper
            paper_folder = root_path / planilha
            caminhos = list(paper_folder.glob(f"{idx}_*.pdf"))
            if len(caminhos) > 0:
                paper_path = caminhos[0]
            else:
                paper_path = root_path / planilha / f"{idx}.pdf"
            if paper_path.exists():
                output_path = Path(f"./neurips/{planilha}/{idx}.txt")
                if output_path.exists():
                    continue
                paper_text = get_pdf_text(paper_path.as_posix())
                paper_prompt = main_prompt.replace("{publicacao}", paper_text)
                paper_prompt = truncate_prompt(
                    paper_prompt, max_tokens=120_000, model="gpt-4o-mini"
                )
                resposta = call_gpt(
                    paper_prompt, model="gpt-4o-mini", temperature=0.1, max_tokens=4096
                )
                print(resposta)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(resposta, encoding="utf8")

                time.sleep(10)


if __name__ == "__main__":
    faz_resumos()
