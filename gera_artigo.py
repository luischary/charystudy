from pathlib import Path
import time

import fitz
from tqdm import tqdm

from src.gpt_utils import call_gpt, truncate_prompt, check_tokens

main_prompt = """
Você é um escritor e pesquisador de ciência de dados e machine learning e está trabalhando
para uma start up de consultoria em inteligência artificial.

Sua tarefa é produzir postagens para redes sociais para divulgação de novas pesquisas.
Você fará a leitura de um trabalho publicado e em seguida elaborará um post seguindo
a seguinte estrutura:

1. Título chamativo: Um resumo intrigante da sua pesquisa em uma única frase.
2. Contexto leigo: Uma explicação simples sobre o tema e o problema abordado.
3. Ponto técnico: Destaque da inovação ou metodologia empregada. Utilize as nomenclaturas técnicas para que seja possível entender os conceitos do estudo (como se fosse um mini-resumo)
4. Resultados: O impacto prático ou técnico resumido em uma ou duas frases.
5. Call to action: Um convite para engajamento com seu conteúdo ou com você.
6. Hashtags: Use termos relacionados ao tema, como #MachineLearning, #DataScience, #PesquisaAcadêmica.

Utilize emojis para identificação de cada um dos tópicos e para tornar a postagem mais amigável ao público.

Lembrando que a postagem tem como objetivos:
- Atrair o público curioso sobre o tema
- Convencer as pessoas do meu domínio sobre o assunto
- Passar um overview que faça as pessoas (público leigo em geral) entendam 'por cima' sobre o que se trata
- Passar um segundo overview mais técnico de forma que eu consiga atrair e contribuir com o conhecimento com pessoas mais técnicas ou acadêmicas que entendem os conceitos
- Deve ser breve o suficiente para que as pessoas tenham paciência de parar para ler
- Usar uma liguagem fácil, amigável e alegre

Antes de escrever, faça um esboço curto sobre o que poderia ter em cada um dos tópicos e no final
escreva o post.

Depois de escrever a postagem, dê uma lista de sugestões de títulos alternativos para pessoas utilizarem quando forem repostar este conteúdo.

## TEXTO DA PUBLICAÇÃO
{texto_paper}
"""

main_prompt = """
Você é um escritor e pesquisador de ciência de dados e machine learning e está trabalhando
para uma start up de consultoria em inteligência artificial.

Sua tarefa é produzir postagens para redes sociais para divulgação de novas pesquisas.
Você fará a leitura de um trabalho publicado e em seguida elaborará um post seguindo
a seguinte estrutura:

1. Título chamativo: Um resumo intrigante da sua pesquisa em uma única frase.
2. Contexto leigo: Uma explicação simples sobre o tema e o problema abordado.
3. Ponto técnico: Destaque da inovação ou metodologia empregada. Utilize as nomenclaturas técnicas para que seja possível entender os conceitos do estudo (como se fosse um mini-resumo)
4. Resultados: O impacto prático ou técnico resumido, utilizando termos técnicos e não técnicos.
5. Call to action: Um convite para engajamento com seu conteúdo ou com você.
6. Hashtags: Use termos relacionados ao tema, como #MachineLearning, #DataScience.

Utilize emojis para identificação de cada um dos tópicos e para tornar a postagem mais amigável ao público.

Lembrando que a postagem tem como objetivos:
- Atrair o público curioso sobre o tema
- Convencer as pessoas do meu domínio sobre o assunto
- Passar um overview que faça as pessoas (público leigo em geral) entendam 'por cima' sobre o que se trata
- Passar um segundo overview mais técnico de forma que eu consiga atrair e contribuir com o conhecimento com pessoas mais técnicas ou acadêmicas que entendem os conceitos
- Deve ser breve o suficiente para que as pessoas tenham paciência de parar para ler
- Usar uma liguagem amigável, alegre e adequada. Onde falamos de mais aspectos técnicos jargões técnicos podem e devem ser utilizados e nas partes do público em geral a linguagem deve ser mais simples
- Ser impessoal, os trabalhos científicos avaliados geralmente são de outras pessoas

Antes de escrever, faça um esboço curto sobre o que poderia ter em cada um dos tópicos e no final
escreva o post.

Depois de escrever a postagem, dê uma lista de sugestões frases que funcionem como chamadas curtas para o post, coisas como "Olha só essa novidade [conteudo relacionado ao post]" ou "Mais um grande avanço em [conteúdo relacionado ao post]".

Formate a resposta final em um json com o seguinte formato:
{
   "titulo": "titulo do paper/trabalho cientifico",
   "publicacao": "texto para o post",
   "chamadas_post": [lista de possíveis chamadas para o post]
}

## TEXTO DA PUBLICAÇÃO
{texto_paper}
"""


def get_pdf_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    textos_paginas = []
    for pagina in doc:
        textos_paginas.append(pagina.get_text())

    return "\n\n".join(textos_paginas)


def make_post(idx: int, pdf_path: str):
    texto_pdf = get_pdf_text(pdf_path)
    texto_truncado = truncate_prompt(texto_pdf, max_tokens=100_000, model="gpt-4o")
    prompt = main_prompt.replace("{texto_paper}", texto_truncado)

    check_tokens(prompt)
    resposta = call_gpt(prompt, model="gpt-4o-mini", temperature=0.1, max_tokens=4096)
    print(resposta)
    Path(f"./artigos/gerados/{idx}.txt").write_text(resposta, encoding="utf8")


# regex limpeza
# \*\*[a-zA-Zà-ú ]+:\*\*
if __name__ == "__main__":
    initial_idx = 12
    idx = 1
    folder_path = Path(r"C:\Users\Luis Felipe Chary\Downloads\neurips_papers")
    for p in tqdm(folder_path.glob("*.pdf")):
        if idx >= initial_idx:
            make_post(idx, p.as_posix())
            time.sleep(60)
        idx += 1
