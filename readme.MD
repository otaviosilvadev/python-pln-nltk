# PLN para Análise de Sentimentos com Python

## Sumário

- [Tecnologias](#tecnologias-e-ferramentas)
- [PLN](#pln-processamento-de-linguagem-natural)
- [Como iniciar](#como-iniciar)

## Tecnologias e Ferramentas:

- [Python](https://www.python.org/)
- [NLTK](https://www.nltk.org/)

## PLN (Processamento de Linguagem Natural)

PLN é uma área de Inteligência Artificial (IA) que estuda e desenvolve métodos para que computadores possam entender, interpretar e gerar linguagem humana.

É comumente utilizada para Análise de Sentimentos, consistindo em um tipo de processamento para identificar se determinado texto é positivo, negativo ou neutro.

Para este caso, o processo consiste em etapas que de modo geral são definidas em tokenização, stopwords, lemmatização, e processamento.

- Tokenização: remoção de espaços, números, caracteres especiais etc.
- Stopwords: remoção de palavras irrelevantes, como por exemplo pronomes e artigos. A biblioteca NLTK dispõe de uma base de dados com stop_words em diferentes idiomas.
- Lematização: gramática para encontrar palavras, especialmente verbos, no infinitivo. Ex: transforma "correndo", "correu", "corrida" em apenas "correr". O NLTK também dispõe de um módulo pronto para lematizizar palavras.
- Processamento: pode-se utilizar aqui um modelo já treinado como o SentimentIntensityAnalyzer do NLTK ou treinar um modelo novo específico para um cenário ou caso de uso.

### Código-fonte

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import nltk

# downloads necessários
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# stopwords em português (vai usar apenas para filtrar palavras irrelevantes)
stop_words = set(stopwords.words('english'))

# lematizador e analisador VADER
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

translator = Translator()

def process(texto):
    # tradução para inglês
    texto_en = translator.translate(texto, src='pt', dest='en').text
    # pré-processamento simples
    tokens = texto_en.lower().split()
    tokens = [t for t in tokens if t.isalpha()]  # remove pontuação
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lematiza em inglês
    texto_processado = ' '.join(tokens)
    # análise de sentimento em inglês
    sentimento = sia.polarity_scores(texto_processado)
    print(f"\nFrase original: {texto}")
    print(f"Frase processada (inglês): {texto_processado}")
    print(f"Análise de sentimentos: {sentimento}")

# lista de frases (vírgulas corrigidas)
frases = [
    "Eu realmente amei o hamburger! É maravilhoso e saboroso.",
    "Eu odiei o hamburger. A entrega é lenta e não atendem o telefone.",
    "A pizza é boa, não tem nada de especial mas não é ruim."
]

# processar todas as frases
for f in frases:
    process(f)
```

### Saída

```bash
Frase original: Eu realmente amei o hamburger! É maravilhoso e saboroso.
Frase processada (inglês): really loved wonderful
Análise de sentimentos: {'neg': 0.0, 'neu': 0.109, 'pos': 0.891, 'compound': 0.847}

Frase original: Eu odiei o hamburger. A entrega é lenta e não atendem o telefone.
Frase processada (inglês): hated slow answer
Análise de sentimentos: {'neg': 0.677, 'neu': 0.323, 'pos': 0.0, 'compound': -0.6369}

Frase original: A pizza é boa, não tem nada de especial mas não é ruim.
Frase processada (inglês): pizza nothing special
Análise de sentimentos: {'neg': 0.53, 'neu': 0.47, 'pos': 0.0, 'compound': -0.3089}
```

## Como Iniciar

Crie o ambiente virtual:

```bash
python3 -m venv venv
```

Inicie o ambiente virtual:

- Windows

```
.\venv\Scripts\activate
```

- Linux

```
source venv/bin/activate
```

Instale as dependências e inicie o projeto:

```bash
pip install -r requirements.txt
python3 src/index.py
```

## Autor

Otávio Silva

[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/otaviosilva22/)](https://www.linkedin.com/in/otaviosilva22/)
[![Gmail Badge](https://img.shields.io/badge/-Gmail-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:otavio.ssilva22@gmail.com)](mailto:otavio.ssilva22@gmail.com)
