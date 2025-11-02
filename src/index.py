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