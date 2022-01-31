import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import tokenize
import matplotlib.pyplot as plt


def get_num_words_per_sample(text):
    """Obtém a mediana de palavras por amostra determinado corpus.
    # Argumentos
        text: lista, amostra texto.
    # Retorno
        int, mediana de palavras por amostra.
    """
    num_words = [len(s.split()) for s in text]
    return np.median(num_words)

def plot_frequency_distribution_of_words(text, n_words):    
    """Distribuição mostrando a frequência (número de ocorrências) n_pavavras mais frequêntes no conjunto de dados.
    # Argumentos
       text: lista, amostra texto.
       n_words: numero de palavras mais frequêntes a mostrar na plotagem
    # Retorno
       diagrama de pareto.
    """
    textWords = ' '.join([text for text in text])
    tokenizing = tokenize.WhitespaceTokenizer()
    tokenizedWords = tokenizing.tokenize(textWords)
    frequency = nltk.FreqDist(tokenizedWords)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Fequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Fequency", n = n_words)
    plt.figure(figsize=(20,6))
    ax = sns.barplot(data =  df_frequency, x = "Word", y = "Fequency", color = 'gray')
    ax.set(ylabel = "Count")
    plt.show()


def plot_class_distribution(labels):
    """Plota a distribuição de amostra por classe.
    # Arguments
        labels: lista, labels.
    """
    sns.countplot(x=labels, label = 'count')