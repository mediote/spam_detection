# Relatório dos procedimentos executados e resultados obtidos.

## Objetivos

- Treinar e avaliar um modelo **Multi Layer Perceptron** (MLP) para classificação de **SPAM** usando o dataset **train_data.**
- Treinar e avaliar um modelo de **Regressão Logística** com o mesmo objetivo a título de comparação.
- Analisar os erros de predição com o **LIME.**
- Classificar a coluna **SMS** do dataset **validation_data** como **“ok”** ou **“blocked”** a partir dos modelos treinados.

## Explorando o dataset

A partir das amostras de texto presentes na colula **“SMS”** do dataset **train_data,** foram extraidas métricas que auxiliaram a entender os dados, como prepara-los e na difinição de critérios para a escolha do modelo adequado:

- **Número de amostras:** total de amostras do datset.
- **Número de classes**: total de classes no dataset na coluna **“LABEL”**.
- **Número de amostras por classe:** número de exemplos por classe.
- **Mediana de palavras por amostra:** mediana do número de palavras em uma unica amostra em todo dataset.
- **Distribuição de frequência:** gráfico com a distribuição do número de ocorrências das 15 palavras mais frequêntes no dataset.

| Métrica | Valor |
| --- | --- |
| Número de amostras | 6000 |
| Número de classes | 2 |
| Número de amostras classe  “ok” | 4500 |
| Número de amostras classe  “blocked” | 1500 |
| Mediana de palavras por amostra | 10 |

**Tabela 1: train_data métricas.**

![distribuicao-orig.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/distribuicao-orig.jpg)

    **Figura 1: Distribuição de frequência.** 

```
Exemplos de SMS não bloqueadas:

recuperamos seu usuario e senha de acesso no infojobs! usuario: carolalmeidagaldino20@xn--bo-9pa.com.br. senha: miguel28. obrigado! 

MARSH CORRETORA: Anna, boleto parc. 01 do Seg Auto com venc.: 28/12/2018 enviado para:anna.barroso@c-a-m.com com esclarecimentos e instrucoes 

Host : RB_Bicanga Ip: 170.244.231.14 nao esta respondendo ao ping - 2019-04-19 22:30:23

----------------------------------------------------------------------------------------

Exemplos de SMS bloqueadas:

BOLETO REFERENTE AS PARCELAS EM ATRASO DO CONSÓRCIO PELO BB.COM VENCIMENTO PARA HOJE Ñ PODE HAVER QUEBRA NO ACORDO. BONATTO ADV 0800 606 3301.

050003DA0202|lcloud-apple-lnc.com/?iphone=VtBqROY .

BB INFORMA:VALIDE SUA SENHA E EVITE TRANSTORNO. ACESSE: www.Bbrasildesbloqueio.com/?7R8BQ8CI
```

**Figura 2: Amostras de texto** 

Com base na **Tabela 1**, observa-se que existem 2 classes e que elas estão desbalanceadas, além disso, a distribuição no **Gráfico 1** e a **Figura 2** mostram que o texto contém letras maiúsculas, minúsculas, números, pontuação, links, stopwords e caracteres especiais.

## Escolha dos modelos

Os modelos podem ser amplamente classificados em duas categorias: os que usam informações de ordenação de palavras (modelos de sequência) e aqueles que apenas veem o texto como “sacos” (conjuntos) de palavras (modelos n-gram). 

Os modelos de sequência incluem redes neurais convolucionais **(CNNs)**, redes neurais recorrentes **(RNNs)** e suas variações. Os tipos de modelos **n-gram** incluem regressão logística, multi layer perceptrons simples **MLPs** ou redes neurais totalmente conectadas, gradient boosted trees e support vector machines.

Com base nas informações acima e nas métricas extraídas, foi levado em consideração a razão entre o **número de amostras (S)** e a **mediana de palavras por amostra (W)** como principal critério ****para a escolha do modelo.  Quando o valor dessa razão é pequeno (<1500), **MLPs** ou **Regresçoes Logísticas** alimentandas por n-grams possuem um bom desempenho.

Nesta análise, o valor **S/W** obtido no dataset **train_data** foi **** de 600 ( 6000 / 10) , por isso foram escolhidos os modelos **MPLs** e **Regressão Logística**.

## Preparando os dados

Os dados passaram pelas seguintes etapas: 

1. **Pré-processamento:** apesar de não ter influenciado significativamente no desempenho geral do modelo, foi incluida uma etpa de pré-processamento para remoção de acentuação, stopwords e o texto foi colocado em lowercase.
2. **Downsampling da maioria:** as classes com a maioria de amostras foram balanceadas de acordo com as classes com o menor número de amostras. Testes executados, demostraram uma melhora nos resultados. 
3. **Holdout:** os dados foram divididos em  subconjuntos mutuamente exclusivos, de treinamento e teste na proporção **70/30** respectivamente.
4. **Tokenizção e Vetorização**: divisão do texto em tokens e conversão em vetores numéricos com **TfidfVectorizer**.
5. **Feature Selection:** selcionado as top 20.000 features mais importantes para determinado rótulo com **SelectKbest** e **f-classif.**

## Construção, treino e avaliação dos Modelos

## Multi Layer Perceptron

Para construção do modelo **MLPs,**  foram usados os frameworks **TensorFlow** e **Keras**. O modelo possui duas camadas **Dense**, adicionando algumas camadas **Dropout** para regularização (para evitar overfitting). Foi utilizado o callback **EarlyStop** para interromper o treinamento quando os validadion loss não diminuirem em dois passos consecutivos.

Os paramêtros para treinar o modelo foram:

```python
learning_rate=1e-3,
epochs=1000,
batch_size=128,
layers=2,
units=64,
dropout_rate=0.2
```

Após executar a função de treinamento, o modelo convergiu em **29** épocas com uma perda média de **0.0082** e acurácia de **~99.7 %** conforme a linha abaixo.

```python
29/29 - 0s - loss: 0.0082 - acc: 0.9967 - 25ms/epoch - 874us/step
[0.008226083591580391, 0.996666669845581]
```

Na **Figura 3a**, observamos a relação entre a acurácia nas amostras de treino e teste e a evolução das épocas. Os resultados mostram que o modelo generaliza adequadamente. **A Figura 3b**, no mesmo sentido, mostra a diminuição dos erros à medida que a acurácia aumenta no decorrer das épocas. 

![hsitory.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/hsitory.jpg)

                    **Figura 3a: Treino e Validação acurácia.                Figura 3b  Treino e Validação perda.**

Através da matriz de confusão e das métrica na **Figura 4**, podemos ter mais informações sobre o desempenho do modelo de classificação em questão. O modelo classificou corretamente **462** das 464 amostras **não spam** , obtendo **Precision = 99,5%**, porém classficou **erroneamente** como **não spam** uma das **436** amostras **spam**, alcançando um  **Recall = 99.8%**.

![matriz.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/matriz.jpg)

                                               **Figura 4: Matriz de confusão e métricas de classificação MLP.**

Para entender os erros cometidos pelo modelo, foi usado o **LIME**. Através dele, é possível inspecionar localmente os erros de predição para entender quais features exerceram mais infliência.  Na **Figura 5**, a amostra analisada é um falso negativo, algo indesejado quando se trata de segurança.

A partir de insights fornecidos pelo **LIME**, é possivel alterar algumas abodagens como pré-processamento, tokenização dentre outras coisas e com isso melhorar a qualidade do modelo.

![explicabilidade.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/explicabilidade.jpg)

    **Figura 5:  Explicação do Lime para um falso negativo MLP.**

Os termos **15, you, to, code, sent e with** estão contribuindo para o modelo classificar como não spam e os termos **http, itunes, com e link** para classificar como spam. 

## Regressão Logística

O modelo de  Regressão Logística obteve  uma **Accuracy** = **99,7%.** O modelo classificou corretamente **todas** as **464** amostras spam com **Precision = 100%** porém errou **três** das **436** amostras não spam com **Recall = 99,3%**.

![matriz2.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/matriz2.jpg)

                                          **Figura 5: Matriz de confusão e métricas de classificação Regressão.**

Na **Figura 6**, os termos **15, you, to, code, sent e with** estão contribuindo para o modelo classificar como não spam e os termos **http, com e link** para classificar como spam.

![lime.jpg](Avaliac%CC%A7a%CC%83o%20te%CC%81cnica%20Axur%200bee9d675d504ce5ab3ebd738bf0adb0/lime.jpg)

**Figura 6:  Explicação do Lime para um falso negativo Regressão.**

## Conclusão

Foi criado um modelo **Multi Layer Perceptron** usando frameworks como **Keras** e **TensorFlow** para classificar dados de **SMS** do dataset **train_data.** Após vários testes o modelo atingiu um bom resultado mostrando ser aplicável em dados reais. O código e os resultados desse modelo estão disponíveis no arquivo **mlp.ipynb.**

A título de comparação, foi criado um modelo de **Regressão Logística** usando o **scikit-learn,** adotando os mesmos critérios para **Preparação dos Dados.** O código e os resultados desse modelo estão disponíveis no arquivo **regressão_logistica.ipynb.**

Apesar do **Regressão Logística** ser mais simples, os resultados obtidos foram satisfatórios, com desempenho semelhante ao modelo **MLP** porém com um maior número de falsos negativos, com **Recall = 99,3%** contra **99,8%** alcançados pela **MLP**.

Essa diferença de **Recall** deve ser levada em consideração na escolha dentre os modelos, principalmente em um cenário de segurança onde um erro de classificação pode causar prejuízos.

Foram analisados os erros de predição cometidos por ambos modelos com o **LIME.** A amostra analisada é a mesma nos dois modelos. Ambos parecem considerar as mesmas features para classificar as amostras.
