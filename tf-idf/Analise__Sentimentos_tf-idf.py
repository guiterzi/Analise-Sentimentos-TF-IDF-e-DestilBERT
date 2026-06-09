import pandas as pd
import matplotlib.pyplot as plt
import re
import unicodedata
# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np


df_reviews = pd.read_csv("B2W-Reviews01.csv")

#-------------------------------Analise exploratoria-------------------------------------

# print(df_reviews.info())
# print(len(df_reviews))

#Vendo o head de cada coluna para saber como tao dispostas as informacoes de cada coluna
# for i in df_reviews:
#     print(df_reviews[i].head())
#     print("\n")

#Observacoes:
#Como sera feita uma classificacao de sentimentos usando a vetorizacao TF-IDF e algum algoritmo 
# classico de classificacao, entao as colunas de interesse são:

# 1. A avaliação escrita do usuario ta separada em titulo e a nota escrita, elas tao em review_title e review_text
# há mais review_title que review_text, entao alguns usuarios usaram somente o titulo para avaliar

# 2. Tem a nota dada ao produto pelo usuario em overall_rating, avaliando quais valores tem nela:
# print(df_reviews["overall_rating"].unique())
# A nota tem 5 valores: de 1 a 5

# 3. Tem a coluna se o usuario recomenda o produto, tá em recommend_to_a_friend
# print(df_reviews["recommend_to_a_friend"].unique())
#Tem 3 valores: sim, nao e nan
# print(df_reviews["recommend_to_a_friend"].isna().sum())
# Só 18 valores sao nan, comparado ao conjunto de dados que tem 132k de registros, então será removido

#Analisando a relacao entre recommend_to_a_friend e overall_rating:
# Tabela de contagem
# print(df_reviews.groupby(["overall_rating", "recommend_to_a_friend"]).size().unstack())

# Proporção por nota
# proporcao = df_reviews.groupby("overall_rating")["recommend_to_a_friend"].value_counts(normalize=True).unstack()
# print(proporcao)


#Alguns resultados ja eram esperados, quem deu nota 1 nao recomenda na maioria das vezes e quem deu nota 4 ou 5 recomenda
#na maioria das vezes. Mas tem alguns mais curiosos, quem deu nota 2 recomenda em 25% das vezes, e quem deu nota 3
#recomenda em 88% das vezes.
#Entao a avaliacao é um pouco ambigua, o usuario apesar de nao gostar do produto ainda recomenda ele, ou gosta do produto e nao
# recomenda, como recommend_to_a_friend é uma resposta mais direta, só tem sim ou não, então vamos usar ela como alvo
#Os atributos vamos juntar o review_title com review_text para usar no TF-IDF


#---------------------------------------- Tratamento -------------------------------------------------------

#Tratar as colunas de interesse para entregar o mais 'limpo' possivel para a vetorizacao TF-IDF

#Remover as linhas que tem nan do recommend_to_a_friend
df_reviews = df_reviews.dropna(subset=["recommend_to_a_friend"])
# print(len(df_reviews))

#Como vamos utilizar o titulo e o corpo da mensagem, entao concatenamos para tratar eles juntos
# print(df_reviews["review_title"].head())
# print(df_reviews["review_text"].head())
df_reviews["review_completo"] = df_reviews["review_title"].fillna("") + " " + df_reviews["review_text"].fillna("")
# print(df_reviews["review_completo"])

#Cortando onde o review_completo foi feito de review_title vazio e df_review_text tmb vazio, sobrando só o espaço da juncao dos dois
df_reviews = df_reviews[df_reviews["review_completo"].str.strip().str.len() > 0]
# print(len(df_review))

#Padronizando o texto para minusculo

df_reviews["review_completo"] = df_reviews["review_completo"].str.lower()
# print(df_reviews["review_completo"].head())


#Tirando a acentuacao
def remover_acentos(texto):
    if not isinstance(texto, str):
        return ""
    
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    texto_sem_acentos = texto_normalizado.encode("ascii", "ignore")
    texto_final = texto_sem_acentos.decode("ascii")
    return texto_final

df_reviews["review_completo"] = df_reviews["review_completo"].apply(remover_acentos)
# print(df_reviews["review_completo"].head())

#Removendo os caracteres especiais
df_reviews["review_completo"] = df_reviews["review_completo"].str.replace(r"[^a-z ]", "", regex=True)
# print(df_reviews["review_completo"].head())


#Agora a remocao dos stop-words para o uso do TF-IDF
stops = set(stopwords.words("portuguese"))
# print(sorted(stops))
# "não" é uma stop-word, mas como tratou antes de tirar o acento, então ela não foi afetada
def remover_stopwords(texto):
    palavras = texto.split()
    palavras_filtradas = []
    for palavra in palavras:
        if palavra not in stops:
            palavras_filtradas.append(palavra)
    return " ".join(palavras_filtradas)

df_reviews["review_completo"] = df_reviews["review_completo"].apply(remover_stopwords)
# print(df_reviews["review_completo"])


#Criar o atributo de sentimentos a partir do recommend_to_a_friend

#Como a recomendação é uma string, Yes e No, entao vamos mapear ela para 1-Yes e 0-NO
df_reviews["sentimentos"]= df_reviews["recommend_to_a_friend"].map({"Yes":1,"No":0})
# print(df_reviews["sentimentos"].value_counts())
#Aqui temos um problema, as classes estao desbalanceadas. Tem mto mais classes Sim do que nao
#Entao temos que fazer um "balanceamento" delas, senao o modelo vai sempre tentar prever a classe
#predominante. O balanceamento pode ser feito na divisao do treino,validacao e teste e tambem
#no parametro da regressao logistica


#------------Divisao treino, validacao e teste -------------------
#Vamos separar em 60% para treino, 20% validacao e 20% para teste

#Atributos e alvo
X = df_reviews["review_completo"]
y = df_reviews["sentimentos"]

#Separar treino (60%)
X_treino, X_resto, y_treino, y_resto = train_test_split(X,y,train_size=0.6,stratify=y, random_state=7)

#Separa teste(20%) e validação(20%)
X_validacao, X_teste, y_validacao, y_teste = train_test_split(X_resto,y_resto, test_size=0.5, stratify=y_resto,random_state=7)

# print(f"Treino:    {len(X_treino)}")
# print(f"Validação: {len(X_validacao)}")
# print(f"Teste:     {len(X_teste)}")
# print(y_treino.value_counts(normalize=True))
# print(y_validacao.value_counts(normalize=True))
# print(y_teste.value_counts(normalize=True))

#---------------------------------------- TF-IDF ------

#Cria um objeto pra transformar o texto em numeros, tamo usando tambem um bigrama
#tipo palavras como "nao recomendo" sao uteis
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))

#Usa o treino para aprender o vocabulario, e os outros só transforma o texto em um vetor numerico
X_treino_tfidf = tfidf.fit_transform(X_treino)
X_validacao_tfidf   = tfidf.transform(X_validacao)
X_teste_tfidf  = tfidf.transform(X_teste)

#------Modelagem da Regressao Logistica----


#Encontrando o melhor c na validacao
for i in [0.001, 0.01, 0.1, 1, 10, 100]:
    regressao_log = LogisticRegression(max_iter=1000, C=i)
    regressao_log.fit(X_treino_tfidf, y_treino)

    val_acc = regressao_log.score(X_validacao_tfidf, y_validacao)
    print(f"Para c = {i}    Val Accuracy: {val_acc:.4f}\n")

# Usando o melhor c no teste
regressao_log_melhor = LogisticRegression(max_iter=1000, C=1)
regressao_log_melhor.fit(X_treino_tfidf, y_treino)
y_pred_test = regressao_log_melhor.predict(X_teste_tfidf)

print(classification_report(y_teste, y_pred_test))
print(confusion_matrix(y_teste, y_pred_test))
print(confusion_matrix(y_teste, y_pred_test))

np.save("y_pred_tfidf.npy", y_pred_test)
np.save("y_teste_tfidf.npy", y_teste.to_numpy())
np.save("X_teste_tfidf.npy", X_teste.to_numpy())


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Pegar os nomes das features (palavras)
features = tfidf.get_feature_names_out()

# Pegar os coeficientes do modelo
coeficientes = regressao_log_melhor.coef_[0]

# Criar DataFrame
df_pesos = pd.DataFrame({"palavra": features,"peso":coeficientes})

# Top 20 palavras mais positivas (Yes)
print("── Palavras mais positivas ──")
print(df_pesos.nlargest(20, "peso")[["palavra", "peso"]].to_string())

# Top 20 palavras mais negativas (No)
print("\n── Palavras mais negativas ──")
print(df_pesos.nsmallest(20, "peso")[["palavra", "peso"]].to_string())


# Palavras positivas
pesos_positivos = df_pesos[df_pesos["peso"] > 0].set_index("palavra")["peso"].to_dict()

# Palavras negativas (valor absoluto para o tamanho)
pesos_negativos = df_pesos[df_pesos["peso"] < 0].set_index("palavra")["peso"].abs().to_dict()

# Nuvem positiva
wc_positivo = WordCloud(width=800, height=400, background_color="white", colormap="Greens")
wc_positivo.generate_from_frequencies(pesos_positivos)

# Nuvem negativa
wc_negativo = WordCloud(width=800, height=400, background_color="white", colormap="Reds")
wc_negativo.generate_from_frequencies(pesos_negativos)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(wc_positivo, interpolation="bilinear")
axes[0].set_title("Palavras Positivas (Recomenda)", fontsize=14)
axes[0].axis("off")

axes[1].imshow(wc_negativo, interpolation="bilinear")
axes[1].set_title("Palavras Negativas (Não Recomenda)", fontsize=14)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("nuvem_palavras.png", dpi=150)
plt.show()