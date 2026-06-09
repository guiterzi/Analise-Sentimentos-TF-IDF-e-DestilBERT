import numpy as np
import pandas as pd

# Carregar
y_pred_tfidf = np.load("y_pred_tfidf.npy")
y_pred_bert  = np.load("y_pred_distilbert_128.npy")
y_teste      = np.load("y_teste_tfidf.npy")
X_teste      = np.load("X_teste_tfidf.npy", allow_pickle=True)

# Comparação
df_erros = pd.DataFrame({"texto":X_teste,"real":y_teste,"pred_tfidf": y_pred_tfidf,"pred_bert":  y_pred_bert})

# TF-IDF errou, BERT acertou
tfidf_errou = df_erros[(df_erros["pred_tfidf"] != df_erros["real"]) &(df_erros["pred_bert"]  == df_erros["real"])]

# BERT errou, TF-IDF acertou
bert_errou = df_erros[(df_erros["pred_bert"]  != df_erros["real"]) &(df_erros["pred_tfidf"] == df_erros["real"])]

print(f"TF-IDF errou, BERT acertou: {len(tfidf_errou)}")
print(f"BERT errou, TF-IDF acertou: {len(bert_errou)}")

print("\n── TF-IDF errou, BERT acertou ──")
print(tfidf_errou[["texto", "real", "pred_tfidf", "pred_bert"]].head(10).to_string())

print("\n── BERT errou, TF-IDF acertou ──")
print(bert_errou[["texto", "real", "pred_tfidf", "pred_bert"]].head(10).to_string())