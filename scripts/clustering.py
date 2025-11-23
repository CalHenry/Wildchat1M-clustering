import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    return KMeans, TfidfVectorizer, mo, np, pl, plt, silhouette_score, sns


@app.cell
def _(pl):
    # import data

    with open("data/processed/vocab.txt", "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()

    tokens_vars = "first_user_content_tokens"

    # ! this can be memory intensive (select the right variables before collecting) !
    # (there is a 'low_memory' option on .scan_parquet())
    df_conv = (
        pl.scan_parquet("data/clean/df_conv.parquet")
        .select(pl.col(tokens_vars))
        .collect(engine="streaming")
    )
    return df_conv, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TF_IDF
    """)
    return


@app.cell
def _(TfidfVectorizer, df_conv, pl, vocab):
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,  # dummy tokenizer
        vocabulary=vocab,
        # we use no preprocessing steps from the sklearn tfidf. All done earlier.
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        max_features=None,
        min_df=1,
        max_df=1.0,
    )

    tfidf_matrix = vectorizer.fit_transform(
        df_conv.select(pl.col("first_user_content_tokens")).to_series().to_list()
    )
    return tfidf_matrix, vectorizer


@app.cell
def _(np, tfidf_matrix, vectorizer):
    feature_names = vectorizer.get_feature_names_out()

    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    return feature_names, mean_tfidf


@app.cell
def _(feature_names, mean_tfidf, pl, plt, sns):
    _tfidfviz = pl.DataFrame({"word": feature_names, "tfidf_score": mean_tfidf})

    # Get top N words
    top_n = 20
    top_words = _tfidfviz.sort("tfidf_score", descending=True).head(top_n)

    # Create the barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_words, y="tfidf_score", x="word", hue="word")

    plt.title(f"Top {top_n} Words by TF-IDF Score", fontsize=16, fontweight="bold")
    plt.xlabel("Mean TF-IDF Score", fontsize=12)
    plt.tick_params(axis="x", labelrotation=45)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustering


    ###Kmeans
    """)
    return


@app.cell
def _(KMeans, tfidf_matrix):
    # find number of clusters

    inertia_values = []
    k_range = range(1, 11)


    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(tfidf_matrix)
        inertia_values.append(kmeans.inertia_)
    return inertia_values, k_range


@app.cell
def _(inertia_values, k_range, plt):
    plt.figure(figsize=(10, 7))
    plt.plot(k_range, inertia_values, 'bo-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Inertie')
    plt.title('Méthode du Coude pour le Choix de K')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(KMeans, np, plt, silhouette_score, tfidf_matrix):
    silhouette_values = []
    for _k in range(2, 11): # Le score de silhouette n'est défini que pour K >= 2
        _kmeans = KMeans(n_clusters=_k, init='k-means++', n_init=10, random_state=42)
        labels = _kmeans.fit_predict(tfidf_matrix)
        silhouette_values.append(silhouette_score(tfidf_matrix, labels))

    # Tracer la courbe du score de silhouette
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 11), silhouette_values, 'ro-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Score de Silhouette Moyen')
    plt.title('Score de Silhouette pour le Choix de K')
    plt.xticks(range(2, 11))
    plt.grid(True)
    plt.show()
    best_k_silhouette = np.argmax(silhouette_values) + 2 # +2 car on commence à k=2
    return


if __name__ == "__main__":
    app.run()
