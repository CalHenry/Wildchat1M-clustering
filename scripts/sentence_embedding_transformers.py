import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import numpy as np
    from scipy import sparse
    import polars as pl
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans, HDBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer

    import os
    import json
    from clustering import (
        sample,
        run_kmeans,
        get_tsne_2D_coords,
        nbr_obs_clusters,
        top_tfidf_terms,
        most_overrepresented_terms,
        plot_clusters,
        plot_clusters_grid,
        umap_tfidf_matrix,
    )
    import matplotlib.pyplot as plt

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # to avoid deadlocks when encoding


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set up

    Transformers models are trained on complete piece of text, including stop words, punctuation ...
    We should not clean the data like we had to with tfidf in order to get the best out of the model.
    """)
    return


@app.cell
def _():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # faster, smaller model, no differences in performance compare to the best ranked model: all-mpnet-base-v2
    return (model,)


@app.cell
def _():
    # import data
    lf_conv = pl.scan_parquet("data/clean/df_conv_raw.parquet").select(
        pl.col("first_user_content")
    )
    # df_conv_raw contains our data with no cleanings, only filtering the english content.

    df_conv = sample(df=lf_conv, fraction=0.1, seed=42).collect(engine="streaming")
    # df_conv = lf_conv.collect(engine="streaming"
    return (df_conv,)


@app.cell
def _(df_conv):
    df_conv.select(pl.col("first_user_content")).to_series().drop_nulls().to_list()
    return


@app.cell
def _(df_conv):
    # get vocab
    with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        lowercase=True,
    )


    tfidf_matrix = vectorizer.fit_transform(
        df_conv.select(pl.col("first_user_content"))
        .to_series()
        .drop_nulls()
        .to_list()
    )
    return tfidf_matrix, vectorizer


@app.cell
def _():
    # tfidf_matrix = sparse.load_npz("data/processed/tfidf_matrix.npz")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Embeddings
    """)
    return


@app.cell
def _(df_conv, model):
    embeddings = model.encode(
        df_conv.select(pl.col("first_user_content"))
        .to_series()
        .drop_nulls()
        .to_list(),
        show_progress_bar=True,
    )
    return (embeddings,)


@app.cell
def _(embeddings):
    tsne_2d = get_tsne_2D_coords(embeddings, use_svd=False)
    return (tsne_2d,)


@app.cell
def _(df_conv):
    df_conv.schema
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## HDBSCAN

    We will run the same HDBSCAN model as before.
    We will only look for the best value of *min_cluster_size* and no other hyperparameters.
    There is a breakpoint for the value of *min_cluster_size*. Above, the number of clusters and the percentage of noise drops very low. Actually creating 1 gigantic cluster that contains all the noise, similar to kmeans logic that forces each point into a cluster.
    Instead we prefer to let hdbscan express itself on the data, mark noise as noise and find many dense areas in the data, which is closer to our goal of exploring/ discovering what is in our data and closer to the reality of chatgpt conversations.
    """)
    return


@app.cell
def _(embeddings):
    X_umap = umap_tfidf_matrix(embeddings)
    return (X_umap,)


@app.cell
def _(X_umap):
    sizes = range(60, 70, 1)  # best is <66

    n_clusters = []
    noise_pct = []

    for size in sizes:
        clusterer = HDBSCAN(
            min_cluster_size=sizes,
            copy=True,
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(X_umap)
        n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
        noise_pct.append(100 * np.sum(labels == -1) / len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sizes, n_clusters, marker="o")
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("Number of clusters")

    ax2.plot(sizes, noise_pct, marker="o", color="orange")
    ax2.set_xlabel("min_cluster_size")
    ax2.set_ylabel("Noise %")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_umap):
    hdb = HDBSCAN(min_cluster_size=65, copy=True, n_jobs=-1)
    hdb.fit(X_umap)
    print(
        f"nbr of clusters: {len(np.unique(hdb.labels_)) - 1}"
    )  # -1 to remove the noise being the cluster -1
    return (hdb,)


@app.cell
def _(hdb, tsne_2d):
    plot_clusters(hdb, tsne_2d, hide_noise=False)
    plot_clusters(hdb, tsne_2d, hide_noise=True)
    plot_clusters_grid(hdb, tsne_2d, hide_noise=False, n_clusters_to_display=18)
    return


@app.cell
def _():
    mo.md(r"""
    ## Conclusion on the Transformers models:
    - Same clustering results as with TF-IDF, more clusters, over
    -  well defined small clusters and a huge noise cluster (58% of the obs)
    - Clusters are better represented in 2d TSNE reduction
    - Noise spreads on the entire area of the 2d plot, execpt for the part where the cluster 6 is located and very specific areas on the edges
    - Embedding model is fast and efficient (~2 minutes for the sample of 10% of the data, thanks to GPU acceleration with torch)
    """)
    return


@app.cell
def _(df_conv, hdb, tfidf_matrix, vectorizer):
    # Clusters exploration
    print(f"nbr of clusters: {len(np.unique(hdb.labels_)) - 1}")
    print("-" * 30)

    nbr_obs_clusters(hdb, size_df=df_conv.height)
    print("-" * 30)

    top_tfidf_terms(hdb, tfidf_matrix, vectorizer, nbr_terms=10)
    print("-" * 30)

    most_overrepresented_terms(hdb, tfidf_matrix, vectorizer, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion on the HDBSCAN clusters

    - 65 clusters
    - 58% of noise
    - 56 clusters < 1% of the total obs count
    - 5 clusters between 1% and 2% (26, 28, 55, 59, 62)
    - 3 clusters between 2% and 5% (25, 39, 41)
    - 1 cluster > 5% (cluster 6)

    Many very small clusters (<1% represents < 490 conversations)

    This result is at the same time similar and very different from the HDBSCAN clusters with TF-IDF. Let's do a compartive analysis of the cluster size, the top tfidf terms and most overrepresented terms for both HDBSCAN runs (TF-IDF vs Transformer embeddings).

    ## How does the clusters compare the the HDBSCAN's clusters with TF-IDF as the input ?

    - 2.3x the number of clusters (28 to 65)
    - More noise (43% to 58%)
    - Lowered the value of *min_cluster_size* (340 to 65)


    <details>

    <summary>AI to the rescue</summary>

    To compare clusters from both approaches, we have to compare side by side the outputs of the exploration functions that deliver the real insights about the clusters. For the model above, the output of the cell is more than 1600 lines long. <br>
    To make this task easier, i gave both outputs to [Claude](https://claude.ai) and asked him to report the major differences between the 2 clusters sets (the other output is delivered by the *clustering.py* notebook).   <br>
    Most of the content below comes or from Claude's response.

    </details>


    Analysis: The transformer's semantic understanding splits conversations into more nuanced groups, while TF-IDF groups by keyword similarity, creating broader categories.



    #### **Clusters That Match Well:**

    | Theme | TF-IDF | Transformer | Notes |
    |-------|--------|-------------|-------|
    | **Image prompts (Midjourney-style)** | Clusters 0 & 1 (17.6% combined) | Clusters 0,1,5,6,8,9,12-21 (~10% combined) | TF-IDF merges into 2 large clusters; Transformer splits by subtle prompt variations |
    | **Doki Doki Literature Club** | Cluster 5 (4.7%) | Cluster 25 (4.1%) | Strong match - character names dominate both |
    | **Message rewriting** | Cluster 4 (1.6%) | Cluster 26 (1.6%) | Near-perfect alignment |
    | **Fish farting stories** | Cluster 6 (1.1%) | Cluster 28 (1.1%) | Identical niche topic |
    | **Sports scripts** | Cluster 7 (1.3%) | Cluster 10 (0.6%) | Similar but transformer splits smaller |
    | **Jane/Aaron characters** | Cluster 3 (0.8%) | Cluster 23 (0.8%) | Character-based clustering matches |
    | **Naruto/Freedom Planet** | Cluster 8 (1.8%) | Clusters 44-45 (0.9% combined) | Transformer separates fandoms more |
    | **Code generation** | Cluster 21 (3.2%) | Clusters 59-60,63-64 (~3% combined) | TF-IDF broader; Transformer splits by language |
    | **Email writing** | Cluster 23 (1.1%) | Cluster 62 (1.2%) | Good match |
    | **Story writing** | Cluster 24 (1.3%) | Cluster 41 (2.5%) | Semantic approach captures more |
    | **Chinese/Imperial history** | Cluster 25 (0.8%) | Cluster 49 (0.3%) | TF-IDF groups by keywords; Transformer more specific |

    Some clusters appear identical or very close between the 2 approaches while others are usually smaller with Transformers embedding. Since we have ~15% more noise than before, we can expect those points to be classified as noise

    ---

    Image Generation Prompts (TF-IDF Clusters 0 & 1)

    TF-IDF: Lumps nearly all image prompts into 2 clusters based on keywords like "ar", "prompt", "description"
    Transformer: Splits into 15+ micro-clusters by semantic themes:

    - Cluster 0: Gundam/mecha imagery
    - Cluster 1: Christmas/Santa themes
    - Cluster 12: Chinese Qipao fashion
    - Cluster 18: Photography styles
    - Cluster 19: Anime (ufotable) style
    - And many more specialized prompt types



    Verdict: Transformers reveals the semantic diversity hidden within keyword-similar prompts.


    #### **Transformer finds more than TF-IDF:**

    1. **Cluster 2 (Transformer)**: "hello/there" greetings (0.3%)
       - TF-IDF buries this in general clusters
       - Semantic model recognizes conversational opening patterns

    2. **Cluster 30 (Transformer)**: Meta-questions about GPT itself (0.4%)
       - "what version are you", "chat gpt capabilities"
       - TF-IDF can't distinguish these from general questions

    3. **Cluster 31 (Transformer)**: Jailbreak attempts (DAN prompts) (0.3%)
       - Overrepresented terms: "nsfwgpt", "evilbot", "narotica"
       - Critical for safety analysis - invisible to TF-IDF

    4. **Cluster 22 (Transformer)**: Hypnosis roleplay (0.9%)
       - "nadine", "hypnotism", "footrubs", "submissively"
       - TF-IDF merges with other creative writing

    5. **Cluster 56 (Transformer)**: TikTok content curation (0.1%)
       - "greenzen", "vitalityboost", "tiktoks", "minimalist"
       - Too contextually specific for keyword matching

    Examples of more detailled clusters from Transformers with code/ technical content:

    **TF-IDF Approach:**
    - Cluster 19: Code snippets (1.0%)
    - Cluster 20: Excel/data manipulation (1.1%)
    - Cluster 21: General programming (3.2%)

    **Transformer Approach:**
    - Cluster 58: Excel/VBA specifically (0.3%)
    - Cluster 59: Frontend (React/JavaScript) (1.4%)
    - Cluster 60: C++/algorithms (0.7%)
    - Cluster 63: PyTorch/ML (0.4%)
    - Cluster 64: Python general (0.4%)

    **Analysis**: Transformer understands programming **contexts and purposes**, splitting by domain (web dev vs systems vs ML). TF-IDF only sees syntax patterns.

    **The "Hello" Problem**
    Neither approach handles simple greetings elegantly:

    TF-IDF Cluster 2: Just "hello" (0.8%)
    Transformer Cluster 2: "hello/there" (0.3%)
    Both are noise that indicates users testing the system. Could be filtered pre-clustering.


    **In conclusion**, Transformers is capable or revealing semantic structure that TF-IDF can't see.
    Both approach are valid, **TF-IDF** is better to categorize (less clusters, less specific) and **Transformers** is better to discovery (detailled, catches nuances)
    """)
    return


if __name__ == "__main__":
    app.run()
