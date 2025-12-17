import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")

with app.setup:
    import json
    import random
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import sparse
    import polars as pl
    import seaborn as sns
    from sklearn.cluster import HDBSCAN, MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    from umap import UMAP


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    This notebook is organized into 3 columns:
    - **1:** TF-IDF and helper functions
    - **2:** K-means clustering and K-means-specific code
    - **3:** HDBSCAN clustering and HDBSCAN-specific code

    {mo.outline()}

    ## Objective

    Currently, our data is a black box. We can only examine individual data points and basic statistics. Clustering will provide both a global overview of the dataset and a detailed understanding of specific cluster characteristics.

    #### Data Overview

    We are clustering ChatGPT conversations using only the first message from each conversation.

    **Memory constraints:** Algorithms like HDBSCAN, UMAP, t-SNE, and clustering evaluation metrics require more memory than our 16GB RAM can provide. However, we don't need the full dataset to extract meaningful insights. We'll work with a 50,000-observation subset, representing approximately 10% of the total data (50,000 out of 475,000 conversations).

    We expect to discover distinct prompt types, themes, and vocabulary patterns that form meaningful clusters.

    #### Technical Approach

    Our conversations have been cleaned and transformed into a TF-IDF matrix—a high-dimensional sparse representation where data points are widely separated. This makes direct clustering difficult.

    To address this, we'll reduce dimensionality using manifold learning algorithms:
    - **TruncatedSVD**
    - **UMAP**

    We'll apply two complementary clustering algorithms:
    - **K-means:** Best for spherical, evenly-sized clusters. Excels at identifying well-defined groupings but requires specifying the number of clusters upfront.
    - **HDBSCAN:** A density-based algorithm better suited for **CONTINUE**
    """)
    return


@app.cell
def _():
    # import data

    with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # ! this can be memory intensive (select the right variables before collecting)
    # (there is a 'low_memory' option on .scan_parquet())
    """
    df_conv = (
        pl.scan_parquet("data/clean/df_conv.parquet")
        .select(pl.col("first_user_content", "first_user_content_tokens"))
        .head(50000)
        .collect(engine="streaming")
    )
    """
    lf_conv = pl.scan_parquet("data/clean/df_conv.parquet").select(
        pl.col("first_user_content", "first_user_content_tokens")
    )

    df_conv = sample(df=lf_conv, fraction=0.1, seed=42).collect(engine="streaming")
    # df_conv = lf_conv.collect(engine="streaming")

    df_size = df_conv.height
    return df_conv, lf_conv, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. TF-IDF
    """)
    return


@app.cell
def _(df_conv, vocab):
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,  # dummy tokenizer
        vocabulary=vocab,
        # no preprocessing steps from sklearn's tfidf. All done earlier and manually with polars.
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        max_features=None,
    )

    # matrix_path = Path("data/processed/tfidf_matrix.npz")
    matrix_path = Path("data/processed/tfidf_matrix_sample.npz")
    if matrix_path.exists():
        print(f"Load TF-IDF matrix from {matrix_path}")
        tfidf_matrix = sparse.load_npz(matrix_path)
    else:
        print("Compute TF-IDF matrix")
        tfidf_matrix = vectorizer.fit_transform(
            df_conv.select(pl.col("first_user_content_tokens"))
            .to_series()
            .to_list()
        )
    return matrix_path, tfidf_matrix, vectorizer


@app.cell
def _(matrix_path, tfidf_matrix):
    sparse.save_npz(matrix_path, tfidf_matrix)
    return


@app.cell
def _(tfidf_matrix, vectorizer):
    feature_names = vectorizer.get_feature_names_out()

    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    return feature_names, mean_tfidf


@app.cell
def _(feature_names, mean_tfidf):
    # Top words tfidf

    _tfidfviz = pl.DataFrame({"word": feature_names, "tfidf_score": mean_tfidf})

    # Get top N words
    top_n = 20
    top_words = _tfidfviz.sort("tfidf_score", descending=True).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_words, y="tfidf_score", x="word", hue="word")

    plt.title(f"Top {top_n} Words by TF-IDF Score", fontsize=16, fontweight="bold")
    plt.xlabel("Mean TF-IDF Score", fontsize=12)
    plt.tick_params(axis="x", labelrotation=45)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(tfidf_matrix):
    tsne_2d = get_tsne_2D_coords(tfidf_matrix)
    return (tsne_2d,)


@app.cell
def _(tfidf_matrix):
    umap_2d = get_umap_2D_coords(tfidf_matrix)
    return (umap_2d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Helpers
    ---
    """)
    return


@app.function
# https://github.com/Yixuan-Wang/blog-contents/issues/29 (source code of the sample() function)
def sample(
    df: pl.LazyFrame,
    *,
    size: int | None = None,
    fraction: float | None = None,
    seed: int | None = None,
    row_index_name: str = "row_index",
):
    if not ((size is None) ^ (fraction is None)):
        raise ValueError(
            "One and only one of `size` or `fraction` must be specified."
        )

    # First let's scan the whole LazyFrame to see how many data we have
    # If you are sure that the LazyFrame is homogenous,
    # you can probably just sample from the head
    height: int = df.select(pl.len()).collect().item()
    if size is None:
        size = int(height * fraction)  # type: ignore

    # Use numpy to create a sample of the indices
    samples = set(
        np.random.default_rng(seed).choice(height, size=size, replace=False)
    )

    return (
        df.with_row_index(row_index_name)
        .filter(pl.col(row_index_name).is_in(samples))
        # Let's short-circuit when we already have the right size
        # this makes the best case time to be O(m^2)
        .head(size)
        .drop(row_index_name)
    )


@app.function
def get_tsne_2D_coords(
    embeddings, perplexity: int = 30, use_svd: bool = True
) -> dict:
    """
    run TSNE with 2 components after a TruncatedSVD to reduce computation time
    (input is a tfidf matrix which is sparse)

    ! TSNE can be long and expensive compute
    args:
        embeddings: either TF-IDF sparse matrix or sentence transformer embeddings (dense numpy array)
    """
    if use_svd:
        svd = TruncatedSVD(n_components=50, random_state=42)
        embeddings_reduced = svd.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # tsne
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_coords = tsne.fit_transform(embeddings_reduced)

    tsne_result = {
        "name": tsne.__class__.__name__,
        "coords": tsne_coords,
    }
    return tsne_result


@app.function
def get_umap_2D_coords(
    tfidf_matrix,
    umap_metric: str = "cosine",
    umap_n_neighbors: int = 200,
) -> dict:
    """
    run UMAP with 2 components after a TruncatedSVD to reduce computation time
    (input is a tfidf matrix which is sparse)
    """
    svd = TruncatedSVD(n_components=50, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    umap = UMAP(
        n_components=2, metric=umap_metric, n_neighbors=umap_n_neighbors
    )
    umap_coords = umap.fit_transform(tfidf_reduced)
    umap_result = {
        "name": umap.__class__.__name__,
        "coords": umap_coords,
    }
    return umap_result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Cluster exploration functions
    """)
    return


@app.function
# explore clusters
def nbr_obs_clusters(
    clustering_model, size_df: int, clusters: int | list[int] | str = "all"
):
    """
    print the nbr of obs per cluster
    args:
        clustering_model: A fitted clustering model that has a labels_ attribute
        size_df: should be = df.height (in polars)
        clusters: select clusters by index (first cluster is index 0)
    """
    all_clusters = clustering_model.labels_
    nbr_clusters = len(np.unique(all_clusters)) - (
        1 if -1 in all_clusters else 0
    )

    if isinstance(clusters, int):  # int to list[int] for consistency
        clusters = [clusters]
    # Determine which clusters to process
    if clusters == "all":
        clusters_to_process = range(nbr_clusters)
    elif isinstance(clusters, list):
        for c in clusters:
            if c < 0 or c >= nbr_clusters:
                raise ValueError(
                    f"Cluster {c} is out of range. Valid range: 0-{nbr_clusters - 1}"
                )
        clusters_to_process = clusters
    else:
        raise TypeError(
            f"clusters must be 'all', int, or list[int], got {type(clusters)}"
        )

    for cluster_num in clusters_to_process:
        cluster_indices = np.where(all_clusters == cluster_num)[0]
        print(
            f"Cluster {cluster_num} has {len(cluster_indices)} conversations, {len(cluster_indices) / size_df:.1%} of total"
        )


@app.function
def top_tfidf_terms(
    clustering_model,
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    nbr_terms: int = 15,
    clusters: int | list[int] | str = "all",
):
    """
    Print top n terms with the higest TFIDF score for each cluster

    args:
        clustering_model: A fitted clustering model that has a labels_ attribute
        tfidf_matrix: fit_transformed tfidf matrix
        vectorizer: fitted TfidfVectorizer instance
        clusters: select clusters by index (first cluster is index 0)
    """
    all_clusters = clustering_model.labels_
    nbr_clusters = len(np.unique(all_clusters)) - (
        1 if -1 in all_clusters else 0
    )
    feature_names = vectorizer.get_feature_names_out()

    if isinstance(clusters, int):  # int to list[int] for consistency
        clusters = [clusters]

    # Determine which clusters to process
    if clusters == "all":
        clusters_to_process = range(nbr_clusters)
    elif isinstance(clusters, list):
        for c in clusters:
            if c < 0 or c >= nbr_clusters:
                raise ValueError(
                    f"Cluster {c} is out of range. Valid range: 0-{nbr_clusters - 1}"
                )
        clusters_to_process = clusters
    else:
        raise TypeError(
            f"clusters must be 'all', int, or list[int], got {type(clusters)}"
        )

    print("Top TFIDF terms:")
    for cluster_num in clusters_to_process:
        cluster_indices = np.where(all_clusters == cluster_num)[0]

        # TF-IDF scores for the cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1

        # Get top n terms
        top_indices = cluster_tfidf.argsort()[-nbr_terms:][::-1]
        top_terms = [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

        print(f"\nCluster {cluster_num}:")
        for term, score in top_terms:
            print(f"  {term}: {score:.4f}")


@app.function
def sample_rdm_conversations(
    clustering_model,
    content_series: pl.Series,
    nbr_conv_to_print: int,
    clusters: int | list[int] | str = "all",
):
    """
    Print n random conversations for each cluster
    print first 500 characters of each message (! can print a lot to the console)

    args:
        clustering_model: A fitted clustering model that has a labels_ attribute
        content_series: variable that has the texts (ex: df_conv['first_user_content'])
        clusters: select clusters by index (first cluster is index 0)
    """
    all_clusters = clustering_model.labels_
    nbr_clusters = len(np.unique(all_clusters)) - (
        1 if -1 in all_clusters else 0
    )

    if isinstance(clusters, int):  # int to list[int] for consistency
        clusters = [clusters]
    # Determine which clusters to process
    if clusters == "all":
        clusters_to_process = range(nbr_clusters)
    elif isinstance(clusters, list):
        for c in clusters:
            if c < 0 or c >= nbr_clusters:
                raise ValueError(
                    f"Cluster {c} is out of range. Valid range: 0-{nbr_clusters - 1}"
                )
        clusters_to_process = clusters
    else:
        raise TypeError(
            f"clusters must be 'all', int, or list[int], got {type(clusters)}"
        )

    # get and print messages
    for cluster_num in clusters_to_process:
        cluster_indices = np.where(all_clusters == cluster_num)[0]

        # Sample 10 random conversations
        sample_size = min(nbr_conv_to_print, len(cluster_indices))
        sample_indices = random.sample(list(cluster_indices), sample_size)

        print(f"\n{'=' * 50}")
        print(f"CLUSTER {cluster_num} - Sample Conversations")
        print(f"{'=' * 50}")

        for idx in sample_indices:
            print(f"\n--- Conversation {idx} ---")
            print(content_series.to_series().to_list()[idx][:500])
            print("...")


@app.function
def most_overrepresented_terms(
    clustering_model,
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    nbr_terms: int = 15,
    clusters: int | list[int] | str = "all",
):
    """
    Print top n most overrepresented TFIDF terms for each cluster

    args:
        clustering_model: A fitted clustering model that has a labels_ attribute
        tfidf_matrix: fit_transformed tfidf matrix
        vectorizer: fitted TfidfVectorizer instance
        clusters: select clusters by index (first cluster is index 0)
    """

    overall_mean = tfidf_matrix.mean(axis=0).A1

    all_clusters = clustering_model.labels_
    nbr_clusters = len(np.unique(all_clusters)) - (
        1 if -1 in all_clusters else 0
    )
    feature_names = vectorizer.get_feature_names_out()

    if isinstance(clusters, int):  # int to list[int] for consistency
        clusters = [clusters]
    # Determine which clusters to process
    if clusters == "all":
        clusters_to_process = range(nbr_clusters)
    elif isinstance(clusters, list):
        for c in clusters:
            if c < 0 or c >= nbr_clusters:
                raise ValueError(
                    f"Cluster {c} is out of range. Valid range: 0-{nbr_clusters - 1}"
                )
        clusters_to_process = clusters
    else:
        raise TypeError(
            f"clusters must be 'all', int, or list[int], got {type(clusters)}"
        )

    print("Most Overrepresented Terms:")
    for cluster_num in clusters_to_process:
        cluster_indices = np.where(all_clusters == cluster_num)[0]
        cluster_mean = tfidf_matrix[cluster_indices].mean(axis=0).A1

        # Calculate ratio (overrepresentation)
        ratio = cluster_mean / (
            overall_mean + 1e-10
        )  # Add small value to avoid division by zero

        # Get most overrepresented terms
        top_ratio_indices = ratio.argsort()[-nbr_terms:][::-1]

        print(f"\nCluster {cluster_num}:")
        for idx in top_ratio_indices:
            print(f"  {feature_names[idx]}: {ratio[idx]:.2f}x more common")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Plot functions
    """)
    return


@app.function
# Big picture
def plot_clusters(clustering_model, manifold_result: dict, hide_noise: bool):
    """
    plot all the points, colored by cluster on a 2D representation
    (TSNE usually better to see clusters in 2D)

    args:
        clustering_model: A fitted clustering model that has a labels_ attribute
        manifold_result: dict of a name and the coords of a manifold algo
    """
    clusters = clustering_model.labels_
    manifold_name = manifold_result["name"]
    manifold_coords = manifold_result["coords"]

    plot_title = f"{manifold_name} Clusters Visualization"

    # check if noise cluster existe, if not do nothing
    has_noise = -1 in clusters
    if hide_noise and has_noise:
        no_noise_mask = clusters != -1
        coord_2d = manifold_coords[no_noise_mask]
        clusters = clusters[no_noise_mask]
        # plot_title = f"{manifold_name} Clusters Visualization (noise hidden)"
        plot_title += " (noise hidden)"
    else:
        coord_2d = manifold_coords

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        x=coord_2d[:, 0],
        y=coord_2d[:, 1],
        hue=clusters,
        palette="tab20",
        alpha=0.6,
        s=20,
        edgecolor="k",
        linewidth=0.5,
    )

    plt.title(
        label=plot_title,
        fontsize=16,
        fontweight="bold",
    )
    if len(set(clusters)) < 8:
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend_.remove()
    plt.xlabel(f"{manifold_name} 1")
    plt.ylabel(f"{manifold_name} 2")
    plt.tight_layout()
    plt.show()


@app.function
# detailled picture
def plot_clusters_grid(
    clustering_model,
    manifold_result: dict,
    hide_noise: bool,
    n_clusters_to_display: int = 20,
):
    """
    Plot each of the top n clusters in a grid of n*6. Sorted descending order of cluster size
    Args:
        clustering_model: A fitted clustering model with a `labels_` attribute
        manifold_result: Dict with keys "name" (str) and "coords" (2D array)
        hide_noise: If True, exclude points labeled as noise (-1)
        n_clusters_to_display: Number of top clusters to display (<= 20)
    """
    if n_clusters_to_display > 20:
        raise ValueError("n_clusters_to_display needs to be <= to 20")

    clusters = clustering_model.labels_
    # manifold_name = manifold_result["name"]
    manifold_coords = manifold_result["coords"]

    # Handle noise
    has_noise = -1 in clusters
    if hide_noise and has_noise:
        no_noise_mask = clusters != -1
        coord_2d = manifold_coords[no_noise_mask]
        clusters = clusters[no_noise_mask]
    else:
        coord_2d = manifold_coords

    # Count the number of points in each cluster
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    # Sort clusters by size (descending) and take top N
    sorted_indices = np.argsort(-cluster_counts)[:n_clusters_to_display]
    sorted_clusters = unique_clusters[sorted_indices]
    sorted_counts = cluster_counts[sorted_indices]
    n_clusters = len(sorted_clusters)

    # Global min and max for x and y
    global_x_min, global_x_max = coord_2d[:, 0].min(), coord_2d[:, 0].max()
    global_y_min, global_y_max = coord_2d[:, 1].min(), coord_2d[:, 1].max()

    # 6 max per row for clarity
    n_cols = min(n_clusters, 6)
    n_rows = int(np.ceil(n_clusters / n_cols))

    # grid
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_clusters:
            ax.axis("off")  # Hide unused subplots
            continue
        cluster = sorted_clusters[i]
        mask = clusters == cluster
        ax.scatter(
            coord_2d[mask, 0],
            coord_2d[mask, 1],
            color=sns.color_palette("tab20")[i],
            s=20,
            edgecolor="k",
            linewidth=0.5,
        )
        ax.set_title(f"Cluster {cluster} (Size: {sorted_counts[i]})")
        # Set same scale for all subplots
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)

    plt.tight_layout()
    plt.show()


@app.function
def compute_cosine_similarity(clusters_labels, tfidf_matrix):
    for cluster_id in np.unique(clusters_labels):
        # Get indices of points in this cluster
        cluster_mask = clusters_labels == cluster_id
        cluster_points = tfidf_matrix[cluster_mask]

        # Compute pairwise cosine similarities within cluster
        similarities = cosine_similarity(cluster_points)

        # Get upper triangle (exclude diagonal and duplicates)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]

        print(f"Cluster {cluster_id}:")
        print(f"  Size: {cluster_points.shape[0]}")
        print(f"  Mean cosine similarity: {upper_triangle.mean():.3f}")
        print(f"  Std cosine similarity: {upper_triangle.std():.3f}")
        print()


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## 2. Kmeans
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Determine the number of clusters for K-means

    K-means requires specifying the number of clusters beforehand. We can estimate the optimal value using metrics but we will use intertia (elbow method) and silhouette score. (Intertia is build in kmeans and silhouette score is slow but add no pressure to the RAM).

    - **Elbow method:** Examines inertia (within-cluster sum of squares) to identify where adding more clusters yields diminishing returns
    - **Silhouette score:** Measures how well-separated and compact the clusters are

    Our goal is to **minimize inertia** while **maximizing the silhouette score**.
    """)
    return


@app.function
def find_optimal_clusters_kmeans(
    tfidf_matrix, max_k: int, sample_size: int | None = None, random_state=42
) -> dict:
    """
    Find the optimal number of clusters using Elbow method and Silhouette score.

    args:
        max_k (int): Maximum number of clusters to evaluate
        sample_size (int): Number of samples to use for Silhouette score. if None use the whole sample
    """
    # Elbow method
    inertia_values = []
    for k in range(1, max_k):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state,
        )
        kmeans.fit(tfidf_matrix)
        inertia_values.append(kmeans.inertia_)

    # Silhouette score
    if sample_size is None:
        sample_size = tfidf_matrix.shape[0]
    sample_size = min(sample_size, tfidf_matrix.shape[0])

    silhouette_values = []
    for k in range(2, max_k):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(tfidf_matrix)
        indices = np.random.choice(
            tfidf_matrix.shape[0],
            min(sample_size, tfidf_matrix.shape[0]),
            replace=False,
        )
        silhouette_values.append(
            silhouette_score(
                tfidf_matrix[indices],
                labels[indices],
                random_state=random_state,
            )
        )
        print(f"Run {k} done")

    best_k_silhouette = (
        np.argmax(silhouette_values) + 2
    )  # +2 because k starts at 2
    print("")
    return {
        "inertia_values": inertia_values,
        "silhouette_values": silhouette_values,
        "best_k_silhouette": best_k_silhouette,
    }


@app.function
# find best K for kmeans: plot elbow and silhouette methods
def plot_silhouette_results(result: dict, max_k: int):
    """
    Plot the Elbow Method and Silhouette Score results.
    please to run find_optimal_clusters() first to get the dict input

    args:
        result: returned by `find_optimal_clusters()`
        max_k: number of clusters evaluated, has to be the same as the max_k of 'find_optimal_clusters_kmeans()'
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --- First subplot: Elbow Method ---
    k_range = range(1, max_k)
    ax[0].plot(k_range, result["inertia_values"], "bo-")
    ax[0].set_xlabel("Number of clusters (K)")
    ax[0].set_ylabel("Inertia")
    ax[0].set_title("Elbow Method")
    ax[0].set_xticks(k_range)
    ax[0].grid(True)

    # --- Second subplot: Silhouette Score ---
    silhouette_k_range = range(2, max_k)
    ax[1].plot(silhouette_k_range, result["silhouette_values"], "ro-")
    ax[1].set_xlabel("Number of clusters (K)")
    ax[1].set_ylabel("Mean Silhouette Score")
    ax[1].set_title("Silhouette Score")
    ax[1].set_xticks(silhouette_k_range)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


@app.cell
def _(tfidf_matrix):
    max_k = 20
    optimal_clusters_infos = find_optimal_clusters_kmeans(
        tfidf_matrix, max_k=max_k
    )
    plot_silhouette_results(optimal_clusters_infos, max_k=max_k)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Silhouettes score are low (< 0.2) which indicates a poor clustering quality.
    - Scores from 6 to 10 varies a lot between 0.10 and 0.16, which menas that kmeans merges clusters or not with no clear trend.
    - Best score for k=15, 2 hight values before droping to scores near 0. Another evidence of heavely overlapping clusters, passing 15 clusters, new clusters are probably arbitrary subset of already small and overlapping clusters. We can expect very small clusters.
    Nevertheless, we will look at the results for k=6 and k=15/. 2 pictures, one conservative, one with the 'better' score.
    Most of all, we will look into the clusters for k=15 and see if they hold meaning.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Run kmeans
    """)
    return


@app.function
def run_kmeans(n_clusters: int, embeddings, n_init: int = 10):
    """
    args:
        embeddings: either TF-IDF sparse matrix or sentence transformer embeddings (dense numpy array)
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42
    )
    kmeans.fit_predict(embeddings)
    return kmeans


@app.cell
def _(tfidf_matrix):
    kmeans_6 = run_kmeans(n_clusters=6, embeddings=tfidf_matrix)
    return (kmeans_6,)


@app.cell
def _(kmeans_6, umap_2d):
    plot_clusters(
        kmeans_6, umap_2d, hide_noise=False
    )  # umap shows global distance between clusters
    return


@app.cell
def _(kmeans_6, tsne_2d):
    # TSNE shows visual clusters but only local distance (for a given cluster) is relevant
    plot_clusters(kmeans_6, tsne_2d, hide_noise=True)
    # 3 major clusters, one very spread

    plot_clusters_grid(kmeans_6, tsne_2d, hide_noise=False)
    # Clusters 4 is very small but well defined
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both representations are distorded representation of the reality.
    **TSNE:**
    - distance between the clusters is meaningless (global structure)

    **UMAP:**
    - preserve local and global structure
    - create an artificial structure

    ⚠️ Both representations in 2D are distortions of the reality because they force a 2D mapping of a hight dimensional space (TFIDF is very hight dimension).

    Text data is hight dimensional by nature.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **K=6**

    - **1 huge cluster** spread across a huge area
    - **2 well defined clusters** in their area
    - **0ther clusters** are spread or very localized or small
    - **Cluster 1 has isolated points in the middle** with white space around them. Likely very far points forced into 2d space resulting in the white space (y=0). They belong to cluster 1 but can be because cluster 1 is the closest cluster.
    -> What kind of messages are those points ?
    ---
    """)
    return


@app.cell
def _(tfidf_matrix):
    kmeans_15 = run_kmeans(n_clusters=15, embeddings=tfidf_matrix)
    return (kmeans_15,)


@app.cell
def _(kmeans_15, umap_2d):
    plot_clusters(kmeans_15, umap_2d, hide_noise=True)
    plot_clusters_grid(kmeans_15, umap_2d, hide_noise=False)
    return


@app.cell
def _(kmeans_15, tsne_2d):
    plot_clusters(kmeans_15, tsne_2d, hide_noise=True)
    plot_clusters_grid(kmeans_15, tsne_2d, hide_noise=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **K=15**
    With more clusters, the pattern persists:
    - still 1 massive cluster (10) dominating ~50% of the points
    - 3 well defined clusters (2, 5, and 1)
    - previous large cluster has fragmented into multiple smaller clusters (6, 7, 14, etc.)
    - most remaining clusters are scattered across regions and overlap with the dominant cluster

    Adding more clusters has not improved the clustering quality:
    - same fundamental structure as with k=7
    - 1 enormous cluster capturing the bulk of data points remains problematic
    - only 3/15 clusters are cleanly separated and were already distinct at k=7
    - the boundary between the massive cluster and the poorly defined ones is ambiguous

    These observations align with what the silhouette score indicated:
    - low scores reflect significant cluster overlap
    - stable scores across different k values suggest weak clustering effectiveness, with only 3 clusters truly well-separated

    Now we can examine the clusters through their tfidf terms and sample messages to address 2 questions:
    - Are the well defined clusters distinctive in their tfidf terms and messages?
    - Do the poorly defined clusters differ meaningfully from each other and from the dominant cluster?
    """)
    return


@app.cell
def _(df_conv, kmeans_15, tsne_2d):
    # What are the few points in the middle of the TSNE representation ?

    # target_x, target_y = 25, 0
    target_x, target_y = 60, 0

    # Calculate distances to target point
    distances = np.sqrt(
        (tsne_2d["coords"][:, 0] - target_x) ** 2
        + (tsne_2d["coords"][:, 1] - target_y) ** 2
    )

    # Find closest point
    idx = np.argmin(distances)

    print(f"Closest point index: {idx}")
    print(
        f"Coordinates: ({tsne_2d['coords'][idx, 0]:.2f}, {tsne_2d['coords'][idx, 1]:.2f})"
    )
    print(f"Cluster label: {kmeans_15.labels_[idx]}")
    print(f"Distance from target: {distances[idx]:.2f}")
    print("\n" + "=" * 50)
    print("Text content:")
    print(
        df_conv.select(pl.col("first_user_content_tokens"))
        .to_series()
        .to_list()[idx]
    )
    print("=" * 50)

    # Check if it's really isolated
    orange_cluster = kmeans_15.labels_[idx]
    same_cluster_mask = kmeans_15.labels_ == orange_cluster
    nearby_mask = distances < 20  # points within distance 20

    same_cluster_nearby = np.sum(same_cluster_mask & nearby_mask)
    print(f"\nOther points from same cluster nearby: {same_cluster_nearby - 1}")
    return (same_cluster_nearby,)


@app.cell(hide_code=True)
def _(same_cluster_nearby):
    mo.md(rf"""
    The isolated point is not alone, we found {same_cluster_nearby - 1} points in the close area.
    Looking at the content of the text, we can see that it's a prompt to generate images with **Midjourney** a popular LLM for image generation.
    All the {same_cluster_nearby - 1} points are probably the same Midjourney prompt with potential variations but the same core.
    We should find the keywoard and such prompts if we dive into the content of the cluster 0.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Clusters deep dive
    """)
    return


@app.cell
def _(kmeans_15):
    # Clusters exploration to see if we can spot differences between them or if we can't differentiate them as the clusters suggest
    # Cluster 0 contains the Midjourney prompts

    print(
        f"nbr of clusters: {len(np.unique(kmeans_15.labels_))}"
    )  # -1 if hdbscan to exclude noise
    print("-" * 30)

    nbr_obs_clusters(kmeans_15)
    print("-" * 30)

    top_tfidf_terms(kmeans_15, 10)
    print("-" * 30)

    most_overrepresented_terms(kmeans_15, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Number of Observations per Cluster:
    - **Two massive clusters dominate**: Cluster 10 (37.0%) and Cluster 5 (28.5%) together contain 65.5% of all conversations
    - **One large cluster**: Cluster 2 (17.4%)
    - **Medium clusters**: Clusters 1, 6 (3-4% each)
    - **Small clusters**: 10 clusters under 2%, 4 under 1%

    ---

    #### Top terms and most overrepresented terms:

    **Cluster 10 (37.0% - 17,578 conversations) - Generic Coding/Technical Catch-All**
    - Extremely broad technical vocabulary: write, code, name, import, file, translate, list
    - Largest single cluster, absorbing general programming and technical requests
    - Overrepresented terms are random and specific (resistor, Zuora API, Android layouts), suggesting many disparate technical topics lumped together
    - The "everything else technical" bucket with no cohesive theme

    **Cluster 5 (28.5% - 13,516 conversations) - General Conversational Catch-All**
    - Ultra-generic everyday language: make, like, hello, use, would, good, write, want
    - Second-largest cluster capturing casual, open-ended conversations
    - Overrepresented terms are completely random (littlegpt, surfer, celestine), confirming this is a miscellaneous bucket
    - The "everything else non-technical" catch-all for conversations without distinctive features

    ---

    **Cluster 2 (17.4% - 8,272 conversations) - AI Image Generation Prompts**
    - Highly distinctive technical vocabulary for image generation
    - Keywords: prompt, ar (aspect ratio), detail, description, image, style, capture, imagine, lens
    - Well-defined cluster for Midjourney prompts
    - Overrepresented terms include camera equipment (Hasselblad), fashion brands (Balenciaga), dog breeds (corgi, spitz)

    ---

    **Cluster 1 (3.7% - 1,751 conversations) - Doki Doki Literature Club Fan Content**
    - Very specific visual novel fandom
    - Character names: Natsuki, Monika, Sayori, Yuri, MC
    - Keywords: clubroom, literature club, baby (pregnancy storylines common in fan fiction)
    - Highly cohesive niche community content

    **Cluster 6 (3.9% - 1,835 conversations) - Interactive Roleplay/Chat Stories**
    - Narrative content with conversational elements
    - Keywords: story, character, dialogue, chat, female, male
    - Overrepresented terms include RPG systems (rpggpt, adventuregpt, choose-your-own-adventure)
    - Interactive fiction and chatbot-style storytelling

    ---

    **Cluster 12 (2.0% - 938 conversations) - Web Development Code**
    - Frontend web programming focus
    - Keywords: script, const, div, px, function, let, html
    - JavaScript/HTML/CSS code writing requests
    - Specific technical cluster for web development

    **Cluster 0 (1.0% - 496 conversations) - Anime/Gaming Crossover Fan Fiction**
    - Crossover content between franchises
    - Characters: Naruto, Sonic, Goku, Dragon Ball references
    - Keywords: character, react
    - Overrepresented terms include specific franchises (Freedom Planet, Funkin, Xenoverse)

    **Cluster 7 (1.4% - 671 conversations) - Concise Communication/Meta Instructions**
    - Instructions about communication style
    - Keywords: message, raw, concise, response, short, send, discussion
    - Requests for brief, casual, or adjust its response style

    **Cluster 14 (1.4% - 661 conversations) - Text-Based Game/Adventure Narratives**
    - Interactive game-like storytelling
    - Keywords: player, look, see, mob, spider, snow
    - Minecraft or RPG-style second-person narratives
    - Gaming-focused interactive fiction

    **Cluster 9 (1.0% - 480 conversations) - Comedic Absurdist Fiction**
    - Highly specific humorous theme
    - Keywords: fish, fart, pond, comedic, butt, relax
    - Overrepresented terms include character names (Frodo, Zagreus, Kristoff)
    - Lowbrow comedy stories with flatulence/bathroom humor

    **Cluster 3 (0.6% - 303 conversations) - Etsy Product Listings**
    - E-commerce product descriptions
    - Keywords: etsy, whimsical, title, tshirt, mug, art
    - Commercial content creation for online marketplaces

    **Cluster 8 (0.7% - 342 conversations) - Specific Character-Based Teen Fiction**
    - Recurring characters: Jane, Sam, Aaron
    - Keywords: old, year, teen, animate, dialogue, girlfriend
    - Appears to be a specific fictional universe with consistent characters
    - Possibly fan fiction for an animated series

    **Cluster 4 (0.7% - 325 conversations) - SEO/Product Variation Content**
    - E-commerce optimization focused
    - Keywords: include, repeat, flower, string, SEOs, newline, br, variation, resale, keyword
    - Generating multiple product description variations
    - Likely for print-on-demand or dropshipping businesses

    **Cluster 11 (0.5% - 215 conversations) - Scatological Comedy Scripts**
    - Extremely specific bathroom humor theme
    - Keywords: woman, pants, poop, aunt, door, odor, relieve

    **Cluster 13 (0.3% - 121 conversations) - RPG Magic Duel System**
    - Fantasy game mechanics
    - Keywords: spell, stat, cast, duel, point, JSON, regress
    - Magic dueling game/scenario ?
    """)
    return


@app.cell
def _():
    # Explore random message from any clusters with this function
    # sample_rdm_conversations(kmeans_17, nbr_conv_to_print=5, clusters=[1, 2, 3])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_conv, kmeans_6, tfidf_matrix, vectorizer):
    nbr_obs_clusters(kmeans_6, size_df=df_conv.height)
    print("-" * 30)

    top_tfidf_terms(kmeans_6, tfidf_matrix, vectorizer, 3)
    print("-" * 30)

    most_overrepresented_terms(kmeans_6, tfidf_matrix, vectorizer, 3)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## 3. HDBCSCAN

    Workflow:

    TF-IDF → TruncatedSVD → UMAP → HDBSCAN

    0. TFIDF matix
    1. dimension reduction with **SVD**
    2. dimension reduction with **UMAP**
    3. **HDBSCAN**
    4. Plots
    5. Exploration of the clusters
    """)
    return


@app.function
# SVD first to deal with the sparse nature of the TFIDF matix, then UMAP on the SVD
# UMAP on TFIDF is very expensive (UMAP's complexity is O(n×d)), SVD takes care of most of the sparse of the matrix


def umap_tfidf_matrix(
    tfidf_matrix,
    umap_n_components: int = 15,
    umap_metric: str = "cosine",
    umap_n_neighbors: int = 200,
):
    """
    Reduce TFIDF matrix dimensions usuing UMAP
    Preprocessing step before hdbscan
    """
    svd_big = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd_big.fit_transform(tfidf_matrix)

    reducer = UMAP(
        n_components=umap_n_components,
        metric=umap_metric,
        n_neighbors=umap_n_neighbors,
    )
    X_umap = reducer.fit_transform(X_reduced)

    return X_umap


@app.cell
def _(tfidf_matrix):
    X_umap = umap_tfidf_matrix(tfidf_matrix)
    return (X_umap,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The minimum cluster size is the most important parameter to tune here.
    Keams produces a massive (or 2) cluster even with a higher cluster number.
    The bigger **min_cluster_size**, the fewer number of clusters.
    We want to inspect how the algorithm react to different values of the parameters. This is informative for the density of the clusters.

    **min_cluster_size** is the key parameter to adjust.
    Keams produces a massive (or 2) cluster even with a higher cluster number.
    **Increasing the minimum cluster size results in fewer overall clusters.**
    How the algorithm responds to different parameter values ? This provides valuable insight into the density and distribution of the clusters.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Determine the best value of *min_cluster_size*
    """)
    return


@app.cell
def _(X_umap):
    sizes = range(320, 360, 5)
    # sizes = range(340, 350, 1)  # run with this range to see the exact breakpoint

    n_clusters = []
    noise_pct = []

    for size in sizes:
        clusterer = HDBSCAN(min_cluster_size=size, n_jobs=-1)
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We see a very clear breakpoint at exactly 347 (reminder: we work with a sample of the full data).

    - **min_cluster_size** < 347: many small, dense clusters (more than kmeans)

    - **min_cluster_size** > 347: HDBSCAN can't form those clusters anymore and instead just finds a few massive blobs in the generic content area.
    This is very clear in the % of noise plot, we drop from ~43% to 5% of noise, meaning almost all the noise points are assigned to a cluster, proably a huge generic one like Kmeans did.

    We will set **min_cluster_size** < 347, since we want many small clusters with noise rather than a huge generic cluster.
    A cluster of size n = 347 would be ~ 0.7% of the total.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Run HDBSCAN
    """)
    return


@app.cell
def _(X_umap):
    hdb = HDBSCAN(min_cluster_size=340, n_jobs=-1)
    hdb.fit(X_umap)
    print(
        f"nbr of clusters: {len(np.unique(hdb.labels_)) - 1}"
    )  # -1 to remove the noise being the cluster -1
    return (hdb,)


@app.cell
def _(hdb, tsne_2d):
    plot_clusters(hdb, tsne_2d, hide_noise=True)  # try it with the noise
    plot_clusters_grid(hdb, tsne_2d, hide_noise=False, n_clusters_to_display=18)
    return


@app.cell
def _(hdb, umap_2d):
    plot_clusters(hdb, umap_2d, hide_noise=True)
    plot_clusters_grid(hdb, umap_2d, hide_noise=False, n_clusters_to_display=18)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Clusters deep dive
    """)
    return


@app.cell
def _(hdb):
    # Clusters exploration
    print(f"nbr of clusters: {len(np.unique(hdb.labels_)) - 1}")
    print("-" * 30)

    nbr_obs_clusters(hdb)
    print("-" * 30)

    top_tfidf_terms(hdb, 10)
    print("-" * 30)

    most_overrepresented_terms(hdb, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **HDBSCAN assigned ~43% (20,551 conversations) to noise**

    - **labeled clusters (27) only represent 57% of the data**
    - K-Means' two massive clusters (37% + 28.5% = 65.5%) likely correspond to **HDBSCAN's noise cluster**
    - HDBSCAN is being **much more conservative**. Only creating clusters for highly cohesive, distinctive content
    - Generic conversations K-Means forced into clusters are now properly labeled as "no clear pattern"

    ---

    #### Distribution Pattern (of the 57% that clustered):
    - **Largest cluster**: 11.8% of total data (20.7% of clustered data)
    - **Three medium clusters**: 4-6% of total (8-10% of clustered)
    - **Most clusters**: 1-3% of total (2-5% of clustered)
    - **Small specialized clusters**: 8 clusters under 1% of total

    ---

    #### Clusters by categories:

    **AI Image Generation (Clusters 0, 1) - 17.6% of total data**
    comparable to K-Means**:
    - **Cluster 0 (11.8%)**: General image prompts
    - **Cluster 1 (5.8%)**: Technical photography prompts

    **Interpretation**: Image generation prompts are extremely specific in our data with the Midjouney prompts.

    ---

    **Programming & Technical (Clusters 19, 20, 21) - 5.7% of total**
    K-Means' 37% mega-cluster broke down into:
    - **~3-6% with clear technical patterns** → HDBSCAN Clusters 19, 20, 21
    - **~31-34% generic technical content** → HDBSCAN noise cluster

    ---

    **Fan Fiction Content (Clusters 3, 5, 8, 14, 16) - 11.2% of total**
    Multiple small but **extremely cohesive** fandoms:
    - **Cluster 5 (4.7%)**: Doki Doki Literature Club - character names create strong signal
    - **Cluster 8 (1.8%)**: Naruto/Sonic crossovers
    - **Cluster 3 (0.8%)**: Jane/Aaron teen series
    - **Cluster 16 (1.5%)**: Text-based game narratives
    - **Cluster 14 (0.8%)**: Genshin Impact pregnancy stories

    **Interpretation**: Fan fiction with recurring character names clusters strongly. Generic creative writing → noise cluster. (K-Means Cluster 6 (3.9% "interactive roleplay") ?)

    ---

    **Creative Writing (Clusters 7, 13, 24) - 5.7% of total**
    Very specific niche themes:
    - **Cluster 7 (1.3%)**: Sports and university ?
    - **Cluster 13 (2.1%)**: TV shows related ?
    - **Cluster 24 (1.3%)**: Generic stories

    ---

    **Business & Academic (Clusters 23, 26) - 4.2% of total**
    - **Cluster 23 (1.1%)**: Formal emails (bank, payment, address)
    - **Cluster 26 (3.1%)**: Business analysis (SWOT, market, strategy)

    Distinctive formal vocabulary allows these to cluster despite conceptual diversity.

    ---

    **Specialized Micro-Clusters**
    **HDBSCAN's strength** - finding tiny but perfectly cohesive groups:

    - **Cluster 6 (1.1%)**: Flatulence comedy (Frodo, pond, fart) - *identical to K-Means*
    - **Cluster 4 (1.6%)**: Meta-instructions (concise, raw, message) - *identical to K-Means*
    - **Cluster 25 (0.8%)**: Alternative history with Chinese/Japanese terms - *no Kmeans equivalent*
    - **Cluster 18 (1.4%)**: Video game, rpg stories ?
    - **Cluster 22 (1.0%)**: Generic or random ?

    Extremely distinctive, repetitive vocabulary

    ---

    **Language & Learning (Clusters 2, 9, 11, 12, 15, 17) - 7.3% of total**
    Only the most formulaic requests clustered:
    - **Cluster 2 (0.8%)**: Mostly greetings keywords ?
    - **Cluster 9 (0.8%)**: Mostly "Explain" keywords ?
    - **Cluster 11 (1.0%)**: Mostly "Give me a list" keywords
    - **Cluster 12 (1.3%)**: Mostly Q&A keywords
    - **Cluster 15 (0.8%)**: Grammar correction
    - **Cluster 17 (1.1%)**: Mostly "Give me names" keywords ?

    Only stereotyped phrasings are in clusters on not labeled as noise.

    ---

    **"General" Clusters (Clusters 10, 27) - 4.0% of total**
    Even these are relatively small:
    - **Cluster 10 (0.9%)**: Generic, extremely overreprented words
    - **Cluster 27 (3.0%)**: Generic verbs/ fandom, fantasy characters ?

    Weak patterns.

    ---

    K-Means or HDBSCAN ?

    **K-Means Cluster 10 (37% - "Generic Coding/Technical")**
    HDBSCAN breakdown:
    - ~6% → Technical clusters (19, 20, 21)
    - ~31% → **Noise cluster**

    **K-Means Cluster 5 (28.5% - "General Conversational")**
    HDBSCAN breakdown:
    - ~4% → Various small clusters (greetings, lists, explanations)
    - ~24% → **Noise cluster**

    **What went to noise?**
    - Generic "help me write code" without distinctive patterns
    - Casual conversations without formulaic language
    - One-off creative requests
    - Mixed-topic conversations
    - Anything without strong, repetitive vocabulary signals

    ---

    #### Key Insights:

    1. **HDBSCAN is conservative** - only 57% of data in clusters
    2. **Validates strong patterns** - clusters that appear in both algorithms are genuine (image generation, DDLC, flatulence comedy)
    3. **K-Means weakness**: creates weak big clusters instead of noise
    4. **Finds true dense points**: cluster <2%
    5. **Noise = heterogeneity** - 43% of conversations are genuinely diverse form the "Generic" group
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After clustering, we learned a lot about our data:
    - identification of real strong clusters
    - discovery of extremely niche, precise messages
    - confirmation that many messages are generic/ a lot of very common and generic words makes the clustering weak. Expected for text data.
    - TF-IDF don't capture all the semantic meaning of the messages but we have are able to retrieve insteresting insights from the dataset

    This is done on a sample 10% of the data. Does those conclusions hold with the full dataset ?
    We can expect:
    - bigger small clusters (10x more data should return smaller clusters 10x times bigger, otherwise they should be labeled as noise)
    - more clusters overall
    - no new big clusters, or known medium/ small clusters upgrading to big clusters

    We can't run **HDBSCAN** on the entire dataset because it doesn't fit into my RAM, but **MiniBatchKmeans** works like a charm and we will be able to compare the results for the full dataset with the results of Kmeans and HDBSCAN on the subset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    """)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ### Full dataset - Kmeans
    """)
    return


@app.cell
def _(lf_conv):
    df_full = lf_conv.collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The full dataset is 475042 rows and it represent tp much data for HDBSCAN and for clusters metrics for kmeans.
    Thanksfully, MinibatchsKmeans run on the full dataset in ~23 seconds.
    The only way we have to determine the optimal number of clusters is to retreive the inertia value, computed by Kmeans.
    """)
    return


@app.cell
def _(tfidf_matrix):
    inertias = []
    K_range = range(10, 25, 1)

    for k in K_range:
        mbkmeans = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=10000, max_iter=100
        )
        mbkmeans.fit(tfidf_matrix)
        inertias.append(mbkmeans.inertia_)
        print(f"k={k}, inertia={mbkmeans.inertia_:.2f}")

    # Plot the elbow
    plt.plot(K_range, inertias, "bo-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Optimal number of clusters is 20.
    """)
    return


@app.cell
def _(tfidf_matrix):
    kmeans_full = run_kmeans(n_clusters=20, embeddings=tfidf_matrix)
    return (kmeans_full,)


@app.cell
def _(kmeans_full):
    print(
        f"nbr of clusters: {len(np.unique(kmeans_full.labels_))}"
    )  # -1 if hdbscan to exclude noise
    print("-" * 30)

    nbr_obs_clusters(kmeans_full)
    print("-" * 30)

    top_tfidf_terms(kmeans_full, 10)
    print("-" * 30)

    most_overrepresented_terms(kmeans_full, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The restults on the full dataset are consistent with the conclusions on the sample with Kmeans and HDBSCAN.
    We have a huge cluster that represent the noise (cluster 19 = 47,7% of the dataset).
    The other clusters can be regroup in the same categories/ topics than the clusters of the sample. Sometimes more clusters are found for a theme, this is normal since we have 5 more clusters.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
