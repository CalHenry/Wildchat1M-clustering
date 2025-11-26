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
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    import random

    import matplotlib.pyplot as plt
    import seaborn as sns
    return (
        MiniBatchKMeans,
        TSNE,
        TfidfVectorizer,
        TruncatedSVD,
        mo,
        normalize,
        np,
        pl,
        plt,
        random,
        silhouette_score,
        sns,
    )


@app.cell
def _(pl):
    # import data

    with open("data/processed/vocab.txt", "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()

    tokens_vars = "first_user_content_tokens"

    # ! this can be memory intensive (select the right variables before collecting) !
    # (there is a 'low_memory' option on .scan_parquet())
    df_conv = (
        pl.scan_parquet("data/clean/df_conv_fuc.parquet")
        # .select(pl.col("first_user_content", "first_user_content_tokens"))
        # .select(pl.col(tokens_vars))
        .collect(engine="streaming")
    )

    df_size = df_conv.height
    return df_conv, df_size, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TF-IDF
    """)
    return


@app.cell
def _(TfidfVectorizer, df_conv, normalize, pl, vocab):
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

    tfidf_normalized = normalize(tfidf_matrix, norm="l2", axis=1)
    return tfidf_matrix, tfidf_normalized, vectorizer


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
def _(MiniBatchKMeans, np, silhouette_score, tfidf_matrix):
    # find number of clusters
    # Elbow method
    inertia_values = []
    k_range = range(1, 11)


    for k in k_range:
        kmeans = MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=10, random_state=42
        )
        kmeans.fit(tfidf_matrix)
        inertia_values.append(kmeans.inertia_)

    # Silhouette score
    silhouette_values = []
    sample_size = min(50000, tfidf_matrix.shape[0])

    for _k in range(2, 10):
        _kmeans = MiniBatchKMeans(
            n_clusters=_k, init="k-means++", n_init=10, random_state=42
        )
        labels = _kmeans.fit_predict(tfidf_matrix)

        # Sample for silhouette score
        indices = np.random.choice(
            tfidf_matrix.shape[0], sample_size, replace=False
        )
        silhouette_values.append(
            silhouette_score(tfidf_matrix[indices], labels[indices])
        )
        print(f"run {_k} done")
    best_k_silhouette = np.argmax(silhouette_values) + 2  # +2 bc it started at k=2
    return inertia_values, k_range, silhouette_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Suggest k=2

    2 seemns on a lower side for cluster in chatgpt conversations.
    Let's see silhouette score with a sample of 50 000 points.
    """)
    return


@app.cell
def _(inertia_values, k_range, plt, silhouette_values):
    def _():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # --- First subplot: Elbow Method ---
        ax[0].plot(k_range, inertia_values, "bo-")
        ax[0].set_xlabel("Number of clusters (K)")  # Custom x-label
        ax[0].set_ylabel("Inertia")  # Custom y-label
        ax[0].set_title("Elbow Method")  # Custom title
        ax[0].set_xticks(k_range)  # Custom x-ticks
        ax[0].grid(True)  # Enable grid

        # --- Second subplot: Silhouette Score ---
        ax[1].plot(range(2, 10), silhouette_values, "ro-")
        ax[1].set_xlabel("Number of clusters (K)")  # Custom x-label
        ax[1].set_ylabel("Mean Silhouette Score")  # Custom y-label
        ax[1].set_title("Silhouette Score")  # Custom title
        ax[1].set_xticks(range(2, 11))  # Custom x-ticks
        ax[1].grid(True)  # Enable grid

        # Adjust layout
        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    todo
    """)
    return


@app.cell
def _(MiniBatchKMeans, tfidf_normalized):
    # Kmean clustering
    best_k = 5
    final_kmeans = MiniBatchKMeans(
        n_clusters=best_k, init="k-means++", n_init=10, random_state=42
    )
    clusters = final_kmeans.fit_predict(tfidf_normalized)
    centroids = final_kmeans.cluster_centers_
    return best_k, clusters


@app.cell
def _(TSNE, TruncatedSVD, tfidf_matrix):
    svd = TruncatedSVD(n_components=2, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # tsne
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_2d = tsne.fit_transform(tfidf_reduced)
    # 23m runtime
    return (coords_2d,)


@app.cell
def _(clusters, coords_2d, plt, sns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        hue=clusters,
        palette="tab10",
        alpha=0.6,
        s=20,
        edgecolor="k",
        linewidth=0.5,
    )

    plt.title(
        f"K-Means Clusters Visualization (TruncatedSVD)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(clusters, coords_2d, np, plt, sns):
    def _():
        n_clusters = len(np.unique(clusters))

        # Calculate global min and max for x and y
        global_x_min, global_x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        global_y_min, global_y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        # Create a grid of subplots
        fig, axes = plt.subplots(
            nrows=1, ncols=n_clusters, figsize=(5 * n_clusters, 5)
        )

        # If there's only one cluster, axes will not be an array, so we make it one for consistency
        if n_clusters == 1:
            axes = [axes]

        # Plot each cluster in its own subplot
        for i, ax in enumerate(axes):
            cluster = i
            mask = clusters == cluster
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                color=sns.color_palette("tab10")[i],
                alpha=0.6,
                s=20,
                edgecolor="k",
                linewidth=0.5,
            )
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            # Set the same scale for all subplots
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(global_y_min, global_y_max)

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(best_k, clusters, df_size, np):
    for _cluster_num in range(best_k):
        _cluster_indices = np.where(clusters == _cluster_num)[0]
        print(
            f"Cluster {_cluster_num} has {len(_cluster_indices)} conversations, {len(_cluster_indices) / df_size:.1%} of total"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What do the plots tell us ?
    - **Cluster 0** is the bigest, takes most of the space. It contains the majority of conversations. It likely captures common conversations and don't represent a particular group.

    - **Clusters 1** and **3** are well defined which means 2 grops with different vocabulary/ semantic coherence.

    - **Clusters 2** is spread on 2 regions and shares it's sparse part with cluster 4. This can be a mixed topic cluster.

    - **Cluster 3** is small, compact and isolated at the top. It's likely a specific or very consistent prompt.

    - **CLuster 4** is moderate size and well defined. It should be another distinct kind of prompt.

    The clusters are very imbalanced with cluster 0 being 76% of the conversations.
    Nevertheless, TFIDF is working, some clusters are well defined.
    But we see it's limitations. Semantic similarity is not well captured. Cluster 2 and 4 being inside cluster 0 area is an example

    - The single point at the center of cluster 1 is likely an effect of TSNE (y=0). It can't be positioned correctly in this 2D plot. The white area shows that it is distant from the other points of the cluster. It can also be an outlier.

    Below we investigate the content of the conversations by clusters to define them better.

    We review per clusters:
    - top words with tfidf scores
    - random conversations
    - more represented words
    - content of the point alone inside of cluster 1
    """)
    return


@app.cell
def _(best_k, clusters, feature_names, np, tfidf_matrix):
    def _():
        for _cluster_num in range(best_k):
            cluster_indices = np.where(clusters == _cluster_num)[0]

            # TF-IDF scores for the cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1

            # Get top 15 terms
            top_indices = cluster_tfidf.argsort()[-15:][::-1]
            top_terms = [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

            print(f"\nCluster {_cluster_num}:")
            for term, score in top_terms:
                print(f"  {term}: {score:.4f}")


    _()
    return


@app.cell
def _(best_k, clusters, df_conv, np, pl, random):
    def _():
        for cluster_num in range(best_k):
            cluster_indices = np.where(clusters == cluster_num)[0]

            # Sample 10 random conversations
            sample_size = min(10, len(cluster_indices))
            sample_indices = random.sample(list(cluster_indices), sample_size)

            print(f"\n{'=' * 50}")
            print(f"CLUSTER {cluster_num} - Sample Conversations")
            print(f"{'=' * 50}")

            for idx in sample_indices:
                print(f"\n--- Conversation {idx} ---")
                print(
                    df_conv.select(pl.col("first_user_content"))
                    .to_series()
                    .to_list()[idx][:500]
                )
        return print("...")


    _()
    return


@app.cell
def _(best_k, clusters, feature_names, np, tfidf_matrix):
    overall_mean = tfidf_matrix.mean(axis=0).A1

    for cluster_num in range(best_k):
        cluster_indices = np.where(clusters == cluster_num)[0]
        cluster_mean = tfidf_matrix[cluster_indices].mean(axis=0).A1

        # Calculate ratio (overrepresentation)
        ratio = cluster_mean / (
            overall_mean + 1e-10
        )  # Add small value to avoid division by zero

        # Get most overrepresented terms
        top_ratio_indices = ratio.argsort()[-20:][::-1]

        print(f"\nCluster {cluster_num} - Most Overrepresented Terms:")
        for idx in top_ratio_indices:
            print(f"  {feature_names[idx]}: {ratio[idx]:.2f}x more common")
    return


@app.cell
def _(clusters, coords_2d, df_conv, pl):
    def _():
        import numpy as np

        # Assuming you have:
        # tsne_results: your t-SNE coordinates (N x 2)
        # cluster_labels: your KMeans cluster labels
        # original_texts: your original text data

        # Find the point near x=110, y=0
        target_x, target_y = 110, 0
        tolerance = 10  # adjust if needed

        # Calculate distances to target point
        distances = np.sqrt(
            (coords_2d[:, 0] - target_x) ** 2 + (coords_2d[:, 1] - target_y) ** 2
        )

        # Find closest point
        idx = np.argmin(distances)

        print(f"Closest point index: {idx}")
        print(f"Coordinates: ({coords_2d[idx, 0]:.2f}, {coords_2d[idx, 1]:.2f})")
        print(f"Cluster label: {clusters[idx]}")
        print(f"Distance from target: {distances[idx]:.2f}")
        print("\n" + "=" * 50)
        print("Text content:")
        print(
            df_conv.select(pl.col("first_user_content_tokens"))
            .to_series()
            .to_list()[idx]
        )
        print("=" * 50)

        # Optional: Check if it's really isolated
        orange_cluster = clusters[idx]
        same_cluster_mask = clusters == orange_cluster
        nearby_mask = distances < 20  # points within distance 20

        same_cluster_nearby = np.sum(same_cluster_mask & nearby_mask)
        return print(
            f"\nOther points from same cluster nearby: {same_cluster_nearby - 1}"
        )


    _()
    return


if __name__ == "__main__":
    app.run()
