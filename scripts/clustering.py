import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import json
    import duckdb
    import polars as pl
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans, HDBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from umap import UMAP
    import random

    import matplotlib.pyplot as plt
    import seaborn as sns

    from typing import Literal
    return (
        HDBSCAN,
        MiniBatchKMeans,
        TSNE,
        TfidfVectorizer,
        TruncatedSVD,
        UMAP,
        json,
        mo,
        normalize,
        np,
        pl,
        plt,
        random,
        silhouette_score,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.outline()
    return


@app.cell
def _(json, pl):
    # import data

    with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokens_vars = "first_user_content_tokens"

    # ! this can be memory intensive (select the right variables before collecting)
    # (there is a 'low_memory' option on .scan_parquet())
    df_conv = (
        pl.scan_parquet("data/clean/df_conv_fuc.parquet")
        .tail(5000)
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
        # no preprocessing steps from sklearn's tfidf. All done earlier and manually with polars.
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


@app.cell
def _(get_tsne_2D_coords, tfidf_matrix):
    tsne_2d = get_tsne_2D_coords(tfidf_matrix)
    return (tsne_2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    #### Helpers
    """)
    return


@app.cell
def _(TSNE, TruncatedSVD):
    def get_tsne_2D_coords(tfidf_matrix, perplexity=30) -> dict:
        """
        run TSNE with 2 components after a TruncatedSVD to reduce computation time
        (input is a tfidf matrix which is sparse)

        ! TSNE can be long and expensive compute
        """

        svd = TruncatedSVD(n_components=2, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)

        # tsne
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_coords = tsne.fit_transform(tfidf_reduced)
        tsne_result = manifold_result = {
            "name": tsne.__class__.__name__,
            "coords": tsne_coords,
        }
        return tsne_result
    return (get_tsne_2D_coords,)


@app.cell
def _(df_conv, df_size, feature_names, np, pl, random, tfidf_matrix):
    def nbr_obs_clusters(clustering_algo):
        """
        print the nbr of obs per cluster
        args:
            clustering_algo: name of the object
        """
        clusters = clustering_algo.labels_
        nbr_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        for cluster_num in range(nbr_clusters):
            cluster_indices = np.where(clusters == cluster_num)[0]
            print(
                f"Cluster {cluster_num} has {len(cluster_indices)} conversations, {len(cluster_indices) / df_size:.1%} of total"
            )


    def top15_tfidf_terms(clustering_algo):
        """
        Print top 15 terms with the higest TFIDF score for each cluster

        args:
            clustering_algo: name of the object
        ex: custom_kmeans.fit(x), provide 'custome_kmeans' as the algo_used
        """
        clusters = clustering_algo.labels_
        nbr_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        for cluster_num in range(nbr_clusters):
            cluster_indices = np.where(clusters == cluster_num)[0]

            # TF-IDF scores for the cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1

            # Get top 15 terms
            top_indices = cluster_tfidf.argsort()[-15:][::-1]
            top_terms = [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

            print(f"\nCluster {cluster_num}:")
            for term, score in top_terms:
                print(f"  {term}: {score:.4f}")


    def sample_rdm_conversations(clustering_algo, nbr_conv_to_print: int):
        """
        Print n random conversations for each cluster
        print first 500 characters of each message (! can print a lot to the console)

        args:
            clustering_algo: name of the object
        """
        clusters = clustering_algo.labels_
        nbr_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        for cluster_num in range(nbr_clusters):
            cluster_indices = np.where(clusters == cluster_num)[0]

            # Sample 10 random conversations
            sample_size = min(nbr_conv_to_print, len(cluster_indices))
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
                print("...")


    def most_overrepresented_terms(clustering_algo):
        """
        Print top 15 most overrepresented TFIDF terms for each cluster

        args:
            clustering_algo: name of the object
        """

        overall_mean = tfidf_matrix.mean(axis=0).A1

        clusters = clustering_algo.labels_
        n_clusters = len(np.unique(clusters))

        for cluster_num in range(n_clusters):
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
def _(coords_2d, np, plt, sns):
    # plot functions

    # Big picture
    def plot_clutsters(clustering_algo, manifold_result: dict, hide_noise: bool):
        """
        plot all the points, colored by cluster on a 2D representation
        (TSNE usually better to see clusters in 2D)

        args:
            clustering_algo: name of the object
            manifold_result: dict of a name and the coords of a manifold algo
        """
        clusters = clustering_algo.labels_
        algo_name = clustering_algo.__class__.__name__
        manifold_name = manifold_result["name"]
        manifold_coords = manifold_result["coords"]

        # check if noise cluster existe, if not do nothing
        has_noise = -1 in clusters
        if hide_noise and has_noise:
            no_noise_mask = clusters != -1
            coord_2d = manifold_coords[no_noise_mask]
        else:
            coord_2d = manifold_coords

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=coord_2d[:, 0],
            y=coord_2d[:, 1],
            hue=clusters,
            palette="tab10",
            alpha=0.6,
            s=20,
            edgecolor="k",
            linewidth=0.5,
        )

        plt.title(
            f"{algo_name} Clusters Visualization (TruncatedSVD)",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel(f"{manifold_name} 1")
        plt.ylabel(f"{manifold_name} 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


    # detailled picture
    def plot_clusters_grid(
        clustering_algo, manifold_result: dict, hide_noise: bool
    ):
        """
        plot each cluster in a seperate cell of a grid
        Allow to see where are all the points of a given cluster
        Helps understand the big picture

        args:
            clustering_algo: name of the object
            manifold_result: dict of a name and the coords of a manifold algo
        """

        clusters = clustering_algo.labels_
        n_clusters = len(np.unique(clusters))
        manifold_name = manifold_result["name"]
        manifold_coords = manifold_result["coords"]

        # check if noise cluster existe, if not do nothing
        has_noise = -1 in clusters
        if hide_noise and has_noise:
            no_noise_mask = clusters != -1
            coord_2d = manifold_coords[no_noise_mask]
        else:
            coord_2d = manifold_coords

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
            ax.set_xlabel(f"{manifold_name} 1")
            ax.set_ylabel(f"{manifold_name} 2")
            # Set same scale for all subplots
            ax.set_xlim(global_x_min, global_x_max)
            ax.set_ylim(global_y_min, global_y_max)

        plt.tight_layout()
        plt.show()
    return plot_clusters_grid, plot_clutsters


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ##Kmeans
    """)
    return


@app.cell
def _(MiniBatchKMeans, np, silhouette_score, tfidf_matrix):
    # find number of clusters
    # Elbow method
    inertia_values = []
    k_range = range(1, 8)


    for k in k_range:
        kmeans = MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=10, random_state=42
        )
        kmeans.fit(tfidf_matrix)
        inertia_values.append(kmeans.inertia_)

    # Silhouette score
    silhouette_values = []
    sample_size = min(50000, tfidf_matrix.shape[0])

    for _k in range(2, 8):
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

    2 seems on a lower side for cluster in chatgpt conversations.
    Let's see silhouette score with a sample of 50 000 points.
    """)
    return


@app.cell
def _(inertia_values, k_range, plt, silhouette_values):
    def _():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # --- First subplot: Elbow Method ---
        ax[0].plot(k_range, inertia_values, "bo-")
        ax[0].set_xlabel("Number of clusters (K)")
        ax[0].set_ylabel("Inertia")
        ax[0].set_title("Elbow Method")
        ax[0].set_xticks(k_range)
        ax[0].grid(True)

        # --- Second subplot: Silhouette Score ---
        ax[1].plot(range(2, 8), silhouette_values, "ro-")
        ax[1].set_xlabel("Number of clusters (K)")
        ax[1].set_ylabel("Mean Silhouette Score")
        ax[1].set_title("Silhouette Score")
        ax[1].set_xticks(range(2, 11))
        ax[1].grid(True)
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
    kmeans_clusters = final_kmeans.fit_predict(tfidf_normalized)
    centroids = final_kmeans.cluster_centers_
    return (kmeans_clusters,)


@app.cell
def _(TSNE, TruncatedSVD, tfidf_matrix):
    svd = TruncatedSVD(n_components=2, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # tsne
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(tfidf_reduced)
    tsne_result = manifold_result = {
        "name": tsne.__class__.__name__,
        "coords": tsne_coords,
    }
    # 23m runtime with the full data
    return


@app.cell
def _(hdb, plot_clutsters, tsne_2d):
    plot_clutsters(hdb, tsne_2d, hide_noise=True)
    return


@app.cell
def _(hdb, plot_clusters_grid, umap_2d):
    plot_clusters_grid(hdb, umap_2d, hide_noise=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What do the plots tell us ?
    - **Cluster 0** is the largest, takes most of the space. It contains the majority of conversations. It likely captures common conversations and don't represent a particular group.

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
def _(coords_2d, df_conv, kmeans_clusters, np, pl):
    # This cell helps us find a specific point, near a position that we can estimate from the plot of the clusters
    # Prints the content of the message

    target_x, target_y = 110, 0

    # Calculate distances to target point
    distances = np.sqrt(
        (coords_2d[:, 0] - target_x) ** 2 + (coords_2d[:, 1] - target_y) ** 2
    )

    # Find closest point
    idx = np.argmin(distances)

    print(f"Closest point index: {idx}")
    print(f"Coordinates: ({coords_2d[idx, 0]:.2f}, {coords_2d[idx, 1]:.2f})")
    print(f"Cluster label: {kmeans_clusters[idx]}")
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
    orange_cluster = kmeans_clusters[idx]
    same_cluster_mask = kmeans_clusters == orange_cluster
    nearby_mask = distances < 20  # points within distance 20

    same_cluster_nearby = np.sum(same_cluster_mask & nearby_mask)
    print(f"\nOther points from same cluster nearby: {same_cluster_nearby - 1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Cluster 0:**
    Contains very common verbs (make, write, give) and nouns (story, message) typical of chatbot conversations.
    Appears to include code-related conversations (random messages can print out code blocks)
    Most represented words include code-related terms but also unexpected words (like "shrek").
    These elements show that this cluster groups together many different kinds of conversations that don't share much semantic meaning but may have vocabulary in common.

    **Cluster 1:**
    Interesting discovery when examining random messages from the cluster: they all start with a prompt for 'MidJourney', a generative model for images ("As a prompt generator for a generative AI called 'Midjourney'..."). The messages following this generic prompt differ from each other.
    Surprisingly, the most represented words don't include 'MidJourney' and include words whose meaning I don't understand.

    **Cluster 2:**
    Random messages show fiction prompts with manga characters
    Almost all the most represented words are constructed with 'art' + word (like "artholiday").
    This cluster is large and spread out, but the information from the messages and top words would better fit a small, compact cluster. There is more to understand about this cluster.

    **Cluster 3:**
    The top words include proper nouns and words that seem to be part of the lexical field of school/club settings found in high school fiction
    Random messages confirm this pattern. All the messages are conversations between Monika, Sayori, Natsuki, MC, and possibly others.
    This cluster, being the smallest and most compact, may be composed only of conversation messages about specific characters, making these prompts very unique compared to others in the dataset thanks to the character names.

    **Cluster 4:**
    Top words include names and words related to fiction
    In the random messages, one prompt appears very frequently: "Write dialogue from" and small variations.   However, this isn't the only kind—others are still related to storytelling and fiction writing.

    **In conclusion**, the largest cluster is probably hiding a substantial part of the dataset's content, while the other clusters represent very specific messages or types of prompts. Fiction content generation appears to be the most common usage, as it is the core of at least 2 of the clusters.
    These results are not surprising and represent expected usage of an AI like ChatGPT.
    TF-IDF shows that it can capture some of the semantic meaning in documents but is limited in its power, as evidenced by the presence of a cluster containing 76% of the observations.
    The next step would be to use a different model with the TF-IDF matrix to see if a different approach can better define the clusters. HDBSCAN is a good candidate as it determines the number of clusters automatically.
    After that, we need to go further and use an embedding model to capture more semantic power in our texts.
    """)
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## HDBCSCAN

    Workflow:

    0. TFIDF matix
    1. dimension reduction with **SVD**
    2. dimension reduction with **UMAP**
    3. **HDBSCAN**
    4. Plots
    5. Exploration of the clusters
    """)
    return


@app.cell
def _(TruncatedSVD, UMAP):
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


    def get_umap_2D_coords(
        tfidf_matrix,
        umap_metric: str = "cosine",
        umap_n_neighbors: int = 200,
    ) -> dict:
        """
        run UMAP with 2 components after a TruncatedSVD to reduce computation time
        (input is a tfidf matrix which is sparse)
        """
        svd = TruncatedSVD(n_components=2, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)

        umap = UMAP(
            n_components=2, metric=umap_metric, n_neighbors=umap_n_neighbors
        )
        umap_coords = umap.fit_transform(tfidf_reduced)
        umap_result = manifold_result = {
            "name": umap.__class__.__name__,
            "coords": umap_coords,
        }
        return umap_result
    return get_umap_2D_coords, umap_tfidf_matrix


@app.cell
def _(get_umap_2D_coords, tfidf_matrix):
    umap_2d = get_umap_2D_coords(tfidf_matrix)
    return (umap_2d,)


@app.cell
def _(tfidf_matrix, umap_tfidf_matrix):
    X_umap = umap_tfidf_matrix(tfidf_matrix)
    return (X_umap,)


@app.cell
def _(HDBSCAN, X_umap):
    hdb = HDBSCAN(min_cluster_size=300, cluster_selection_epsilon=0.5)
    hdb.fit(X_umap)
    return (hdb,)


@app.cell
def _(coords_2d, hdb, plt, sns):
    non_noise_mask = hdb.labels_ != -1

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=coords_2d[non_noise_mask][:, 0],
        y=coords_2d[non_noise_mask][:, 1],
        hue=hdb.labels_[non_noise_mask],
        palette="tab10",
        alpha=0.6,
        s=20,
        edgecolor="k",
        linewidth=0.5,
    )

    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    # plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend("")
    plt.tight_layout()
    plt.show()
    return (non_noise_mask,)


@app.cell
def _(hdb, non_noise_mask, plt, sns, umap_plot):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=umap_plot[non_noise_mask][:, 0],
        y=umap_plot[non_noise_mask][:, 1],
        hue=hdb.labels_[non_noise_mask],
        palette="tab10",
        alpha=0.6,
        s=20,
        edgecolor="k",
        linewidth=0.5,
    )

    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    # plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend("")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Robust ml exploration workflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
