import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import spacy
    import duckdb
    import polars as pl
    return duckdb, mo, pl, spacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison Polars vs Spacy - Clean and tokenize text

    - Polars is faster and itegrates better with the rest of the code
    - Spacy provides a better linguistic quality

    What is the speed difference between the 2 ?
    What are the quality improvements from Spacy ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set up and load data

    Run once to get the data
    """)
    return


@app.cell
def _(duckdb, pl, spacy):
    # Set up
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

    # Load data (lazy)
    df = (
        duckdb.query(
            """
        SELECT * EXCLUDE (toxic, timestamp, redacted, openai_moderation, detoxify_moderation)
        FROM 'data/raw/train-00000-of-00014.parquet' 
        WHERE language = 'English'
        """
        )
        .pl(lazy=True)
        .rename({"language": "lang_conv"})
    )
    conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    struct_fields = (
        df.select("conversation")
        .explode("conversation")
        .head(0)  # don't care about the data, only the var names
        .collect()  # need eager to extract the list
        .to_series()
        .struct.fields
    )

    new_fields = [
        f"{field}_nested" if field in conflicting_cols else field
        for field in struct_fields
    ]
    df_mess_lvl = (
        df.with_columns(["conversation_hash", "conversation"])
        .explode(  # explode the list of struct (-1 layer)
            "conversation"
        )
        .with_columns(
            pl.col("conversation").struct.rename_fields(
                new_fields
            )  # var name conflict between original and nested vars
        )
        .unnest("conversation", "header")  # unnest the struct (now flat)
        .drop(
            "hashed_ip_nested",
            "header_nested",
            "state_nested",
            "country_nested",
            "toxic",
            "user-agent",
            "accept-language",
            "redacted",
            "model",
        )
    ).rename({"language_nested": "language_message"})
    # From the flat dataset, we can aggregate to conversation level
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")

    df_conv_lvl = (
        df_mess_lvl.with_columns(create_conversation_id)
        .group_by("conversation_id")
        .agg(
            [
                # First user message content
                pl.col("content")
                .filter(pl.col("role") == "user")
                .first()
                .alias("first_user_content"),
                # All user messages concatenated
                pl.col("content")
                .filter(pl.col("role") == "user")
                .str.join(delimiter="\n\n")
                .alias("all_user_content"),
                # All assistant messages concatenated
                pl.col("content")
                .filter(pl.col("role") == "assistant")
                .str.join(delimiter="\n\n")
                .alias("all_assistant_content"),
                # All messages concatenated (user and assistant)
                pl.col("content").str.join(delimiter="\n\n").alias("all_content"),
                # Keep other columns (take first value)
                pl.exclude(
                    [
                        "content",
                        "turn_identifier",
                        "role",
                        "timestamp",
                        "language_message",
                    ]
                ).first(),
            ]
        )
    )
    return df_conv_lvl, nlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Contest

    Polars
    """)
    return


@app.cell
def _(pl):
    def _clean_messages(lf: pl.LazyFrame, stop_pattern: str) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("first_user_content")
            .str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"\S+@\S+", "")  # remove emails
            .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
            .str.replace_all(stop_pattern, "")  # remove stop words
            .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
            .str.strip_chars()
            .str.split(" ")
            .alias("token_clean_polars")
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spacy
    """)
    return


@app.cell
def _(nlp, pl):
    def tokenize_spacy_batch(
        lf: pl.LazyFrame, text_column: str, batch_size: int = 5000
    ) -> pl.LazyFrame:
        """
        Use spaCy for better lemmatization and linguistic awareness.
        Slower but better quality for semantic clustering.
        """
        df = lf.collect()
        texts = df[text_column].to_list()

        cleaned_texts = []
        token_lists = []

        # Process in batches
        for doc in nlp.pipe(texts, batch_size=batch_size):
            # Get lemmatized tokens, skip stop words, punctuation, and short tokens
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 2
                and token.is_alpha  # Only alphabetic tokens
            ]

            token_lists.append(tokens)
            cleaned_texts.append(" ".join(tokens))

        # Add both formats to dataframe
        return df.with_columns(
            [
                pl.Series(name=f"{text_column}_tokens", values=token_lists),
                pl.Series(name=f"{text_column}_cleaned", values=cleaned_texts),
            ]
        ).lazy()
    return (tokenize_spacy_batch,)


@app.cell
def _(clean_messages, pl, time, tokenize_spacy_batch):
    def compare_approaches(df: pl.DataFrame, text_column: str):
        """
        Compare different approaches on your data.
        """

        print("Testing tokenization approaches...\n")

        # Test 1: Pure Polars
        start = time.time()
        result1 = clean_messages(df.lazy(), text_column).collect()
        time1 = time.time() - start
        print(f"Polars native: {time1:.3f}s")
        print(f"Sample output: {result1[text_column + '_cleaned'][0][:100]}\n")

        # Test 2: spaCy
        start = time.time()
        result2 = tokenize_spacy_batch(df.lazy(), text_column)
        result2_df = result2.collect()
        time2 = time.time() - start
        print(f"spaCy batch: {time2:.3f}s")
        print(f"Sample output: {result2_df[text_column + '_cleaned'][0][:100]}\n")

        print(f"Speedup: {time2 / time1:.1f}x faster with Polars")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_conv_lvl):
    df_test = df_conv_lvl.head(1000)
    return (df_test,)


@app.cell
def _(df_test, nlp, pl, stop_pattern, time):
    import re
    from typing import List


    def _clean_messages_polars(
        lf: pl.LazyFrame, stop_pattern: str
    ) -> pl.LazyFrame:
        """Original Polars cleaning function"""
        return lf.with_columns(
            pl.col("first_user_content")
            .str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"\S+@\S+", "")  # remove emails
            .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
            .str.replace_all(stop_pattern, "")  # remove stop words
            .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
            .str.strip_chars()
            .str.split(" ")
            .alias("token_clean_polars")
        )


    def _clean_messages_spacy(
        texts: List[str], stop_words: set, batch_size: int = 1000
    ) -> List[List[str]]:
        """
        spaCy cleaning function that mimics Polars behavior with batch processing

        Args:
            texts: List of text strings to process
            stop_words: Set of stop words to remove
            batch_size: Number of texts to process in each batch

        Returns:
            List of token lists (one per text)
        """
        # Convert to lowercase first
        lowercased_texts = [text.lower() for text in texts]

        # Process with spaCy in batches
        results = []
        for doc in nlp.pipe(lowercased_texts, batch_size=batch_size):
            # Get tokens, filtering out URLs, emails, punctuation, and stop words
            tokens = [
                token.text
                for token in doc
                if not token.like_url
                and not token.like_email
                and not token.is_punct
                and token.text not in stop_words
                and token.text.strip()
            ]
            results.append(tokens)

        return results


    def benchmark_comparison(df: pl.DataFrame, stop_words: set, n_runs: int = 3):
        """
        Compare performance between Polars and spaCy cleaning

        Args:
            df: Polars DataFrame with 'first_user_content' column
            stop_words: Set of stop words to remove
            n_runs: Number of benchmark runs for averaging

        Returns:
            Dictionary with timing results and sample outputs
        """
        # Create stop pattern for Polars (regex alternation)
        stop_pattern = r"\b(" + "|".join(stop_words) + r")\b"

        print(f"Benchmarking {n_runs} runs each...\n")

        # Benchmark Polars
        polars_times = []
        for i in range(n_runs):
            lf = df
            start = time.perf_counter()
            result_polars = _clean_messages_polars(lf, stop_pattern).collect()
            end = time.perf_counter()
            polars_times.append(end - start)
            print(f"Polars run {i + 1}: {polars_times[-1]:.4f}s")

        avg_polars = sum(polars_times) / len(polars_times)

        print()

        # Benchmark spaCy
        spacy_times = []
        df_collected = df.collect()
        texts = (
            df_collected.select(pl.col("first_user_content")).to_series().to_list()
        )
        for i in range(n_runs):
            start = time.perf_counter()
            result_spacy = _clean_messages_spacy(texts, stop_words)
            end = time.perf_counter()
            spacy_times.append(end - start)
            print(f"spaCy run {i + 1}: {spacy_times[-1]:.4f}s")

        avg_spacy = sum(spacy_times) / len(spacy_times)

        # Print summary
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)
        print(f"Polars average time: {avg_polars:.4f}s")
        print(f"spaCy average time:  {avg_spacy:.4f}s")
        print(
            f"Speed difference:    {avg_spacy / avg_polars:.2f}x (Polars is faster)"
        )
        print("=" * 50)

        return {
            "polars_avg": avg_polars,
            "spacy_avg": avg_spacy,
            "speedup": avg_spacy / avg_polars,
            "polars_times": polars_times,
            "spacy_times": spacy_times,
        }


    # Example usage
    if __name__ == "__main__":
        # Run benchmark
        results = benchmark_comparison(df_test, stop_pattern, n_runs=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spacy is way too slow. I will either stay with polars, or find another tokeniser library like Tokenizers from Huggingface
    """)
    return


if __name__ == "__main__":
    app.run()
