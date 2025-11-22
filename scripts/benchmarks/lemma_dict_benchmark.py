import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md(r"""
    # Get the list of the unique words in the texts - Polars vs Duckdb

    This notebook is a quick benchmark between 4 approaches (3 Polars and 1 DuckDB).
    The first benchmark draws the data from a lazyframe.
    The second benchmark draws the from a parquet file on disk.

    The data we are insterested in is in a single column containing list[str] values but the dataset has 11 columns.
    The texts are tokenized and we want to get a list of all the unique tokens (necessary in the main workflow for efficient lemmatization). Tokens can be duplicated inside a list and across the lists.
    To get the unique list, we need to explode the lists and then extract all unique values from the pool of all the tokens.
    We have a **LOT** of tokens (1M+ tokens).

    **Technical challenges:**
    Exploding the lists creates one row per token, resulting in a **massive** intermediate dataset.
    Loading the fully exploded dataset into memory was crashing my Python. Therefore I seek solutions and here I am with this benchamark.
    We use lazy evaluation with streaming execution to make this possible.

    How streaming works:

    Polars processes the data in chunks, maintaining a smaller memory footprint. When a chunk is done, it's memory is freed. However, the specific execution strategy varies:

    **Code 1**: Attempts to explode and deduplicate in the lazy query plan before collecting
    **Code 2**: Collects the list column first, then explodes and deduplicates in memory
    **Code 3**: Explicitly processes in batches using iter_slices(), manually aggregating unique tokens


    In the big picture, DuckDB uses the same concepts with a few specificities related to SQL. But it's comparable.

    Both engines optimize by processing data in batches: read batch → explode → deduplicate → move to next batch. This allows handling datasets larger than available RAM.
    The resulting list contains **824,836** unique tokens.


    A second similar benchmark is avaible, this one reads from the same file but on disk as a parquet file instead of a lazyframe in memory (duckdb can't read from a lazyframe).

    The results are quite interesting and we can appreciate the possible optimization of the code.



    Below more details on why I ended up with the codes in the benchmark:
    - I only have 16GB on my macbook
    - **explode()** is the bottleneck of this case. It creates so many rows that leads to OOM.
    - **collect()** is the action crashes python and we have to collect to create a list of unique values.
    - The subset of variables we are in (.select('a_var')), doesn't mean that the other variables are not part of the computations. Look at where the select() paranthesis closes in the final code
    - Exploding a var in the dataset and returning the dataset create a duplicates of the values of the non exploded vars for all the new rows, make the exploded data extremely large.
    - **collect(engine='streaming')** is mandatory
    - the order of the operation is really important for the code to work. **explode()** and **collect()** at the wrong place and it's OOM crashing python. The right flavor of the function is also key. In polars using the right submodule (list, str, struct...) tells what input/ output we have
    - Maximizeing lazy evaluation allow polars to optimize it's query plan which can be the difference between OOM and a working snippet
    """)
    return


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import time
    import tracemalloc
    return duckdb, mo, pl, time, tracemalloc


@app.cell
def _():
    # fmt: off
    try:
        from nltk.corpus import stopwords
    except LookupError:
        import nltk
        nltk.download("stopwords")  # should go to "User/nltk_data/corpus"
        from nltk.corpus import stopwords
        print('stopwords downloaded ✅')
    return (stopwords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the huge cell below to prepare the data for the benchmark.
    It uses the functions on the right col and reproduces the workflow of the main work.
    """)
    return


@app.cell
def _(
    clean_and_tokenize,
    duckdb,
    get_stopwords_pattern,
    pl,
    remove_nolang_messages,
    remove_non_english_mess,
):
    # Load data (lazy)
    lf = (
        duckdb.query(
            """
        SELECT * EXCLUDE (toxic, timestamp, redacted, openai_moderation, detoxify_moderation)
        FROM 'data/raw/*.parquet' 
        WHERE language = 'English'
        """
        )
        .pl(lazy=True)
        .rename({"language": "lang_conv"})
    )

    conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    struct_fields = (
        lf.select("conversation")
        .head(0)  # don't care about the data, only the var names
        .explode("conversation")
        .collect()  # need eager to extract the list
        .to_series()
        .struct.fields
    )

    new_fields = [
        f"{field}_nested" if field in conflicting_cols else field
        for field in struct_fields
    ]

    # no unique id for conversations so we use the hash + the ip to make one
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")


    # Flatten the data (explode nested variables)
    lf_mess_raw = (
        (
            lf.with_columns(["conversation_hash", "conversation"])
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
        )
        .rename({"language_nested": "language_message"})
        .with_columns(create_conversation_id)
    )

    # Clean messages of lf_mess_lvl_raw)
    lf_mess = lf_mess_raw.pipe(remove_nolang_messages).pipe(
        remove_non_english_mess
    )

    stop_patterns = get_stopwords_pattern()

    _lf_conv_init = lf_mess.group_by("conversation_id").agg(
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

    lf_test = clean_and_tokenize(
        _lf_conv_init, "first_user_content", stop_patterns
    )
    return (lf_test,)


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(pl, stopwords):
    def remove_nolang_messages(lf_mess: pl.LazyFrame) -> pl.LazyFrame:
        """
        1. Removes first turns (user + assistant) where user message is empty (Nolang)
        2. Removes any other remaining Nolang messages
        """
        first_turn_empty = (
            lf_mess.filter(
                (pl.col("turn") == 1) & (pl.col("language_message") == "Nolang")
            )
            .select(["conversation_id", "turn_identifier"])
            .unique()
        )

        if first_turn_empty.head(1).collect().height > 0:
            lf_mess = lf_mess.join(  # 1.
                first_turn_empty,
                on=["conversation_id", "turn_identifier"],
                how="anti",
            )

        return lf_mess.remove(pl.col("language_message") == "Nolang")  # 2.


    def remove_non_english_mess(lf_mess: pl.LazyFrame) -> pl.LazyFrame:
        """Keep only messages that are English or Latin"""
        return lf_mess.filter(
            (pl.col("language_message") == "English")
            | (pl.col("language_message") == "Latin")
        )


    # get stopwords list for nltk
    def get_stopwords_pattern() -> str:
        """
        Returns regex pattern of nltk's english stopwords.

        To customize:
        - Add or remove words from `STOP_WORDS` with a dict
        - To exclude words, replace '= set()' by a dict like so:
            REMOVE_FROM_STOPWORDS = {"i", "who",...}
        """
        STOP_WORDS = set(stopwords.words("english"))

        REMOVE_FROM_STOPWORDS = set()
        ADD_TO_STOPWORDS = set()
        STOP_WORDS.update(ADD_TO_STOPWORDS)  # add words
        STOP_WORDS.difference_update(REMOVE_FROM_STOPWORDS)  # remove words

        stop_pattern = "|".join(f"\\b{word}\\b" for word in STOP_WORDS)

        return stop_pattern


    def clean_and_tokenize(
        lf_conv: pl.LazyFrame, columns: str | list[str], stop_pattern: str
    ) -> pl.LazyFrame:
        """Clean and tokenize the aggregated messages"""

        if isinstance(columns, str):  # single string to list for consistency
            columns = [columns]

        return (
            lf_conv.with_columns(
                pl.col(columns)
                .str.to_lowercase()
                .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
                .str.replace_all(r"\S+@\S+", "")  # remove emails
                .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
                .str.replace_all(stop_pattern, "")  # remove stop words
                .str.replace_all(r"\d+", "")  # remove numbers
                .str.replace_all(
                    r"[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]",
                    "",
                )  # remove non latin letters (breakdown below)
                .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
                .str.strip_chars()
                .str.split(" ")  # tokenize
                .list.eval(
                    pl.element()
                    .filter(pl.element() != "")
                    .filter(pl.element().str.len_chars() > 1)
                )  # remove empty tokens and single characters tokens
            )
            .drop_nulls(columns)  # remove rows with empty messages
            .rename({col: f"{col}_tokens" for col in columns})
        )
    return (
        clean_and_tokenize,
        get_stopwords_pattern,
        remove_nolang_messages,
        remove_non_english_mess,
    )


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark
    """)
    return


@app.cell
def _(lf_test, pl, time, tracemalloc):
    def benchmark_code(name, code_func):
        """Benchmark execution time and peak memory usage"""
        tracemalloc.start()
        start_time = time.time()

        result = code_func()

        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{name}:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Peak Memory: {peak / 1024**2:.1f} MB")
        print(
            f"  Result size: {len(result) if isinstance(result, (list, set)) else 'N/A'}"
        )
        print()

        return result, elapsed, peak


    # Define the three approaches
    def code1():
        return (
            lf_test.select(pl.col("first_user_content_tokens").list.explode())
            .unique()
            .drop_nulls()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )


    def code2():
        return (
            lf_test.select(pl.col("first_user_content_tokens"))
            .collect(engine="streaming")
            .select(pl.col("first_user_content_tokens").list.explode().unique())
            .drop_nulls()
            .to_series()
            .to_list()
        )


    def code3():
        return (
            lf_test.select(pl.col("first_user_content_tokens"))
            .explode("first_user_content_tokens")
            .unique("first_user_content_tokens")
            .drop_nulls("first_user_content_tokens")
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )


    # Run benchmarks
    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print()

    result1, time1, mem1 = benchmark_code("Code 1 (explode in lazy)", code1)
    result2, time2, mem2 = benchmark_code("Code 2 (collect then explode)", code2)
    result3, time3, mem3 = benchmark_code("Code 3 (batched processing)", code3)


    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    fastest = min(time1, time2, time3)
    lowest_mem = min(mem1, mem2, mem3)

    if time1 == fastest:
        print(f"⚡ Fastest: Code 1 ({time1:.2f}s)")
    elif time2 == fastest:
        print(f"⚡ Fastest: Code 2 ({time2:.2f}s)")
    else:
        print(f"⚡ Fastest: Code 3 ({time3:.2f}s)")

    if mem1 == lowest_mem:
        print(f"💾 Lowest Memory: Code 1 ({mem1 / 1024**2:.1f} MB)")
    elif mem2 == lowest_mem:
        print(f"💾 Lowest Memory: Code 2 ({mem2 / 1024**2:.1f} MB)")
    else:
        print(f"💾 Lowest Memory: Code 3 ({mem3 / 1024**2:.1f} MB)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Benchmark took 56sec on my machine.
    Batch processing while being comparable in speed is worse for the memory pressure.
    Code 1 is equivalent to code 2 in memory but is ~2 sec faster. It's the winner.


    #### RUN 1
    ==================================================
    BENCHMARK RESULTS
    ==================================================

    Code 1 (explode in lazy):
      Time: 17.50s
      Peak Memory: 50.6 MB
      Result size: 824836

    Code 2 (collect then explode):
      Time: 19.09s
      Peak Memory: 50.6 MB
      Result size: 824836

    Code 3 (batched processing):
      Time: 19.11s
      Peak Memory: 96.0 MB
      Result size: 824836

    ==================================================
    SUMMARY
    ==================================================
    ⚡ Fastest: Code 1 (17.50s)
    💾 Lowest Memory: Code 2 (50.6 MB)


    #### RUN 2

    ==================================================
    BENCHMARK RESULTS
    ==================================================

    Code 1 (explode in lazy):
      Time: 17.31s
      Peak Memory: 54.8 MB
      Result size: 824836

    Code 2 (collect then explode):
      Time: 19.38s
      Peak Memory: 50.6 MB
      Result size: 824836

    Code 3 (batched processing):
      Time: 19.27s
      Peak Memory: 96.1 MB
      Result size: 824836

    ==================================================
    SUMMARY
    ==================================================
    ⚡ Fastest: Code 1 (17.31s)
    💾 Lowest Memory: Code 2 (50.6 MB)


    #### RUN 3 (no batch)
    ==================================================
    BENCHMARK RESULTS
    ==================================================

    Code 1 (explode in lazy):
      Time: 17.29s
      Peak Memory: 50.6 MB
      Result size: 824836

    Code 2 (collect then explode):
      Time: 19.68s
      Peak Memory: 50.8 MB
      Result size: 824836

    ==================================================
    SUMMARY
    ==================================================
    ⚡ Fastest: Code 1 (17.29s)
    💾 Lowest Memory: Code 1 (50.6 MB)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    def code3():
        result = set()
        for batch in (
            lf_test.select("first_user_content_tokens")
            .collect(engine="streaming")
            .iter_slices(100000)
        ):
            tokens = (
                batch["first_user_content_tokens"]
                .list.explode()
                .unique()
                .drop_nulls()
            )
            result.update(tokens.to_list())
        return list(result)
    """)
    return


app._unparsable_cell(
    r"""
    # Same benchmark but reading from a file (same df but in .parquet on disk)
    # reading from disk is much faster, same memory load


    def _():
        file_path = \"data/processed/df_test.parquet\"

        def benchmark_code(name, code_func):
            \"\"\"Benchmark execution time and peak memory usage\"\"\"
            tracemalloc.start()
            start_time = time.time()
            result = code_func()
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f\"{name}:\")
            print(f\"  Time: {elapsed:.2f}s\")
            print(f\"  Peak Memory: {peak / 1024**2:.1f} MB\")
            print(
                f\"  Result size: {len(result) if isinstance(result, (list, set)) else 'N/A'}\"
            )
            print()
            return result, elapsed, peak

        # 4 approaches - all reading from Parquet
        def code1():
            lf = pl.scan_parquet(file_path)
            return (
                lf.select(pl.col(\"first_user_content_tokens\").list.explode())
                .unique()
                .drop_nulls()
                .collect(engine=\"streaming\")
                .to_series()
                .to_list()
            )

        def code2():
            lf = pl.scan_parquet(file_path)
            return (
                lf.select(pl.col(\"first_user_content_tokens\"))
                .collect(engine=\"streaming\")
                .select(
                    pl.col(\"first_user_content_tokens\").list.explode().unique()
                )
                .drop_nulls()
                .to_series()
                .to_list()
            )

         def code3():
            lf = pl.scan_parquet(file_path)
            result = set()
            for batch in (
                lf.select(\"first_user_content_tokens\")
                .collect(engine=\"streaming\")
                .iter_slices(100000)
            ):
                tokens = (
                    batch[\"first_user_content_tokens\"]
                    .list.explode()
                    .unique()
                    .drop_nulls()
                )
                result.update(tokens.to_list())
            return list(result)

        def codeduckdb():
            conn = duckdb.connect()
            query = \"\"\"
            WITH unnested AS (
                SELECT unnest(first_user_content_tokens) as token
                FROM read_parquet(?)
            )
            SELECT DISTINCT token
            FROM unnested
            WHERE token IS NOT NULL
            \"\"\"

            result = conn.execute(query, [file_path]).fetchall()
            return [row[0] for row in result]

        # Run benchmarks
        print(\"=\" * 50)
        print(\"BENCHMARK RESULTS\")
        print(\"=\" * 50)
        print()

        result1, time1, mem1 = benchmark_code(\"Code 1 (explode in lazy)\", code1)
        result2, time2, mem2 = benchmark_code(
            \"Code 2 (collect then explode)\", code2
        )
        result3, time3, mem3 = benchmark_code(\"Code 3 (batched processing)\", code3)
        result4, time4, mem4 = benchmark_code(\"Code 4 (duckdb SQL)\", codeduckdb)

        # Summary
        print(\"=\" * 50)
        print(\"SUMMARY\")
        print(\"=\" * 50)

        fastest = min(time1, time2, time3, time4)
        lowest_mem = min(mem1, mem2, mem3, mem4)

        times = [
            (time1, \"Code 1\"),
            (time2, \"Code 2\"),
            (time3, \"Code 3\"),
            (time4, \"Code 4\"),
        ]
        mems = [
            (mem1, \"Code 1\"),
            (mem2, \"Code 2\"),
            (mem3, \"Code 3\"),
            (mem4, \"Code 4\"),
        ]

        fastest_code = min(times, key=lambda x: x[0])
        lowest_mem_code = min(mems, key=lambda x: x[0])

        print(f\"⚡ Fastest: {fastest_code[1]} ({fastest_code[0]:.2f}s)\")
        return print(
            f\"💾 Lowest Memory: {lowest_mem_code[1]} ({lowest_mem_code[0] / 1024**2:.1f} MB)\"
        )


    _()
    """,
    column=3, disabled=False, hide_code=False, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bacthed processing is the looser.
    Code 2 is the slowest but has the best memory load.
    Duckdb is fast but the memory load is the highest by a lot.
    Code 1 is the winner. It's the fastest and has the lowest memory foot print (same as Code2)

    Polars win !
    """)
    return


if __name__ == "__main__":
    app.run()
