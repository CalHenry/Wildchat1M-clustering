import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import time
    return duckdb, mo, pl, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Speed contest for aggregation - Polars vs Duckdb
    """)
    return


@app.cell
def _(df_mess_lvl, duckdb, pl, time):
    # ============================================================================
    # PIPELINE FUNCTIONS
    # ============================================================================


    def full_polars_pipeline():
        """Complete Polars pipeline: import -> transform -> aggregate -> collect once"""

        # Step 1: Import data
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

        # Step 2: Unnesting and transformation
        conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]
        struct_fields = (
            df.select("conversation")
            .explode("conversation")
            .head(0)
            .collect()
            .to_series()
            .struct.fields
        )
        new_fields = [
            f"{field}_nested" if field in conflicting_cols else field
            for field in struct_fields
        ]

        df_mess_lvl = (
            df.with_columns(["conversation_hash", "conversation"])
            .explode("conversation")
            .with_columns(pl.col("conversation").struct.rename_fields(new_fields))
            .unnest("conversation")
            .drop(
                "hashed_ip_nested",
                "header_nested",
                "state_nested",
                "country_nested",
                "toxic",
                "language_nested",
            )
            .unnest("header")
        )

        # Step 3: Aggregation
        create_conversation_id = (
            pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
        ).alias("conversation_id")

        df_conv_lvl = (
            df_mess_lvl.with_columns(create_conversation_id)
            .group_by("conversation_id")
            .agg(
                [
                    pl.col("content")
                    .filter(pl.col("role") == "user")
                    .first()
                    .alias("first_user_content"),
                    pl.col("content")
                    .filter(pl.col("role") == "user")
                    .str.join(delimiter="\n\n")
                    .alias("all_user_content"),
                    pl.col("content")
                    .filter(pl.col("role") == "assistant")
                    .str.join(delimiter="\n\n")
                    .alias("all_assistant_content"),
                    pl.col("content")
                    .str.join(delimiter="\n\n")
                    .alias("all_content"),
                    pl.exclude(
                        ["content", "turn_identifier", "role", "timestamp"]
                    ).first(),
                ]
            )
            .collect()  # Single collect at the end
        )

        return df_conv_lvl


    def full_duckdb_pipeline():
        """Complete DuckDB pipeline: import -> transform (polars) -> aggregate (duckdb)"""

        # Step 1: Import data
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

        # Step 2: Unnesting and transformation (using Polars)
        conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]
        struct_fields = (
            df.select("conversation")
            .explode("conversation")
            .head(0)
            .collect()
            .to_series()
            .struct.fields
        )
        new_fields = [
            f"{field}_nested" if field in conflicting_cols else field
            for field in struct_fields
        ]

        df_mess_lvl = (
            df.with_columns(["conversation_hash", "conversation"])
            .explode("conversation")
            .with_columns(pl.col("conversation").struct.rename_fields(new_fields))
            .unnest("conversation")
            .drop(
                "hashed_ip_nested",
                "header_nested",
                "state_nested",
                "country_nested",
                "toxic",
                "language_nested",
            )
            .unnest("header")
            .collect()  # FIRST collect for DuckDB to work with
        )

        # Step 3: Aggregation using DuckDB
        df_final = duckdb.sql("""
            SELECT 
                conversation_hash || '_' || hashed_ip as conversation_id,        
                (ARRAY_AGG(content ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'user'))[1] as first_user_content,
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'user') as all_user_content,
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'assistant') as all_assistant_content,
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) as all_content,
                FIRST(conversation_hash) as conversation_hash,
                FIRST(hashed_ip) as hashed_ip,
                FIRST(lang_conv) as lang_conv,
                FIRST(model) as model,
                FIRST(redacted) as redacted,
                FIRST(turn) as turn,
                FIRST(state) as state,
                FIRST(country) as country,
                FIRST('accept-language') as 'accept-language',
                FIRST('user-agent') as 'user-agent',
            FROM df_mess_lvl
            GROUP BY conversation_id
        """).pl()  # SECOND collect (implicit in .pl())

        return df_final


    # ============================================================================
    # BENCHMARK EXECUTION
    # ============================================================================

    print("=" * 70)
    print("FULL PIPELINE BENCHMARK: DuckDB vs Polars")
    print("=" * 70)

    # Warm-up runs (crucial for JIT compilation and caching)
    print("\nWarming up...")
    _ = full_polars_pipeline()
    _ = full_duckdb_pipeline()
    print("Warm-up complete!\n")

    # Actual timing with multiple runs
    n_runs = 10
    print(f"Running {n_runs} iterations for each approach...\n")

    # Time Polars
    polars_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result_polars = full_polars_pipeline()
        end = time.perf_counter()
        polars_times.append(end - start)
    #    print(f"  Polars run {i + 1}: {polars_times[-1] * 1000:.2f}ms")

    # Time DuckDB
    duckdb_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result_duckdb = full_duckdb_pipeline()
        end = time.perf_counter()
        duckdb_times.append(end - start)
    #    print(f"  DuckDB run {i + 1}: {duckdb_times[-1] * 1000:.2f}ms")

    # Statistics
    polars_avg = sum(polars_times) / n_runs
    duckdb_avg = sum(duckdb_times) / n_runs
    polars_min = min(polars_times)
    duckdb_min = min(duckdb_times)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"Polars average:  {polars_avg * 1000:.2f}ms (min: {polars_min * 1000:.2f}ms)"
    )
    print(
        f"DuckDB average:  {duckdb_avg * 1000:.2f}ms (min: {duckdb_min * 1000:.2f}ms)"
    )
    print()

    if polars_avg < duckdb_avg:
        speedup = (duckdb_avg / polars_avg - 1) * 100
        print(f"🏆 Polars faster by {speedup:.1f}%")
    else:
        speedup = (polars_avg / duckdb_avg - 1) * 100
        print(f"🏆 DuckDB faster by {speedup:.1f}%")

    # Verify results are equivalent
    print("\n Same results ?")
    assert result_polars.shape == result_duckdb.shape, "Row/column counts differ!"
    print(
        f"✔️ Both produce {result_polars.shape[0]} rows × {result_polars.shape[1]} columns"
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Results for a test with 100 runs on "train-00000-of-00014.parquet" (180.7mb)**
    
    *(!todo run the test for the full data, here it's part 1/14 only)*
    
    Polars is faster by ~35% in this use case. Impressive results, so I will use the full polars pipeline.
    
    **Why ?**
    
    - Polars only collect the data once
    - It optimizes the full lazy pipeline
    
    - Duckdb requires to collect twice, for the intermediate result and for the final result
    - The duckdb pipeline can only optimize each step independently.
    
    
    Interesting observations for this use case:
    - The duration difference is negligeable (~300ms)
    - Duckdb is faster for the aggregation part alone (by ~24% from my tests, which is impressive)
    
    A full duckdb pipeline, thus in pure SQL, is probably faster
    """)
    return


if __name__ == "__main__":
    app.run()
