import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    return duckdb, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data preparation

    ### A word on the libraries used:

    **Duckdb** is great and **Polars** is awsome.

    I could use Duckdb only with the SQL queries but I am more confortable with **Polars's** synthax for complex and chainned modifications to dataframes.
    They both have lazy excecution and are very fast.

    I can use:
    - SQL at the begining, to import data or a filtered version from raw files. Simple queries can do a lot.
    - Polars for the complex manipulations in the workflow.
    - SQL and Duckdb for big aggregations and joins.

    Benefits of using duckdb in my workflow:
    - reading data is extremly fast (although polars is very fast as well)
    - can import in a  super efficient way with just a simple SQL query
    - SQL relations are similar to polars's expressions and can be very handy
    - SQL views are also very handy


    ➡️ Both libraries are interchangeable here and I can choose the one I prefer for each part of my work and keep a seamless workflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next cell is a great example of seemless integration between the 2 libraries:
    1. We use duckdb for a SQL query
    2. We specify that we want to convert the query's result to Polars df.
    As soon as ```.pl(lazy=True)``` is written, the object is now a polars LazyFrame, to which i can apply polars code.
    I used both libraries in the same sequence.
    """)
    return


@app.cell
def _(duckdb):
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
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Few observations on the raw dataset:

    - This dataset has 3 levels:
    1. **User** level: identified by **'hashed_ip'** (based on the user's IP address)
    2. **Conversations** level: identified by **'conversation_hash'**. *Uniquely* identified by **'conversation_hash'** + **'hashed_ip'**. ('conversation_hash' is not a unique identifier. 2 identical conversations will have the hash regarless the user, messages inside the conv will always have unique id)
    3. **Message** level: identified **'turn_identifier'**

    1 user can have N conversations of N messages.
    1 message belong to 1 conversation that belong to 1 user

    - Messages works in pairs: one **user** message, one **assistant** response. **turn_identifier** + **role** give the order of the messages. The **user** always comes before the **assistant**.
    - **'conversations'** is a complex variable of **List[structs]**. Each element of the list is 1 message. The fields of the **struct** are the message's content and it's metadata. Some fields are specific to the user, others to the assistant (expected missing values).

    ### Tidy: manipulation, variable selection

    We want to be at the **conversation level**, but we don't want a complex schema of **List[structs]** inside a single variable like we have in the raw dataset (the list can be +100 elements for long conversations).
    Logic:
    - flatten the dataset to message level, each row is a unique message, n rows for the same conversation.
    - aggregate back to conversation level by creating features around the messages. We keep metadata variables for the conversations and not the messages.
    - Final tidy dataset at the conversation level with several variables for different aggregations of the  messages's content (see section for details).


    ### Order of the tasks:

    1. Import data, shortlist of variables, filter only English conversations
    2. User level --> Message level (flatten the dataset)
    3. Unique ID for conversations
    4. Message level --> Conversation level
    """)
    return


@app.cell
def _(df):
    # Same variables names in the nested vars of 'conversation'. We have to rename them before unnesting
    # We do it his way because we want to keep the names on the front vars and remove some nested vars bc their info is already in the front vars

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
    return conflicting_cols, new_fields


@app.cell
def _(df, new_fields, pl):
    df_mess_lvl = (
        df.with_columns(["conversation_hash", "conversation"])
        .explode(  # explode the list of struct (-1 layer)
            "conversation"
        )
        .with_columns(
            pl.col("conversation").struct.rename_fields(
                new_fields
            )  # var name conflict between vars and nested vars
        )
        .unnest(  # unnest the struct (now flat)
            "conversation"
        )
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
    return (df_mess_lvl,)


@app.cell
def _(df_mess_lvl, pl):
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
                    ["content", "turn_identifier", "role", "timestamp"]
                ).first(),
            ]
        )
    )
    return (df_conv_lvl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(df, df_conv_lvl, df_mess_lvl, mo):
    mo.md(rf"""
    Takeway: 

    3 levels in the data:
    - User (top level, raw data)
    - Conversation (1 user can have [1:n] conversations)
    - Message

    Manipulation:   
    User level *{df.collect().height} rows* ➡️ Message level *{df_mess_lvl.collect().height} rows* ➡️ Conversation level *{df_conv_lvl.collect().height} rows*

    Most of the users have a single conversation. There is only 71 more conversations than users
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploration

    Conversations are a set of messages between the user and the assistant (AI).
    What are baasic elements about the conversations and messages ?

    - Number of turns (1 turn is a set of 1 message from the user and 1 message (the response) from the assistant)
    - Number of message per conversations
    - user's messages to assistant's messages ratio
    - Lenght of messages/ conversation in tokens
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Test zone
    """)
    return


@app.cell
def _(df):
    dff = df.head(30)
    return (dff,)


@app.cell
def _(conflicting_cols, dff):
    _struct_fields = (
        dff.select("conversation")
        .explode("conversation")
        .head(0)
        .collect()
        .to_series()
        .struct.fields
    )

    _new_fields = [
        f"{field}_nested" if field in conflicting_cols else field
        for field in _struct_fields
    ]
    _struct_fields
    return


@app.cell
def _(dff):
    # Same variables names in the nested vars of 'conversation'. We have to rename them before unnesting
    # We do it his way because we wwant to keep the original names on the vars and some nested vars will be removed bc of duplication of info
    _conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    _struct_fields = (
        dff.select("conversation")
        .explode("conversation")
        .head(0)  # don't care about the data, only the var names
        .collect()  # need eager to extract the list
        .to_series()
        .struct.fields
    )

    _new_fields = [
        f"{field}_nested" if field in _conflicting_cols else field
        for field in _struct_fields
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    """'conversation' schema:

    List(Struct({
    'content': String,
    'country': String,
    'hashed_ip': String,
    'header': Struct({
        'accept-language': String,
        'user-agent': String}),
    'language': String,
    'redacted': Boolean,
    'role': String,
    'state': String,
    'timestamp': Datetime(time_unit='us', time_zone='Europe/Paris'),
    'toxic': Boolean,
    'turn_identifier': Int64}))
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_mess_lvl):
    dft = df_mess_lvl.head(100).collect()
    return (dft,)


@app.cell
def _(mo):
    mo.md(r"""
    aggregate back to conversation level
    - 1 var for the first user message
    - 1 var for all user messages
    - 1 var for all assistant messages
    - 1 var with ALL the messages combined

    'turn' gives the number of round user-assitant
    'hashed_id' is the identifier of the HUMAN that talked to chatgptp (USER level)
    'conversation_hash' is the identifier of the conversation (set of messages) for a given hashed_ip (CONVERSATION level)

    'turn_identifier' gives us the couples user/ assistant fro 2 messages. The *user* message is the first one to come up (MESSAGE level)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_mess_lvl, pl):
    _create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")

    _df_conv_lvl = (
        df_mess_lvl.with_columns(_create_conversation_id)
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
                    ["content", "turn_identifier", "role", "timestamp"]
                ).first(),
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _(dft):
    dft
    return


@app.cell
def _(alt, dft, mo):
    chart = (
        alt.Chart(dft)
        .mark_bar()
        .encode(
            x=alt.X("turn:N", labelAngle=0),
            y="count()",
        )
        .properties(title="Count of turns")
    )

    mo.ui.altair_chart(chart)
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Speed contest for aggregation - Polars vs Duckdb
    """)
    return


@app.cell
def _(df_mess_lvl, duckdb, pl):
    import time

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
    n_runs = 100
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
