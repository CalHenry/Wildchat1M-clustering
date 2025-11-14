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
        )
        .unnest("header")
        .rename({"language_nested": "language_message"})
    )
    return (df_mess_lvl,)


@app.cell
def _(pl):
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")
    return (create_conversation_id,)


@app.cell
def _(create_conversation_id, df_mess_lvl, pl):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
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


@app.cell
def _(dff, new_fields, pl):
    dff.with_columns(
        ["conversation_hash", "conversation"]
    ).explode(  # explode the list of struct (-1 layer)
        "conversation"
    ).with_columns(
        pl.col("conversation").struct.rename_fields(
            new_fields
        )  # var name conflict between vars and nested vars
    ).unnest(  # unnest the struct (now flat)
        "conversation"
    ).drop(
        "hashed_ip_nested",
        "header_nested",
        "state_nested",
        "country_nested",
        "toxic",
    ).unnest("header").rename({"language_nested": "language_message"})
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
    dfff = df_mess_lvl.head(100).collect()
    return


@app.cell
def _(pl):
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")
    return (create_conversation_id,)


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
def _(create_conversation_id, df_mess_lvl, pl):
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
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Speed contest for aggregation - Polars vs Duckdb
    """)
    return


@app.cell
def _(create_conversation_id, df_mess_lvl, duckdb, pl):
    import time

    test_df = df_mess_lvl.clone()


    # Polars lazy approach
    def polars_approach(wip):
        return (
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
                    pl.col("content")
                    .str.join(delimiter="\n\n")
                    .alias("all_content"),
                    # Keep other columns (take first value)
                    pl.exclude(
                        ["content", "turn_identifier", "role", "timestamp"]
                    ).first(),
                ]
            )
            .collect()
        )


    def duckdb_approach(test_df_collected):
        return duckdb.sql("""
            SELECT 
                conversation_hash || '_' || hashed_ip as conversation_id,        
                -- First user message content
                (ARRAY_AGG(content ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'user'))[1] as first_user_content,

                -- All user messages concatenated
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'user') as all_user_content,

                -- All assistant messages concatenated
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) 
                 FILTER (WHERE role = 'assistant') as all_assistant_content,

                -- All messages concatenated
                STRING_AGG(content, '\n\n' ORDER BY turn_identifier) as all_content,

                -- Keep other columns (first value)
                FIRST(hashed_ip) as hashed_ip,
                FIRST(lang_conv) as lang_conv,
                FIRST(model) as model,
                FIRST(redacted) as redacted,
                FIRST(turn) as turn,
                FIRST(state) as state,
                FIRST(country) as country,
                FIRST('accept-language') as 'accept-language',
                FIRST('user-agent') as 'user-agent',

            FROM test_df_collected
            GROUP BY conversation_id
        """).pl()


    # Benchmark
    test_df_collected = test_df.collect()  # Collect once for fair comparison

    # Warm-up runs (important!)
    _ = polars_approach(test_df.clone())
    _ = duckdb_approach(test_df_collected)

    # Actual timing
    n_runs = 100

    start = time.perf_counter()
    for _ in range(n_runs):
        result_polars = polars_approach(test_df.clone())
    end = time.perf_counter()
    polars_time = (end - start) / n_runs

    start = time.perf_counter()
    for _ in range(n_runs):
        result_duckdb = duckdb_approach(test_df_collected)
    end = time.perf_counter()
    duckdb_time = (end - start) / n_runs

    print(f"Polars: {polars_time * 1000:.2f}ms")
    print(f"DuckDB: {duckdb_time * 1000:.2f}ms")
    print(
        f"Winner: {'Polars' if polars_time < duckdb_time else 'DuckDB'} by {abs(polars_time - duckdb_time) / min(polars_time, duckdb_time) * 100:.1f}%"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With 100 runs, Duckdb is faster by ~24%.

    2 options:
    1. Keep polars for the aggregation part
    2. Use duckdb for the aggregation part

    Pros of 1. :
    - Full polars pipeline, can optimize the full pipeline, only collect once at the end

    Pros of 2. :
    - duckdb seemns faster for the aggregation part
    Cons of 2. :
    - duckdb collect the intermediate results (df_mess_level) and return a dataframe.

    To answer this: test both approachs:
    - Full polars
    - polars --> duckdb
    """)
    return


if __name__ == "__main__":
    app.run()
