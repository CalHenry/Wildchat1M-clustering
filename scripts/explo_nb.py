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
    ### Few observations on the raw data:
    - conversation_hash is the id variables for the conversations.
    - Messages works in pairs: one **user** message, one **assistant** response. **hashed_ip** + **role** is how to identify one single observation.
    - **'conversations'** contains the messages + meta data that can be duplicates (more like detailled versions) of vars already in the front dataset.

    ### Tidy: manipulation, variable selection

    We want to be at the **message level**.
    A tidy dataset in this context is a row for each message of the conversations, and a variable for each information.
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
    df_prep = (
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
    # Brouillax
    """)
    return


@app.cell
def _(df):
    dff = df.head(30)
    return (dff,)


@app.cell
def _(dff):
    dff.collect_schema()
    return


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


if __name__ == "__main__":
    app.run()
