from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
import time
from picasso.localize import _db_filename


def fetch_db():
    """Load the local database and return the localized files.

    Returns
    -------
    df : pd.DataFrame
        The ``files`` table, sorted by acquisition date; empty if the table
        does not exist yet.
    """
    try:
        DB_PATH = "sqlite:///" + _db_filename()
        engine = create_engine(DB_PATH, echo=False)
        df = pd.read_sql_table("files", con=engine)

        df = df.sort_values("file_created")
    except ValueError:
        df = pd.DataFrame()

    return df


def fetch_watcher():
    """Load the local database and return the running watchers.

    Returns
    -------
    df : pd.DataFrame
        The ``watcher`` table; empty if it does not exist yet.
    """
    try:
        engine = create_engine("sqlite:///" + _db_filename(), echo=False)
        df = pd.read_sql_table("watcher", con=engine)
    except ValueError as e:
        print(e)
        df = pd.DataFrame()

    return df


def refresh(to_wait: int):
    """Wait for a given amount of time and then stop streamlit.

    Parameters
    ----------
    to_wait : int
        Seconds to count down before rerunning the page.
    """
    ref = st.empty()
    for i in range(to_wait):
        ref.write(f"Refreshing in {to_wait-i} s")
        time.sleep(1)
    st.stop()
