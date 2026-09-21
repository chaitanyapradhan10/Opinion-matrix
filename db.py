# db.py
import streamlit as st
import pymysql
import tempfile
import os

# ----- Lazy connection (only created when first needed) -----
_connection = None

def get_connection():
    """Create and return a MySQL connection using Streamlit secrets."""
    global _connection
    if _connection is not None:
        return _connection

    mysql_secrets = st.secrets["connections"]["mysql"]

    # Handle SSL certificate if present
    ssl_dict = None
    cert_content = mysql_secrets.get("ssl_ca")
    if cert_content:
        # Write certificate to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(cert_content)
            cert_path = f.name
        ssl_dict = {"ca": cert_path}

    try:
        _connection = pymysql.connect(
            host=mysql_secrets["host"],
            user=mysql_secrets["username"],
            password=mysql_secrets["password"],
            database=mysql_secrets["database"],
            port=int(mysql_secrets["port"]),
            ssl=ssl_dict,           # ✅ now using SSL
            autocommit=True,
            connect_timeout=10,
        )
        return _connection
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        raise   # re-raise so the app knows

def save_prediction(review, sentiment, probabilities):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO Prediction
    (review, sentiment, prob_positive, prob_neutral, prob_negative, correct_sentiment)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        sql,
        (
            review,
            sentiment,
            probabilities.get("positive", 0),
            probabilities.get("neutral", 0),
            probabilities.get("negative", 0),
            None   # User can update later
        ),
    )
    cursor.close()
    # No need to commit because autocommit=True