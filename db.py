import streamlit as st
import pymysql

# ----- Load connection parameters from Streamlit secrets -----
mysql_secrets = st.secrets["connections"]["mysql"]

connection = pymysql.connect(
    host=mysql_secrets["host"],
    user=mysql_secrets["username"],
    password=mysql_secrets["password"],
    database=mysql_secrets["database"],
    port=int(mysql_secrets["port"]),
    autocommit=True,
    # If you need SSL and have the certificate embedded, you can add:
    ssl={'ca': mysql_secrets.get("ssl_ca")}
)

def save_prediction(review, sentiment, probabilities):
    cursor = connection.cursor()

    sql = """
    INSERT INTO Prediction
    (
        review,
        sentiment,
        prob_positive,
        prob_neutral,
        prob_negative,
        correct_sentiment
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        sql,
        (
            review,
            sentiment,
            probabilities.get("Positive", 0),
            probabilities.get("Neutral", 0),
            probabilities.get("Negative", 0),
            None   # User can update later
        ),
    )

    cursor.close()