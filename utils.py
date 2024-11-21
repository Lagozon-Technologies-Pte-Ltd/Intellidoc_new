import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_port = os.getenv("db_port")
from sqlalchemy import create_engine

# Create a SQLAlchemy engine and session factory
engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
Session = sessionmaker(bind=engine)

def insert_feedback(collection_name, single_question, context_str, response, source, feedback_type="user not reacted", feedback = "no feedback received"):
    with Session() as session:
        insert_query = text("""
        INSERT INTO intellidoc_results (collection_name, single_question, context_str, response, source, feedback_type, feedback)
        VALUES (:collection_name, :single_question, :context_str, :response, :source, :feedback_type, :feedback)
        """)
        try:
            session.execute(insert_query, {
                "collection_name": collection_name,
                "single_question": single_question,
                "context_str": context_str,
                "response": response,
                "source": source,
                "feedback_type": feedback_type,
                "feedback": feedback
            })
            session.commit()
            print("Feedback inserted successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            session.rollback()

def load_votes(collection_name):
    votes = {"upvotes": 0, "downvotes": 0}

    with Session() as session:
        execute_query = text("""
            SELECT upvotes, downvotes
            FROM intellidoc_votes
            WHERE collection_name = :collection_name
        """)
        result = session.execute(execute_query, {"collection_name": collection_name}).fetchone()

        if result:
            votes["upvotes"] = result[0]
            votes["downvotes"] = result[1]

    return votes

def save_votes(collection_name, votes):
    with Session() as session:
        execute_query = text("""
        INSERT INTO intellidoc_votes (collection_name, upvotes, downvotes, created_at)
        VALUES (:collection_name, :upvotes, :downvotes, CURRENT_DATE)
        ON CONFLICT (collection_name, created_at)
        DO UPDATE SET
            upvotes = EXCLUDED.upvotes,
            downvotes = EXCLUDED.downvotes;

        """)

        try:
            session.execute(execute_query, {
                "collection_name": collection_name,
                "upvotes": votes["upvotes"],
                "downvotes": votes["downvotes"]
            })
            session.commit()
            print("Votes saved successfully.")

        except Exception as e:
            print(f"An error occurred while saving votes: {e}")
            session.rollback()