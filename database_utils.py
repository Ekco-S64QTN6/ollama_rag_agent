import os
import logging
import re
import uuid
import string
import time
from typing import Optional, List, Dict, Union, Tuple, Any
from sqlalchemy import (
    create_engine, Column, Integer, Text, Boolean, DateTime,
    MetaData, Table, func, ForeignKey, Index, select, text, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy import inspect as sqlalchemy_inspect
from datetime import datetime
import sqlalchemy

# --- Environment Configuration ---
DB_USER = os.getenv('KAIA_DB_USER', 'kaiauser')
DB_PASS = os.getenv('KAIA_DB_PASS', '')
DB_HOST = os.getenv('KAIA_DB_HOST', 'localhost')
DB_NAME = os.getenv('KAIA_DB_NAME', 'kaiadb')
DB_PATH = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

# --- Database Initialization ---
engine = None
Session = None
metadata = MetaData()

# --- Table Definitions ---
users_table = Table(
    'users', metadata,
    Column('user_id', Text, primary_key=True),
    Column('created_at', DateTime(timezone=True), default=func.now())
)

user_preferences_table = Table(
    'user_preferences', metadata,
    Column('preference_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('preference_key', Text, nullable=False),
    Column('preference_value', Text),
    Column('last_updated', DateTime(timezone=True), default=func.now(), onupdate=func.now()),
    UniqueConstraint('user_id', 'preference_key', name='uq_user_preference_key')
)

facts_table = Table(
    'facts', metadata,
    Column('fact_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('fact_text', Text, nullable=False),
    Column('created_at', DateTime(timezone=True), default=func.now()),
    Index('idx_user_fact', 'user_id')
)

interaction_history_table = Table(
    'interaction_history', metadata,
    Column('interaction_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    Column('user_query', Text, nullable=False),
    Column('kaia_response', Text, nullable=False),
    Column('response_type', Text)
)

# --- Natural Language Processing Helpers ---
def normalize_query(query: str) -> Tuple[str, set]:
    """Normalize query for NLP processing"""
    translator = str.maketrans('', '', string.punctuation)
    clean_query = query.lower().translate(translator).strip()
    tokens = set(clean_query.split())
    stopwords = {"what", "do", "you", "me", "my", "the", "a", "an", "is", "are", "show", "list", "about", "tell"}
    keywords = tokens - stopwords
    return clean_query, keywords

def match_query_category(clean_query: str, keywords: set) -> str:
    """Determine query category using enhanced NLP matching"""
    category_patterns = [
        ("about_me", {
            "phrases": [
                "about me", "know about me", "remember about me", "tell me about myself",
                "what know", "what remember", "my information", "my profile", "my data", "what do you know", "list all memories"
            ],
            "keywords": {"myself", "profile", "information", "summary", "overview", "data", "memories"}
        }),
        ("preferences", {
            "phrases": [
                "my preferences", "user preferences", "show preferences", "list preferences",
                "what preferences", "preferences know", "my settings", "user settings", "what options"
            ],
            "keywords": {"preferences", "settings", "options", "theme", "mode", "editor", "favorite"}
        }),
        ("facts", {
            "phrases": [
                "my facts", "remembered facts", "show facts", "list facts",
                "what facts", "facts know", "my information", "stored facts", "what remember"
            ],
            "keywords": {"facts", "information", "remembered", "stored", "knows", "data"}
        }),
        ("history", {
            "phrases": [
                "interaction history", "chat history", "show history", "list history",
                "previous conversations", "past interactions", "our conversations", "my conversations", "list interactions", "what is our interaction history", "what history is known"
            ],
            "keywords": {"history", "interactions", "conversations", "past", "previous", "logs"}
        })
    ]

    for category, pattern in category_patterns:
        if any(phrase in clean_query for phrase in pattern["phrases"]):
            return category

    for category, pattern in category_patterns:
        if keywords & pattern["keywords"]:
            return category

    return "unknown"

# --- Database Operations ---
def initialize_db() -> bool:
    """Initializes the database connection with retry logic"""
    global engine, Session
    logging.info("Initializing PostgreSQL database")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            engine = create_engine(DB_PATH, pool_pre_ping=True)
            Session = sessionmaker(bind=engine)
            metadata.create_all(engine)
            logging.info(f"PostgreSQL database initialized successfully for {DB_PATH}")
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Connection failed (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Could not connect to database after {max_retries} attempts: {e}")
                return False
        except SQLAlchemyError as e:
            logging.error(f"Database initialization error: {e}")
            return False
    return False

def get_session():
    """Returns a new SQLAlchemy session with context manager support"""
    if Session is None or engine is None:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    return Session()

def ensure_user(user_id: str):
    """Ensures a user exists using efficient upsert"""
    session = get_session()
    try:
        stmt = postgresql.insert(users_table).values(user_id=user_id)
        stmt = stmt.on_conflict_do_nothing(index_elements=['user_id'])
        session.execute(stmt)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Error ensuring user '{user_id}': {e}")
    finally:
        session.close()

def handle_memory_storage(user_id: str, content: str) -> Tuple[bool, str]:
    """
    Parses the content (extracted by the LLM) to store facts or preferences in the database.
    Examples of 'content' input:
    "I prefer dark mode" -> preference (key: "dark mode", value: "enabled")
    "my preferred editor is Neovim" -> preference (key: "preferred editor", value: "Neovim")
    "the sky is blue" -> fact
    """
    session = get_session()
    try:
        pref_match = re.match(r"(?:i prefer|my preference is)\s*(.+)", content, re.I)
        if pref_match:
            preference_phrase = pref_match.group(1).strip()
            key = ""
            value = ""
            key_value_match = re.match(r"(.+?)(?:\s+is\s+|=)\s*(.+)", preference_phrase, re.I)
            if key_value_match:
                key = key_value_match.group(1).strip()
                value = key_value_match.group(2).strip()
            else:
                key = preference_phrase
                value = "enabled"

            if not key:
                return False, "Please specify what preference you want me to remember (e.g., 'dark mode' or 'my theme is dark')."

            stmt = postgresql.insert(user_preferences_table).values(
                user_id=user_id,
                preference_key=key,
                preference_value=value
            )
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['user_id', 'preference_key'],
                set_=dict(preference_value=stmt.excluded.preference_value, last_updated=func.now())
            )
            session.execute(on_conflict_stmt)
            session.commit()
            logging.info(f"Preference '{key}: {value}' stored for user '{user_id}'.")
            return True, f"Okay, I'll remember that your preference for '{key}' is '{value}'."

        fact_text = content.strip()
        if not fact_text:
            return False, "Please provide content to remember."

        session.execute(facts_table.insert().values(user_id=user_id, fact_text=fact_text))
        session.commit()
        logging.info(f"Fact '{fact_text}' stored for user '{user_id}'.")
        return True, f"Got it. I'll remember that: {fact_text}."

    except IntegrityError:
        session.rollback()
        return False, "There was a database error storing that. It might be a duplicate."
    except Exception as e:
        session.rollback()
        logging.error(f"Error storing memory: {e}", exc_info=True)
        return False, f"An unexpected error occurred while trying to remember that: {e}"
    finally:
        session.close()

def handle_data_retrieval(user_id: str, query: str) -> Dict[str, Any]:
    """Improved data retrieval with NLP categorization"""
    session = get_session()
    try:
        clean_query, keywords = normalize_query(query)
        category = match_query_category(clean_query, keywords)

        if category == "about_me":
            return get_user_profile(session, user_id)
        elif category == "preferences":
            return get_user_preferences(session, user_id)
        elif category == "facts":
            return get_user_facts(session, user_id)
        elif category == "history":
            return get_interaction_history(session, user_id)
        else:
            return {
                'message': "I can retrieve your preferences, facts, or interaction history. Please be more specific.",
                'data': [],
                'response_type': "unhandled_retrieval_query"
            }

    except SQLAlchemyError as e:
        logging.error(f"Database error during retrieval: {e}", exc_info=True)
        return {
            'message': "A database error occurred while retrieving your information.",
            'data': [],
            'response_type': "retrieval_error"
        }
    except Exception as e:
        logging.error(f"Unexpected retrieval error: {e}", exc_info=True)
        return {
            'message': "An unexpected error occurred while processing your request.",
            'data': [],
            'response_type': "retrieval_error"
        }
    finally:
        session.close()

# --- Modular Data Retrieval Functions ---
def get_user_profile(session, user_id: str) -> Dict[str, Any]:
    """Retrieve combined user profile"""
    all_data = []

    prefs = session.execute(
        select(
            user_preferences_table.c.preference_key,
            user_preferences_table.c.preference_value
        ).filter_by(user_id=user_id)
    ).fetchall()

    if prefs:
        all_data.append("Your preferences:")
        all_data.extend(f"• {key}: {value}" for key, value in prefs)
    else:
        all_data.append("You haven't told me any preferences yet.")

    facts = session.execute(
        select(facts_table.c.fact_text)
        .filter_by(user_id=user_id)
    ).fetchall()

    if facts:
        all_data.append("\nFacts I remember:")
        all_data.extend(f"• {fact[0]}" for fact in facts)
    else:
        all_data.append("\nI haven't stored any facts for you yet.")

    return {
        'message': "Here's what I know about you:",
        'data': all_data,
        'response_type': "user_profile_retrieved"
    }

def get_user_preferences(session, user_id: str) -> Dict[str, Any]:
    """Retrieve user preferences"""
    prefs = session.execute(
        select(
            user_preferences_table.c.preference_key,
            user_preferences_table.c.preference_value
        ).filter_by(user_id=user_id)
    ).fetchall()

    if prefs:
        return {
            'message': "Your preferences:",
            'data': [f"{key}: {value}" for key, value in prefs],
            'response_type': "preferences_retrieved"
        }
    return {
        'message': "You haven't told me any preferences yet.",
        'data': [],
        'response_type': "no_preferences"
    }

def get_user_facts(session, user_id: str) -> Dict[str, Any]:
    """Retrieve user facts"""
    facts = session.execute(
        select(facts_table.c.fact_text)
        .filter_by(user_id=user_id)
    ).fetchall()

    if facts:
        return {
            'message': "Facts I remember:",
            'data': [fact[0] for fact in facts],
            'response_type': "facts_retrieved"
        }
    return {
        'message': "I haven't stored any facts for you yet.",
        'data': [],
        'response_type': "no_facts"
    }

def get_interaction_history(session, user_id: str) -> Dict[str, Any]:
    """Retrieve interaction history"""
    history = session.execute(
        select(
            interaction_history_table.c.timestamp,
            interaction_history_table.c.user_query,
            interaction_history_table.c.kaia_response
        )
        .filter_by(user_id=user_id)
        .order_by(interaction_history_table.c.timestamp.desc())
        .limit(10)
    ).fetchall()

    if history:
        formatted = []
        for timestamp, query, response in history:
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            formatted.append(f"[{time_str}] You: {query} | Kaia: {response}")
        return {
            'message': "Recent interactions:",
            'data': formatted,
            'response_type': "history_retrieved"
        }
    return {
        'message': "No interaction history found.",
        'data': [],
        'response_type': "no_history"
    }

def log_interaction(user_id: str, user_query: str, kaia_response: str, response_type: str):
    """Logs user interaction with Kaia."""
    session = get_session()
    try:
        session.execute(interaction_history_table.insert().values(
            user_id=user_id,
            user_query=user_query,
            kaia_response=kaia_response,
            response_type=response_type
        ))
        session.commit()
        logging.info(f"Interaction logged for user '{user_id}'.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error logging interaction for user '{user_id}': {e}", exc_info=True)
    finally:
        session.close()

# --- Database Status Check ---
def get_database_status() -> Dict:
    """Get database connection status"""
    if not engine:
        return {'connected': False, 'tables': []}

    try:
        inspector = sqlalchemy_inspect(engine)
        return {
            'connected': True,
            'tables': inspector.get_table_names()
        }
    except Exception as e:
        logging.error(f"Error getting database status: {e}", exc_info=True)
        return {
            'connected': False,
            'error': str(e)
        }

# --- User Identification ---
def get_current_user() -> str:
    """Get current OS user as default ID"""
    return os.getlogin() or "default_user"
