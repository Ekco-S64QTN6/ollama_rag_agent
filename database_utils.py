import os
import logging
import re
import time
import string
from typing import Optional, List, Dict, Union, Tuple, Any
from sqlalchemy import (
    create_engine, Column, Integer, Text, DateTime,
    MetaData, Table, func, ForeignKey, Index, select, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy import inspect as sqlalchemy_inspect
from datetime import datetime

# --- Environment Configuration ---
# Establishes database connection parameters, using environment variables with sensible defaults.
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
# Defines the database schema using SQLAlchemy's Table objects for clarity and maintainability.

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
    """
    Cleans and tokenizes a query for NLP matching.
    Removes punctuation, converts to lowercase, and splits into a set of keywords, ignoring common stopwords.
    """
    translator = str.maketrans('', '', string.punctuation)
    clean_query = query.lower().translate(translator).strip()
    tokens = set(clean_query.split())
    stopwords = {"what", "do", "you", "me", "my", "the", "a", "an", "is", "are", "show", "list", "about", "tell"}
    keywords = tokens - stopwords
    return clean_query, keywords

def match_query_category(clean_query: str, keywords: set) -> str:
    """
    Determines the sub-category of a data retrieval query using keyword and phrase matching.
    This function is called *after* the main LLM has already classified the intent as "retrieve_data".
    """
    # More specific patterns are checked first to avoid incorrect broad matches.
    category_patterns = [
        ("about_me", {
            "phrases": ["about me", "know about me", "what do you know", "list all memories"],
            "keywords": {"myself", "profile", "summary", "overview", "memories"}
        }),
        ("preferences", {
            "phrases": ["my preferences", "show preferences", "list preferences"],
            "keywords": {"preferences", "settings", "options", "editor", "favorite"}
        }),
        ("facts", {
            "phrases": ["my facts", "show facts", "list facts"],
            "keywords": {"facts", "remembered", "stored"}
        }),
        ("history", {
            "phrases": ["interaction history", "chat history", "show history", "list history", "list interactions"],
            "keywords": {"history", "interactions", "conversations", "past", "logs"}
        })
    ]

    for category, pattern in category_patterns:
        if any(phrase in clean_query for phrase in pattern["phrases"]) or keywords & pattern["keywords"]:
            return category

    return "unknown"

# --- Database Operations ---
def initialize_db() -> bool:
    """Initializes the database connection and creates tables if they don't exist, with retry logic."""
    global engine, Session
    logging.info("Initializing PostgreSQL database")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            engine = create_engine(DB_PATH, pool_pre_ping=True)
            metadata.create_all(engine) # This is idempotent and safe to run on every startup.
            Session = sessionmaker(bind=engine)
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
    """Provides a new SQLAlchemy session."""
    if Session is None:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    return Session()

def ensure_user(user_id: str):
    """Ensures a user record exists in the database using an efficient 'INSERT ... ON CONFLICT DO NOTHING'."""
    with get_session() as session:
        try:
            stmt = postgresql.insert(users_table).values(user_id=user_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=['user_id'])
            session.execute(stmt)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error ensuring user '{user_id}': {e}")

def handle_memory_storage(user_id: str, content: str) -> Tuple[bool, str]:
    """
    Parses content from the LLM to store facts or preferences.
    Uses regex to differentiate between a preference (e.g., "I prefer X") and a general fact.
    """
    with get_session() as session:
        try:
            # Regex to capture preferences like "I prefer dark mode" or "my preferred editor is Neovim"
            pref_match = re.match(r"(?:i prefer|my preference is|my preferred\s\w+\s?is)\s*(.+)", content, re.I)
            if pref_match:
                preference_phrase = pref_match.group(1).strip()
                # Attempt to split into a key-value pair
                key_value_match = re.match(r"(.+?)(?:\s+is\s+|=)\s*(.+)", preference_phrase, re.I)
                if key_value_match:
                    key = key_value_match.group(1).strip()
                    value = key_value_match.group(2).strip()
                else: # If no explicit value, treat the phrase as the key and assume "enabled"
                    key = preference_phrase
                    value = "enabled"

                if not key:
                    return False, "Could not determine the preference key. Please be more specific."

                # Upsert the preference for the user
                stmt = postgresql.insert(user_preferences_table).values(
                    user_id=user_id, preference_key=key, preference_value=value
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['user_id', 'preference_key'],
                    set_=dict(preference_value=stmt.excluded.preference_value, last_updated=func.now())
                )
                session.execute(on_conflict_stmt)
                session.commit()
                logging.info(f"Preference '{key}: {value}' stored for user '{user_id}'.")
                return True, f"Okay, I'll remember that your preference for '{key}' is '{value}'."

            # If it's not a preference, store it as a general fact.
            fact_text = content.strip()
            if not fact_text:
                return False, "Please provide content to remember."

            session.execute(facts_table.insert().values(user_id=user_id, fact_text=fact_text))
            session.commit()
            logging.info(f"Fact '{fact_text}' stored for user '{user_id}'.")
            return True, f"Got it. I'll remember that: {fact_text}."

        except IntegrityError as e:
            session.rollback()
            logging.error(f"Database integrity error during memory storage: {e}")
            return False, "There was a database conflict. It's possible that information already exists."
        except Exception as e:
            session.rollback()
            logging.error(f"Error storing memory: {e}", exc_info=True)
            return False, f"An unexpected error occurred while trying to remember that."

def handle_data_retrieval(user_id: str, query: str) -> Dict[str, Any]:
    """Routes a data retrieval query to the appropriate function based on NLP sub-categorization."""
    with get_session() as session:
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
                # Fallback if sub-categorization fails
                return {
                    'message': "I can retrieve your preferences, facts, or interaction history. Please be more specific.",
                    'data': [], 'response_type': "unhandled_retrieval_query"
                }
        except Exception as e:
            logging.error(f"Unexpected retrieval error: {e}", exc_info=True)
            return {'message': "An unexpected error occurred processing your request.", 'data': [], 'response_type': "retrieval_error"}

# --- Modular Data Retrieval Functions ---
def get_user_profile(session, user_id: str) -> Dict[str, Any]:
    """Retrieves a combined summary of a user's stored preferences and facts."""
    all_data = []
    prefs = session.execute(select(user_preferences_table.c.preference_key, user_preferences_table.c.preference_value).filter_by(user_id=user_id)).fetchall()
    if prefs:
        all_data.append("Your preferences:")
        all_data.extend([f"• {key}: {value}" for key, value in prefs])

    facts = session.execute(select(facts_table.c.fact_text).filter_by(user_id=user_id)).fetchall()
    if facts:
        if all_data: all_data.append("\nFacts I remember:")
        else: all_data.append("Facts I remember:")
        all_data.extend([f"• {fact[0]}" for fact in facts])

    if not all_data:
        return {'message': "I don't have any preferences or facts stored for you yet.", 'data': [], 'response_type': "no_profile_data"}

    return {'message': "Here's what I know about you:", 'data': all_data, 'response_type': "user_profile_retrieved"}

def get_user_preferences(session, user_id: str) -> Dict[str, Any]:
    """Retrieves only the user's preferences."""
    prefs = session.execute(select(user_preferences_table.c.preference_key, user_preferences_table.c.preference_value).filter_by(user_id=user_id)).fetchall()
    if prefs:
        return {'message': "Your preferences:", 'data': [f"{key}: {value}" for key, value in prefs], 'response_type': "preferences_retrieved"}
    return {'message': "You haven't told me any preferences yet.", 'data': [], 'response_type': "no_preferences"}

def get_user_facts(session, user_id: str) -> Dict[str, Any]:
    """Retrieves only the user's stored facts."""
    facts = session.execute(select(facts_table.c.fact_text).filter_by(user_id=user_id)).fetchall()
    if facts:
        return {'message': "Facts I remember:", 'data': [fact[0] for fact in facts], 'response_type': "facts_retrieved"}
    return {'message': "I haven't stored any facts for you yet.", 'data': [], 'response_type': "no_facts"}

def get_interaction_history(session, user_id: str, limit: int = 10) -> Dict[str, Any]:
    """Retrieves the most recent interaction history for the user."""
    history = session.execute(
        select(interaction_history_table.c.timestamp, interaction_history_table.c.user_query, interaction_history_table.c.kaia_response)
        .filter_by(user_id=user_id).order_by(interaction_history_table.c.timestamp.desc()).limit(limit)
    ).fetchall()
    if history:
        formatted = [f"[{ts.strftime('%Y-%m-%d %H:%M')}] You: {q} | Kaia: {r[:60]}..." for ts, q, r in history]
        return {'message': "Recent interactions:", 'data': formatted, 'response_type': "history_retrieved"}
    return {'message': "No interaction history found.", 'data': [], 'response_type': "no_history"}

def log_interaction(user_id: str, user_query: str, kaia_response: str, response_type: str):
    """Logs a user-Kaia interaction to the database."""
    with get_session() as session:
        try:
            session.execute(interaction_history_table.insert().values(
                user_id=user_id, user_query=user_query, kaia_response=kaia_response, response_type=response_type
            ))
            session.commit()
            logging.info(f"Interaction logged for user '{user_id}'.")
        except Exception as e:
            session.rollback()
            logging.error(f"Error logging interaction for user '{user_id}': {e}", exc_info=True)

# --- Database Status and User Identification ---
def get_database_status() -> Dict:
    """Checks the database connection status and lists available tables."""
    if not engine:
        return {'connected': False, 'error': 'Engine not initialized', 'tables': []}
    try:
        with engine.connect() as connection:
            inspector = sqlalchemy_inspect(engine)
            return {'connected': True, 'tables': inspector.get_table_names()}
    except Exception as e:
        logging.error(f"Error getting database status: {e}", exc_info=True)
        return {'connected': False, 'error': str(e), 'tables': []}

def get_current_user() -> str:
    """Gets the current OS username to identify the user, with a fallback."""
    try:
        return os.getlogin()
    except (OSError, AttributeError):
        return "default_user"
