import logging
from typing import Optional, List, Dict, Union
from sqlalchemy import create_engine, Column, Integer, Text, Boolean, DateTime, MetaData, Table, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql # Import postgresql dialects for specific clauses
from sqlalchemy.exc import IntegrityError, OperationalError
from datetime import datetime
import sqlalchemy # Import for get_database_status to work with inspector

# --- Database Initialization ---
engine = None
Session = None
metadata = MetaData()

# Define table schemas with type hints
user_preferences_table = Table(
    'user_preferences', metadata,
    Column('preference_id', Integer, primary_key=True),
    Column('user_id', Text, nullable=False, default='default_user'),
    Column('preference_key', Text, nullable=False),
    Column('preference_value', Text),
    Column('last_updated', DateTime(timezone=True), default=func.now()),
    # Removed sqlite_autoincrement=True
)

facts_table = Table(
    'facts', metadata,
    Column('fact_id', Integer, primary_key=True),
    Column('fact_text', Text, nullable=False),
    Column('source', Text, default='user_input'),
    Column('context', Text, default='general'),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    # Removed sqlite_autoincrement=True
)

interaction_history_table = Table(
    'interaction_history', metadata,
    Column('interaction_id', Integer, primary_key=True),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    Column('user_query', Text, nullable=False),
    Column('kaia_response', Text, nullable=False),
    Column('response_type', Text, default='chat'),
    # Removed sqlite_autoincrement=True
)

tools_table = Table(
    'tools', metadata,
    Column('tool_id', Integer, primary_key=True),
    Column('tool_name', Text, nullable=False, unique=True),
    Column('tool_description', Text),
    Column('tool_function_name', Text),
    Column('is_enabled', Boolean, default=True),
    Column('last_modified', DateTime(timezone=True), default=func.now()),
    # Removed sqlite_autoincrement=True
)

kaia_persona_details_table = Table(
    'kaia_persona_details', metadata,
    Column('detail_id', Integer, primary_key=True),
    Column('detail_key', Text, nullable=False, unique=True),
    Column('detail_value', Text, nullable=False),
    Column('last_updated', DateTime(timezone=True), default=func.now()),
    # Removed sqlite_autoincrement=True
)

def initialize_db(database_url: str) -> bool:
    """Initialize the database engine and session factory.

    Args:
        database_url: Connection string for the database

    Returns:
        bool: True if successful, False otherwise
    """
    global engine, Session
    try:
        engine = create_engine(database_url)
        # Verify connection by attempting to connect
        with engine.connect() as connection:
            connection.execute(func.now())
        Session = sessionmaker(bind=engine)
        metadata.create_all(engine)  # Create tables if they don't exist
        logging.info(f"PostgreSQL database initialized successfully for {database_url}")
        return True
    except OperationalError as e:
        logging.error(f"PostgreSQL database connection failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during PostgreSQL initialization: {e}")
        return False

def get_session():
    """Get a new database session."""
    if not Session:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    return Session()

# --- Core CRUD Operations ---

def insert_default_persona_details() -> bool:
    """Insert default persona details if they don't exist."""
    default_details = [
        {"detail_key": "pet_name", "detail_value": "Pixel"},
        {"detail_key": "favorite_music_genre", "detail_value": "Jazz"},
        {"detail_key": "core_philosophy", "detail_value": "Logic, verifiable data, and clear causality"},
        {"detail_key": "sarcasm_level", "detail_value": "dry, often sarcastic wit"},
        {"detail_key": "favorite_operating_system", "detail_value": "Arch Linux"},
        {"detail_key": "cpu_type", "detail_value": "AMD Ryzen"},
        {"detail_key": "motherboard_model", "detail_value": "ROG STRIX B650-A GAMING WIFI"},
    ]

    session = get_session()
    try:
        for detail in default_details:
            # Use PostgreSQL's ON CONFLICT DO NOTHING for upsert
            insert_stmt = postgresql.insert(kaia_persona_details_table).values(**detail)
            on_conflict_stmt = insert_stmt.on_conflict_do_nothing(
                index_elements=['detail_key']
            )
            session.execute(on_conflict_stmt)
        session.commit()
        logging.info("Default persona details inserted/skipped.")
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to insert persona details: {e}")
        return False
    finally:
        session.close()

def log_interaction(user_query: str, kaia_response: str, response_type: str = "chat") -> bool:
    """Log an interaction to the database."""
    session = get_session()
    try:
        stmt = interaction_history_table.insert().values(
            user_query=user_query,
            kaia_response=kaia_response,
            response_type=response_type
        )
        session.execute(stmt)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to log interaction: {e}")
        return False
    finally:
        session.close()

def store_fact(fact_text: str, source: str = "user_input", context: str = None) -> Optional[int]:
    """Store a new fact and return its ID."""
    session = get_session()
    try:
        stmt = facts_table.insert().values(
            fact_text=fact_text,
            source=source,
            context=context
        ).returning(facts_table.c.fact_id)
        result = session.execute(stmt)
        session.commit()
        return result.scalar_one_or_none() # Use scalar_one_or_none for consistent behavior
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to store fact: {e}")
        return None
    finally:
        session.close()

def set_user_preference(user_id: str, preference_key: str, preference_value: str) -> bool:
    """Set or update a user preference."""
    session = get_session()
    try:
        # Use PostgreSQL's ON CONFLICT (UPSERT)
        insert_stmt = postgresql.insert(user_preferences_table).values(
            user_id=user_id,
            preference_key=preference_key,
            preference_value=preference_value
        )
        on_conflict_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['user_id', 'preference_key'],
            set_={'preference_value': insert_stmt.excluded.preference_value,
                  'last_updated': func.now()}
        )
        session.execute(on_conflict_stmt)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to set preference: {e}")
        return False
    finally:
        session.close()

# --- Data Retrieval Functions ---

def get_persona_detail(detail_key: str) -> Optional[str]:
    """Get a specific persona detail."""
    session = get_session()
    try:
        stmt = kaia_persona_details_table.select().where(
            kaia_persona_details_table.c.detail_key == detail_key
        )
        result = session.execute(stmt).fetchone()
        return result.detail_value if result else None
    except Exception as e:
        logging.error(f"Failed to get persona detail: {e}")
        return None
    finally:
        session.close()

def get_all_facts() -> List[Dict]:
    """Get all facts ordered by timestamp."""
    session = get_session()
    try:
        stmt = facts_table.select().order_by(facts_table.c.timestamp.desc())
        return [dict(row._mapping) for row in session.execute(stmt).fetchall()] # Use _mapping for dict conversion
    except Exception as e:
        logging.error(f"Failed to get facts: {e}")
        return []
    finally:
        session.close()

def get_user_preference(user_id: str, preference_key: str) -> Optional[str]:
    """Get a specific user preference."""
    session = get_session()
    try:
        stmt = user_preferences_table.select().where(
            user_preferences_table.c.user_id == user_id,
            user_preferences_table.c.preference_key == preference_key
        )
        result = session.execute(stmt).fetchone()
        return result.preference_value if result else None
    except Exception as e:
        logging.error(f"Failed to get preference: {e}")
        return None
    finally:
        session.close()

def get_all_user_preferences(user_id: str) -> List[Dict]:
    """Get all preferences for a user."""
    session = get_session()
    try:
        stmt = user_preferences_table.select().where(
            user_preferences_table.c.user_id == user_id
        ).order_by(user_preferences_table.c.preference_key)
        return [dict(row._mapping) for row in session.execute(stmt).fetchall()] # Use _mapping for dict conversion
    except Exception as e:
        logging.error(f"Failed to get preferences: {e}")
        return []
    finally:
        session.close()

# --- New Functions ---

def handle_memory_storage(query: str) -> bool:
    """Handle memory/preference storage requests.

    Args:
        query: The user's input query

    Returns:
        bool: True if the query was handled as a storage request, False otherwise
    """
    query_lower = query.lower().strip()
    user_id = "default_user"

    # Handle "remember that" pattern
    if query_lower.startswith("remember that"):
        fact_text = query.split("remember that", 1)[1].strip()
        if fact_text:
            store_fact(fact_text)
            logging.info(f"Fact stored: {fact_text}")
            return True
        else:
            logging.warning("Attempted to store an empty fact.")
            return False


    # Handle preference patterns
    preference_patterns = {
        "favorite color is": "favorite_color",
        "default editor is": "default_editor",
        "preferred output method is": "output_method",
        "my pet's name is": "pet_name" # Added from persona details
    }

    for phrase, key in preference_patterns.items():
        if phrase in query_lower:
            value = query.split(phrase, 1)[1].strip()
            if value:
                set_user_preference(user_id, key, value)
                logging.info(f"Preference set: {key} = {value}")
                return True
            else:
                logging.warning(f"Attempted to set empty value for preference: {key}")
                return False

    return False


def handle_data_retrieval(query: str) -> Dict[str, Union[str, List[Dict]]]:
    """Handle data retrieval requests.

    Args:
        query: The user's input query

    Returns:
        dict: {
            'message': str response,
            'data': list of results (if applicable),
            'response_type': str category
        }
    """
    query_lower = query.lower()
    user_id = "default_user"

    # Check for specific retrieval patterns
    if "all facts" in query_lower or "what facts" in query_lower or "list all facts" in query_lower:
        facts = get_all_facts()
        if facts:
            # Format facts for display
            formatted_facts = []
            for fact in facts:
                fact_id = fact.get('fact_id', 'N/A')
                fact_text = fact.get('fact_text', 'No text')
                source = fact.get('source', 'Unknown source')
                timestamp = fact.get('timestamp', 'Unknown time').strftime("%Y-%m-%d %H:%M:%S") if isinstance(fact.get('timestamp'), datetime) else 'Unknown time'
                formatted_facts.append(f"ID: {fact_id}, Fact: \"{fact_text}\", Source: {source}, Time: {timestamp}")
            return {
                'message': "Here are the stored facts:",
                'data': formatted_facts, # Return formatted strings in data
                'response_type': "facts_retrieved"
            }
        return {
            'message': "No facts are currently stored.",
            'data': [],
            'response_type': "facts_retrieved"
        }

    if "preferences" in query_lower or "my preferences" in query_lower or "user preferences" in query_lower:
        prefs = get_all_user_preferences(user_id)
        if prefs:
            formatted_prefs = []
            for pref in prefs:
                key = pref.get('preference_key', 'N/A')
                value = pref.get('preference_value', 'No value')
                formatted_prefs.append(f"{key.replace('_', ' ').title()}: {value}")
            return {
                'message': "Here are your stored preferences:",
                'data': formatted_prefs, # Return formatted strings in data
                'response_type': "preferences_retrieved"
            }
        return {
            'message': "No preferences are currently stored.",
            'data': [],
            'response_type': "preferences_retrieved"
        }

    # Check for specific persona details
    # The persona details are now stored in the kaia_persona_details_table
    # The existing persona_detail action in llamaindex_ollama_rag.py should handle this,
    # but for handle_data_retrieval specifically, we can also check.
    persona_detail_keywords = {
        "my pet's name": "pet_name",
        "favorite music genre": "favorite_music_genre",
        "core philosophy": "core_philosophy",
        "sarcasm level": "sarcasm_level",
        "favorite operating system": "favorite_operating_system",
        "cpu type": "cpu_type",
        "motherboard model": "motherboard_model",
        "my persona details": "all_persona_details" # A new keyword to retrieve all
    }

    for phrase, key in persona_detail_keywords.items():
        if phrase in query_lower:
            if key == "all_persona_details":
                session = get_session()
                try:
                    stmt = kaia_persona_details_table.select().order_by(kaia_persona_details_table.c.detail_key)
                    all_details = [dict(row._mapping) for row in session.execute(stmt).fetchall()]
                    if all_details:
                        formatted_details = []
                        for detail in all_details:
                            formatted_details.append(f"{detail['detail_key'].replace('_', ' ').title()}: {detail['detail_value']}")
                        return {
                            'message': "Here are my persona details:",
                            'data': formatted_details,
                            'response_type': "persona_details_retrieved"
                        }
                    else:
                        return {
                            'message': "No persona details are currently stored.",
                            'data': [],
                            'response_type': "persona_details_retrieved"
                        }
                finally:
                    session.close()
            else:
                value = get_persona_detail(key)
                if value:
                    return {
                        'message': f"My {phrase} is {value}.",
                        'data': [{'key': key, 'value': value}],
                        'response_type': "persona_detail_retrieved"
                    }
                return {
                    'message': f"I don't have information about my {phrase} stored.",
                    'data': [],
                    'response_type': "persona_detail_not_found"
                }


    # Default response if no patterns matched
    return {
        'message': "I couldn't determine what specific data you wanted to retrieve.",
        'data': [],
        'response_type': "data_retrieval_failed"
    }

# --- Utility Functions ---

def get_database_status() -> Dict:
    """Get database status information."""
    if not engine:
        return {
            'connected': False,
            'tables': []
        }
    session = get_session()
    try:
        inspector = sqlalchemy.inspect(engine)
        tables = inspector.get_table_names()
        return {
            'connected': True,
            'tables': tables
        }
    except Exception as e:
        logging.error(f"Error getting database status: {e}")
        return {
            'connected': False,
            'tables': [],
            'error': str(e)
        }
    finally:
        session.close()
