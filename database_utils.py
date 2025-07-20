import os
import logging
from typing import Optional, List, Dict, Union
from sqlalchemy import (
    create_engine, Column, Integer, Text, Boolean, DateTime,
    MetaData, Table, func, ForeignKey, Index, select
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
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
    Column('last_updated', DateTime(timezone=True), default=func.now())
)

facts_table = Table(
    'facts', metadata,
    Column('fact_id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False, default='system'),
    Column('fact_text', Text, nullable=False),
    Column('source', Text, default='user_input'),
    Column('context', Text, default='general'),
    Column('timestamp', DateTime(timezone=True), default=func.now())
)

interaction_history_table = Table(
    'interaction_history', metadata,
    Column('interaction_id', Integer, primary_key=True),
    Column('timestamp', DateTime(timezone=True), default=func.now()),
    Column('user_id', Text, ForeignKey('users.user_id'), nullable=False),
    Column('user_query', Text, nullable=False),
    Column('kaia_response', Text, nullable=False),
    Column('response_type', Text, nullable=False),
    Column('action_details', Text)
)

# --- Indexes for Performance ---
Index('idx_interaction_timestamp', interaction_history_table.c.timestamp)
Index('idx_facts_context', facts_table.c.context)
Index('idx_user_preferences_user_key',
      user_preferences_table.c.user_id,
      user_preferences_table.c.preference_key,
      unique=True)

# --- Database Initialization Function ---
def initialize_db():
    """Initialize database connection and create tables"""
    global engine, Session
    try:
        engine = create_engine(
            DB_PATH,
            pool_size=10,
            max_overflow=2,
            pool_recycle=300
        )
        metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        logging.info(f"Database initialized: {DB_HOST}/{DB_NAME}")
        return True
    except OperationalError as e:
        logging.error(f"Database connection failed: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Unexpected database error: {e}", exc_info=True)
        return False

# --- User Management ---
def get_session():
    """Get a new database session"""
    if not Session:
        raise RuntimeError("Database not initialized. Call initialize_db() first.")
    return Session()

def ensure_user(user_id: str):
    """Ensure user exists in database"""
    session = get_session()
    try:
        # Insert user if not exists
        stmt = postgresql.insert(users_table).values(user_id=user_id)
        stmt = stmt.on_conflict_do_nothing(index_elements=['user_id'])
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error ensuring user exists: {e}", exc_info=True)
    finally:
        session.close()

# --- CRUD Operations ---
def log_interaction(user_id: str, user_query: str, kaia_response: str,
                   response_type: str, action_details: str = None) -> bool:
    """Log an interaction with user reference"""
    ensure_user(user_id)
    session = get_session()
    try:
        stmt = postgresql.insert(interaction_history_table).values(
            user_id=user_id,
            user_query=user_query,
            kaia_response=kaia_response,
            response_type=response_type,
            action_details=action_details
        )
        session.execute(stmt)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error logging interaction: {e}", exc_info=True)
        return False
    finally:
        session.close()

def set_user_preference(user_id: str, key: str, value: str) -> bool:
    """Set user preference with user reference"""
    ensure_user(user_id)
    session = get_session()
    try:
        stmt = postgresql.insert(user_preferences_table).values(
            user_id=user_id,
            preference_key=key,
            preference_value=value
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['user_id', 'preference_key'],
            set_={'preference_value': value, 'last_updated': func.now()}
        )
        session.execute(stmt)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error setting preference: {e}", exc_info=True)
        return False
    finally:
        session.close()

def get_user_preference(user_id: str, key: str) -> Optional[str]:
    """Get user preference"""
    session = get_session()
    try:
        stmt = select(user_preferences_table.c.preference_value).where(
            user_preferences_table.c.user_id == user_id,
            user_preferences_table.c.preference_key == key
        )
        result = session.execute(stmt).scalar_one_or_none()
        return result
    except Exception as e:
        logging.error(f"Error getting preference: {e}", exc_info=True)
        return None
    finally:
        session.close()

def add_fact(fact_text: str, user_id: str = 'system',
            source: str = 'user_input', context: str = 'general') -> bool:
    """Add a new fact to knowledge base"""
    ensure_user(user_id)
    session = get_session()
    try:
        stmt = postgresql.insert(facts_table).values(
            user_id=user_id,
            fact_text=fact_text,
            source=source,
            context=context
        )
        session.execute(stmt)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Error adding fact: {e}", exc_info=True)
        return False
    finally:
        session.close()

def get_facts(context: str = None, user_id: str = None) -> List[Dict]:
    """Retrieve facts with optional context filter"""
    session = get_session()
    try:
        stmt = select(facts_table)
        if context:
            stmt = stmt.where(facts_table.c.context == context)
        if user_id:
            stmt = stmt.where(facts_table.c.user_id == user_id)
        results = session.execute(stmt)
        return [dict(row._mapping) for row in results]  # Return full dictionaries
    except Exception as e:
        logging.error(f"Error getting facts: {e}", exc_info=True)
        return []
    finally:
        session.close()

# --- Data Retrieval Handler ---
def handle_data_retrieval(query: Union[str, List[str]], user_id: str = 'default_user') -> Dict:
    """
    Handle data retrieval requests focusing on user-specific data
    Persona details are no longer stored in database
    """
    # Normalize input
    if isinstance(query, list):
        query = " ".join(query)
        logging.warning("Converted list input to string for data retrieval")
    query_lower = query.lower()

    # Handle persona detail queries
    persona_phrases = [
        "who are you", "what is your name", "describe yourself",
        "your role", "your personality", "your philosophy",
        "your response style", "your constraints", "your capabilities",
        "topics to avoid", "all persona details", "your persona"
    ]
    for phrase in persona_phrases:
        if phrase in query_lower:
            return {
                'message': "My persona details and core identity are defined in my source Markdown files and are not stored in this database.",
                'data': [],
                'response_type': "persona_info_file_based"
            }

    # Handle facts retrieval
    if "fact" in query_lower or "remember" in query_lower:
        facts = get_facts(user_id=user_id)
        if facts:
            return {
                'message': "Here are stored facts:",
                'data': facts,  # Return full dictionaries
                'response_type': "facts_retrieved"
            }
        return {
            'message': "No facts currently stored",
            'data': [],
            'response_type': "no_facts"
        }

    # Handle preferences retrieval
    if "preference" in query_lower or "setting" in query_lower:
        session = get_session()
        try:
            stmt = select(user_preferences_table).where(
                user_preferences_table.c.user_id == user_id
            )
            prefs = session.execute(stmt)
            prefs_list = [dict(row._mapping) for row in prefs]  # Return full dictionaries
            if prefs_list:
                return {
                    'message': "Your preferences:",
                    'data': prefs_list,
                    'response_type': "preferences_retrieved"
                }
        except Exception as e:
            logging.error(f"Error retrieving preferences: {e}", exc_info=True)
        finally:
            session.close()
        return {
            'message': "No preferences stored",
            'data': [],
            'response_type': "no_preferences"
        }

    # Default response
    return {
        'message': "Retrieve data using: 'list my preferences' or 'show facts'",
        'data': [],
        'response_type': "data_retrieval_instructions"
    }

# --- Database Status Check ---
def get_database_status() -> Dict:
    """Get database connection status"""
    if not engine:
        return {'connected': False, 'tables': []}

    try:
        inspector = sqlalchemy.inspect(engine)
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
