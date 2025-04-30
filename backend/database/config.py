from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Global client instances
qa_mongo_client: Optional[AsyncIOMotorClient] = None
automation_mongo_client: Optional[AsyncIOMotorClient] = None
qa_evaluations: Optional[AsyncIOMotorCollection] = None
eval_collection: Optional[AsyncIOMotorCollection] = None

_is_initialized = False

def init_qa_db() -> Tuple[AsyncIOMotorClient, AsyncIOMotorCollection]:
    """Initialize QA Platform database connection"""
    mongodb_url = os.getenv("MONGODB_URL")
    if not mongodb_url:
        raise ValueError("MONGODB_URL environment variable is not set")
        
    client = AsyncIOMotorClient(mongodb_url)
    qa_db = client.qa_platform
    evaluations = qa_db.qa_evaluations
    logger.info("Successfully connected to QA Platform database")
    return client, evaluations

def init_eval_db() -> Tuple[Optional[AsyncIOMotorClient], Optional[AsyncIOMotorCollection]]:
    """Initialize Automation database connection"""
    eval_mongo_uri = os.getenv('EVAL_MONGO_URI')
    if not eval_mongo_uri:
        logger.warning("EVAL_MONGO_URI environment variable is not set. Automated QA features will be disabled.")
        return None, None
        
    client = AsyncIOMotorClient(
        eval_mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        minPoolSize=2,
        connectTimeoutMS=0
    )
    eval_db = client['asc-fin-data']
    evaluations = eval_db['evaluations']
    logger.info("Successfully connected to Automation database")
    return client, evaluations

def init_db_connections():
    """Initialize all database connections"""
    global qa_mongo_client, automation_mongo_client, qa_evaluations, eval_collection, _is_initialized
    
    if _is_initialized:
        return

    # Initialize QA Platform database
    try:
        qa_mongo_client, qa_evaluations = init_qa_db()
    except Exception as e:
        logger.error(f"Error connecting to QA Platform database: {str(e)}")
        raise

    # Initialize Automation database
    try:
        automation_mongo_client, eval_collection = init_eval_db()
    except Exception as e:
        logger.error(f"Error connecting to Automation database: {str(e)}")
        # Don't raise here as automated QA is optional

    _is_initialized = True

def get_qa_collection() -> AsyncIOMotorCollection:
    """Get the QA evaluations collection, initializing if necessary"""
    global qa_evaluations
    if not _is_initialized:
        init_db_connections()
    if qa_evaluations is None:
        raise ValueError("Failed to initialize QA database connection")
    return qa_evaluations

def get_eval_collection() -> Optional[AsyncIOMotorCollection]:
    """Get the automated evaluations collection, initializing if necessary"""
    global eval_collection
    if not _is_initialized:
        init_db_connections()
    return eval_collection  # May be None if EVAL_MONGO_URI is not set 