"""
MongoDB connection management
"""
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure
from ..config import settings

logger = logging.getLogger(__name__)

# Global database client
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo():
    """Initialize MongoDB connection"""
    global _client, _database
    
    try:
        _client = AsyncIOMotorClient(settings.MONGODB_URL)
        _database = _client[settings.DATABASE_NAME]
        
        # Test connection
        await _client.admin.command('ping')
        logger.info(f"Connected to MongoDB: {settings.DATABASE_NAME}")
        
        # Create indexes for performance
        await create_indexes()
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    global _client
    if _client:
        _client.close()
        logger.info("Disconnected from MongoDB")


async def create_indexes():
    """Create database indexes for optimal performance"""
    if not _database:
        return
        
    # Orders collection indexes
    orders_collection = _database.orders
    await orders_collection.create_index("order_id", unique=True)
    await orders_collection.create_index("kitchen_id")
    await orders_collection.create_index("current_state")
    await orders_collection.create_index("created_at")
    
    # Kitchens collection indexes
    kitchens_collection = _database.kitchens
    await kitchens_collection.create_index("kitchen_id", unique=True)
    await kitchens_collection.create_index("status")
    
    # Analytics collection indexes
    analytics_collection = _database.analytics
    await analytics_collection.create_index("kitchen_id")
    await analytics_collection.create_index("timestamp")
    await analytics_collection.create_index([("kitchen_id", 1), ("timestamp", -1)])
    
    logger.info("Database indexes created successfully")


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance"""
    if not _database:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return _database


def get_collection(collection_name: str) -> AsyncIOMotorCollection:
    """Get a specific collection"""
    database = get_database()
    return database[collection_name]