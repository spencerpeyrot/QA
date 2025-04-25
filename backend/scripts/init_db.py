from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def init_db():
    # Connect to MongoDB
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client.qa_platform
    
    # Create collection if it doesn't exist
    if "qa_evaluations" not in await db.list_collection_names():
        await db.create_collection("qa_evaluations")
    
    # Create indexes
    await db.qa_evaluations.create_index("created_at")
    await db.qa_evaluations.create_index([("agent", 1), ("sub_component", 1)])
    
    print("Database initialized successfully!")
    
    # Close the connection
    client.close()

if __name__ == "__main__":
    asyncio.run(init_db()) 