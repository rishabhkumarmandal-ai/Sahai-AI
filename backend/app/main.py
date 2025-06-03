python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes import assistant, user
from app.database.firebase_config import initialize_firebase
import uvicorn

app = FastAPI(title="Sahai Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase
initialize_firebase()

# routers
app.include_router(assistant.router, prefix="/api/assistant", tags=["assistant"])
app.include_router(user.router, prefix="/api/user", tags=["user"])

@app.get("/")
async def root():
    return {"message": "Sahai Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Sahai"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```