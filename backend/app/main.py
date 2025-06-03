"""
Sahai - AI Assistant Backend
FastAPI Main Application Entry Point

"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from datetime import datetime
import os
from contextlib import asynccontextmanager
import asyncio


from routes.assistant import router as assistant_router
from routes.user import router as user_router


from services.nlp_engine import NLPEngine
from services.weather_service import WeatherService
from services.news_service import NewsService
from services.calendar import CalendarService
from services.culture_knowledge import CultureKnowledgeService


from database.firebase_config import initialize_database


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sahai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    
    logger.info(" Starting Sahai Assistant Backend...")
    
    try:
       
        logger.info(" Initializing database connection...")
        await initialize_database()
        
       
        logger.info(" Initializing NLP Engine...")
        services['nlp'] = NLPEngine()
        await services['nlp'].initialize()
        
        logger.info(" Initializing Weather Service...")
        services['weather'] = WeatherService()
        
        logger.info(" Initializing News Service...")
        services['news'] = NewsService()
        
        logger.info(" Initializing Calendar Service...")
        services['calendar'] = CalendarService()
        
        logger.info(" Initializing Culture Knowledge Service...")
        services['culture'] = CultureKnowledgeService()
        await services['culture'].load_knowledge_base()
        
        logger.info(" All services initialized successfully!")
        
    except Exception as e:
        logger.error(f" Failed to initialize services: {str(e)}")
        raise
    
    yield
    
   
    logger.info(" Shutting down Sahai Assistant Backend...")
    
    
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f" Right {service_name} service cleaned up")
        except Exception as e:
            logger.error(f"wrong Error cleaning up {service_name}: {str(e)}")


app = FastAPI(
    title="Sahai - AI Assistant Backend",
    description="A culturally-aware AI assistant for Indian users with multilingual support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


settings = Settings()


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests and add processing time
    """
    start_time = datetime.now()
    
  
    logger.info(f"📨 {request.method} {request.url.path} - Client: {request.client.host}")
    

    response = await call_next(request)
    
   
    process_time = (datetime.now() - start_time).total_seconds()
    
    
    response.headers["X-Process-Time"] = str(process_time)
    
 
    logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response


@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with basic API information
    """
    return {
        "message": "🙏 Welcome to Sahai - Your AI Assistant",
        "description": "Culturally-aware AI assistant for Indian users",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "assistant": "/api/v1/assistant",
            "user": "/api/v1/user"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    try:
     
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
       
        for service_name, service in services.items():
            try:
                if hasattr(service, 'health_check'):
                    health_status["services"][service_name] = await service.health_check()
                else:
                    health_status["services"][service_name] = "active"
            except Exception as e:
                health_status["services"][service_name] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/status", tags=["System"])
async def system_status():
    """
    Detailed system status and metrics
    """
    return {
        "system": "Sahai AI Assistant",
        "version": "1.0.0",
        "uptime": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "services_count": len(services),
        "features": [
            "Multilingual Support (Hindi/English)",
            "Weather Updates",
            "News Integration",
            "Calendar & Reminders",
            "Cultural Knowledge Base",
            "Voice Input/Output",
            "Personalization"
        ],
        "supported_languages": ["hi", "en"],
        "regions": ["India"]
    }


async def get_services():
    """
    Dependency to provide services to route handlers
    """
    return services


app.include_router(
    assistant_router,
    prefix="/api/v1/assistant",
    tags=["Assistant"],
    dependencies=[Depends(get_services)]
)

app.include_router(
    user_router,
    prefix="/api/v1/user",
    tags=["User Management"],
    dependencies=[Depends(get_services)]
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with proper logging
    """
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions
    """
    logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket, user_id: str):
    """
    WebSocket endpoint for real-time chat functionality
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for user: {user_id}")
    
    try:
        while True:
           
            data = await websocket.receive_json()
            
          
            if 'nlp' in services:
                response = await services['nlp'].process_message(
                    message=data.get('message', ''),
                    user_id=user_id,
                    language=data.get('language', 'en')
                )
                
              
                await websocket.send_json({
                    "type": "response",
                    "data": response,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "NLP service not available",
                    "timestamp": datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
    finally:
        logger.info(f"WebSocket connection closed for user: {user_id}")


if __name__ == "__main__":
  
    config = {
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "log_level": "info" if settings.DEBUG else "warning",
        "access_log": settings.DEBUG
    }
    
    logger.info(f" Starting Sahai Backend on {config['host']}:{config['port']}")
    logger.info(f" Debug mode: {settings.DEBUG}")
    logger.info(f"API Documentation: http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run(
        "main:app",
        **config
    )