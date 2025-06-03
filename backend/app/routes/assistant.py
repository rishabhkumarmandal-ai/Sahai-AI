"""
Sahai - AI Assistant Routes
Backend API routes for assistant functionality including chat, NLP processing,
weather, news, reminders, and cultural knowledge.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
import json
import asyncio
from enum import Enum


from models.chat import ChatSession, ChatMessage, MessageType, LanguageCode
from models.user import UserProfile


from fastapi import Request

logger = logging.getLogger(__name__)


router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat messages"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Message language")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    voice_input: Optional[bool] = Field(default=False, description="Whether input came from voice")
    location: Optional[Dict[str, float]] = Field(None, description="User location for context")

    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for chat messages"""
    response: str = Field(..., description="Assistant response")
    language: LanguageCode = Field(..., description="Response language")
    session_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    intent: Optional[str] = Field(None, description="Detected user intent")
    entities: Optional[Dict[str, Any]] = Field(default={}, description="Extracted entities")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Context information")
    suggestions: Optional[List[str]] = Field(default=[], description="Follow-up suggestions")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class WeatherRequest(BaseModel):
    """Request model for weather information"""
    location: Optional[str] = Field(None, description="Location name")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Lat/lng coordinates")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    user_id: str = Field(..., description="User identifier")

class NewsRequest(BaseModel):
    """Request model for news"""
    category: Optional[str] = Field(None, description="News category")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    user_id: str = Field(..., description="User identifier")
    location: Optional[str] = Field(None, description="Location for local news")
    limit: int = Field(default=5, ge=1, le=20, description="Number of articles")

class ReminderRequest(BaseModel):
    """Request model for creating reminders"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    datetime: datetime = Field(..., description="Reminder datetime")
    user_id: str = Field(..., description="User identifier")
    recurrence: Optional[str] = Field(None, description="Recurrence pattern")
    priority: Optional[str] = Field(default="medium", description="Priority level")

class CulturalRequest(BaseModel):
    """Request model for cultural information"""
    query: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = Field(None, description="Cultural category")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    user_id: str = Field(..., description="User identifier")

class IntentType(str, Enum):
    """Available intent types"""
    GREETING = "greeting"
    WEATHER = "weather"
    NEWS = "news"
    REMINDER = "reminder"
    CULTURAL = "cultural"
    GENERAL = "general"
    FAREWELL = "farewell"


async def get_services(request: Request):
    """Get services from app state"""
    return request.app.extra.get("services", {})


@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    services: dict = Depends(get_services)
):
    """
    Main chat endpoint for conversing with the AI assistant
    """
    try:
        logger.info(f"Chat request from user {chat_request.user_id}: {chat_request.message[:50]}...")
        
        
        nlp_service = services.get('nlp')
        if not nlp_service:
            raise HTTPException(status_code=503, detail="NLP service unavailable")
        
        
        response_data = await nlp_service.process_message(
            message=chat_request.message,
            user_id=chat_request.user_id,
            language=chat_request.language,
            session_id=chat_request.session_id,
            context=chat_request.context,
            location=chat_request.location
        )
        
        
        chat_response = ChatResponse(
            response=response_data.get('response', ''),
            language=response_data.get('language', chat_request.language),
            session_id=response_data.get('session_id', ''),
            message_id=response_data.get('message_id', ''),
            timestamp=datetime.now(),
            intent=response_data.get('intent'),
            entities=response_data.get('entities', {}),
            context=response_data.get('context', {}),
            suggestions=response_data.get('suggestions', []),
            metadata=response_data.get('metadata', {})
        )
        
        
        background_tasks.add_task(
            save_chat_history,
            chat_request,
            chat_response,
            services
        )
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Chat error for user {chat_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/weather")
async def get_weather_info(
    weather_request: WeatherRequest,
    services: dict = Depends(get_services)
):
    """
    Get weather information for specified location
    """
    try:
        weather_service = services.get('weather')
        if not weather_service:
            raise HTTPException(status_code=503, detail="Weather service unavailable")
        
      
        weather_data = await weather_service.get_weather(
            location=weather_request.location,
            coordinates=weather_request.coordinates,
            language=weather_request.language
        )
        
        return {
            "weather": weather_data,
            "timestamp": datetime.now(),
            "language": weather_request.language
        }
        
    except Exception as e:
        logger.error(f"Weather error for user {weather_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weather service failed: {str(e)}")


@router.post("/news")
async def get_news_updates(
    news_request: NewsRequest,
    services: dict = Depends(get_services)
):
    """
    Get latest news updates
    """
    try:
        news_service = services.get('news')
        if not news_service:
            raise HTTPException(status_code=503, detail="News service unavailable")
        
       
        news_data = await news_service.get_news(
            category=news_request.category,
            language=news_request.language,
            location=news_request.location,
            limit=news_request.limit
        )
        
        return {
            "news": news_data,
            "timestamp": datetime.now(),
            "category": news_request.category,
            "language": news_request.language
        }
        
    except Exception as e:
        logger.error(f"News error for user {news_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"News service failed: {str(e)}")


@router.post("/reminders")
async def create_reminder(
    reminder_request: ReminderRequest,
    background_tasks: BackgroundTasks,
    services: dict = Depends(get_services)
):
    """
    Create a new reminder
    """
    try:
        calendar_service = services.get('calendar')
        if not calendar_service:
            raise HTTPException(status_code=503, detail="Calendar service unavailable")
        

        reminder = await calendar_service.create_reminder(
            title=reminder_request.title,
            description=reminder_request.description,
            datetime=reminder_request.datetime,
            user_id=reminder_request.user_id,
            recurrence=reminder_request.recurrence,
            priority=reminder_request.priority
        )
        
       
        background_tasks.add_task(
            schedule_reminder_notification,
            reminder,
            services
        )
        
        return {
            "reminder": reminder,
            "message": "Reminder created successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Reminder creation error for user {reminder_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reminder creation failed: {str(e)}")

@router.get("/reminders/{user_id}")
async def get_user_reminders(
    user_id: str = Path(..., description="User identifier"),
    limit: int = Query(default=10, ge=1, le=50),
    upcoming_only: bool = Query(default=True),
    services: dict = Depends(get_services)
):
    """
    Get user's reminders
    """
    try:
        calendar_service = services.get('calendar')
        if not calendar_service:
            raise HTTPException(status_code=503, detail="Calendar service unavailable")
        
        reminders = await calendar_service.get_user_reminders(
            user_id=user_id,
            limit=limit,
            upcoming_only=upcoming_only
        )
        
        return {
            "reminders": reminders,
            "count": len(reminders),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Get reminders error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reminders: {str(e)}")


@router.post("/cultural")
async def get_cultural_info(
    cultural_request: CulturalRequest,
    services: dict = Depends(get_services)
):
    """
    Get cultural information and knowledge
    """
    try:
        culture_service = services.get('culture')
        if not culture_service:
            raise HTTPException(status_code=503, detail="Culture service unavailable")
        
       
        cultural_data = await culture_service.get_cultural_info(
            query=cultural_request.query,
            category=cultural_request.category,
            language=cultural_request.language
        )
        
        return {
            "cultural_info": cultural_data,
            "query": cultural_request.query,
            "category": cultural_request.category,
            "language": cultural_request.language,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Cultural info error for user {cultural_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cultural service failed: {str(e)}")


@router.post("/detect-intent")
async def detect_user_intent(
    message: str = Body(..., embed=True),
    language: LanguageCode = Body(default=LanguageCode.ENGLISH, embed=True),
    services: dict = Depends(get_services)
):
    """
    Detect user intent from message
    """
    try:
        nlp_service = services.get('nlp')
        if not nlp_service:
            raise HTTPException(status_code=503, detail="NLP service unavailable")
        
        intent_data = await nlp_service.detect_intent(
            message=message,
            language=language
        )
        
        return {
            "intent": intent_data.get('intent'),
            "confidence": intent_data.get('confidence'),
            "entities": intent_data.get('entities', {}),
            "message": message,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Intent detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intent detection failed: {str(e)}")


@router.get("/chat-history/{user_id}")
async def get_chat_history(
    user_id: str = Path(..., description="User identifier"),
    session_id: Optional[str] = Query(None, description="Specific session ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    services: dict = Depends(get_services)
):
    """
    Get user's chat history
    """
    try:
        nlp_service = services.get('nlp')
        if not nlp_service:
            raise HTTPException(status_code=503, detail="NLP service unavailable")
        
        chat_history = await nlp_service.get_chat_history(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "chat_history": chat_history,
            "user_id": user_id,
            "session_id": session_id,
            "count": len(chat_history),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Chat history error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")


@router.post("/voice/process")
async def process_voice_input(
    audio_data: bytes = Body(...),
    language: LanguageCode = Body(default=LanguageCode.ENGLISH, embed=True),
    user_id: str = Body(..., embed=True),
    services: dict = Depends(get_services)
):
    """
    Process voice input and return text + response
    """
    try:
        
        return {
            "transcription": "Voice processing not implemented yet",
            "response": "Voice processing feature coming soon!",
            "language": language,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Voice processing error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_response(
    chat_request: ChatRequest,
    services: dict = Depends(get_services)
):
    """
    Stream chat response for real-time conversation
    """
    try:
        nlp_service = services.get('nlp')
        if not nlp_service:
            raise HTTPException(status_code=503, detail="NLP service unavailable")
        
        async def generate_response():
            """Generator for streaming response"""
            try:
                
                async for chunk in nlp_service.stream_response(
                    message=chat_request.message,
                    user_id=chat_request.user_id,
                    language=chat_request.language,
                    session_id=chat_request.session_id
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)  
                
                
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Streaming chat error for user {chat_request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


async def save_chat_history(
    chat_request: ChatRequest,
    chat_response: ChatResponse,
    services: dict
):
    """
    Background task to save chat history
    """
    try:
        nlp_service = services.get('nlp')
        if nlp_service and hasattr(nlp_service, 'save_chat_message'):
            await nlp_service.save_chat_message(
                user_id=chat_request.user_id,
                session_id=chat_response.session_id,
                user_message=chat_request.message,
                assistant_response=chat_response.response,
                language=chat_request.language,
                intent=chat_response.intent,
                entities=chat_response.entities
            )
        logger.info(f"Chat history saved for user {chat_request.user_id}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

async def schedule_reminder_notification(reminder: dict, services: dict):
    """
    Background task to schedule reminder notifications
    """
    try:
        calendar_service = services.get('calendar')
        if calendar_service and hasattr(calendar_service, 'schedule_notification'):
            await calendar_service.schedule_notification(reminder)
        logger.info(f"Reminder notification scheduled: {reminder.get('id')}")
    except Exception as e:
        logger.error(f"Failed to schedule reminder notification: {str(e)}")


@router.get("/capabilities")
async def get_assistant_capabilities():
    """
    Get information about assistant capabilities
    """
    return {
        "capabilities": [
            {
                "name": "Natural Language Processing",
                "description": "Understanding and responding to user messages in Hindi and English",
                "languages": ["hi", "en"]
            },
            {
                "name": "Weather Information",
                "description": "Providing current weather and forecasts for Indian cities",
                "features": ["current_weather", "forecast", "location_based"]
            },
            {
                "name": "News Updates",
                "description": "Latest news from Indian and international sources",
                "categories": ["general", "sports", "technology", "business", "entertainment"]
            },
            {
                "name": "Reminders & Calendar",
                "description": "Creating and managing personal reminders and schedules",
                "features": ["one_time", "recurring", "priority_levels"]
            },
            {
                "name": "Cultural Knowledge",
                "description": "Information about Indian culture, festivals, food, and traditions",
                "categories": ["festivals", "food", "traditions", "history", "customs"]
            },
            {
                "name": "Voice Interaction",
                "description": "Voice input and output capabilities",
                "features": ["speech_to_text", "text_to_speech", "multilingual"]
            }
        ],
        "supported_languages": ["hi", "en"],
        "regions": ["India"],
        "version": "1.0.0",
        "timestamp": datetime.now()
    }