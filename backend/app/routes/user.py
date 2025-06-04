"""
Sahai - User Management Routes
Backend API routes for user profiles, preferences, authentication,
and personalization features.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
import hashlib
import jwt
from enum import Enum
import re

# Import models
from models.user import UserProfile, UserPreferences, UserSettings, LanguageCode
from models.chat import ChatSession

# This would be injected from main.py
from fastapi import Request

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Security
security = HTTPBearer()

# Enums for user management
class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class NotificationPreference(str, Enum):
    ALL = "all"
    IMPORTANT = "important"
    NONE = "none"

class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    FRIENDS = "friends"

# Pydantic models for requests/responses
class UserRegistration(BaseModel):
    """User registration model"""
    name: str = Field(..., min_length=2, max_length=100, description="User's full name")
    email: Optional[EmailStr] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Preferred language")
    location: Optional[str] = Field(None, description="User's location/city")
    timezone: str = Field(default="Asia/Kolkata", description="User's timezone")
    age_group: Optional[str] = Field(None, description="Age group")
    interests: Optional[List[str]] = Field(default=[], description="User interests")
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is not None:
            # Indian phone number validation
            phone_pattern = r'^(\+91|91|0)?[6-9][0-9]{9}$'
            if not re.match(phone_pattern, v.replace(' ', '').replace('-', '')):
                raise ValueError('Invalid Indian phone number format')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

class UserLogin(BaseModel):
    """User login model"""
    identifier: str = Field(..., description="Email or phone number")
    password: Optional[str] = Field(None, description="Password (if using password auth)")
    otp: Optional[str] = Field(None, description="OTP for phone login")
    device_info: Optional[Dict[str, str]] = Field(default={}, description="Device information")

class UserProfileUpdate(BaseModel):
    """User profile update model"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = Field(None)
    phone: Optional[str] = Field(None)
    location: Optional[str] = Field(None)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None)
    interests: Optional[List[str]] = Field(None)
    age_group: Optional[str] = Field(None)

class UserPreferencesUpdate(BaseModel):
    """User preferences update model"""
    language: Optional[LanguageCode] = Field(None)
    timezone: Optional[str] = Field(None)
    notification_preferences: Optional[Dict[str, NotificationPreference]] = Field(default={})
    privacy_level: Optional[PrivacyLevel] = Field(None)
    voice_enabled: Optional[bool] = Field(None)
    auto_translate: Optional[bool] = Field(None)
    cultural_content: Optional[bool] = Field(True)
    news_categories: Optional[List[str]] = Field(None)
    weather_location: Optional[str] = Field(None)
    reminder_defaults: Optional[Dict[str, Any]] = Field(default={})

class UserSettingsUpdate(BaseModel):
    """User settings update model"""
    theme: Optional[str] = Field(None, description="UI theme preference")
    font_size: Optional[str] = Field(None, description="Font size preference")
    sound_enabled: Optional[bool] = Field(None)
    vibration_enabled: Optional[bool] = Field(None)
    location_sharing: Optional[bool] = Field(None)
    data_usage_wifi_only: Optional[bool] = Field(None)
    auto_backup: Optional[bool] = Field(None)
    analytics_consent: Optional[bool] = Field(None)

class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    status: UserStatus
    created_at: datetime
    last_active: Optional[datetime]
    profile: Optional[Dict[str, Any]]
    preferences: Optional[Dict[str, Any]]
    settings: Optional[Dict[str, Any]]

# Dependency to get services
async def get_services(request: Request):
    """Get services from app state"""
    return request.app.extra.get("services", {})

# Dependency for authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    services: dict = Depends(get_services)
):
    """Get current authenticated user"""
    try:
        # Decode JWT token
        token = credentials.credentials
        # This would integrate with your auth service
        # For now, return a mock user
        return {"user_id": "mock_user", "email": "user@example.com"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# User registration endpoint
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistration,
    background_tasks: BackgroundTasks,
    services: dict = Depends(get_services)
):
    """
    Register a new user
    """
    try:
        logger.info(f"New user registration: {user_data.email or user_data.phone}")
        
        # Check if user already exists
        # This would integrate with your database service
        existing_user = await check_existing_user(user_data.email, user_data.phone, services)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email or phone already exists"
            )
        
        # Create user profile
        user_id = await create_user_profile(user_data, services)
        
        # Create default preferences
        await create_default_preferences(user_id, user_data, services)
        
        # Send welcome message/notification
        background_tasks.add_task(send_welcome_notification, user_id, user_data, services)
        
        # Get created user
        user = await get_user_by_id(user_id, services)
        
        return UserResponse(
            user_id=user_id,
            name=user_data.name,
            email=user_data.email,
            phone=user_data.phone,
            status=UserStatus.ACTIVE,
            created_at=datetime.now(),
            last_active=None,
            profile=user.get('profile', {}),
            preferences=user.get('preferences', {}),
            settings=user.get('settings', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

# User login endpoint
@router.post("/login")
async def login_user(
    login_data: UserLogin,
    services: dict = Depends(get_services)
):
    """
    User login with email/phone and OTP/password
    """
    try:
        logger.info(f"Login attempt: {login_data.identifier}")
        
        # Authenticate user
        user = await authenticate_user(login_data, services)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Generate JWT token
        token = generate_jwt_token(user['user_id'], user.get('email'))
        
        # Update last login
        await update_last_active(user['user_id'], services)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": UserResponse(
                user_id=user['user_id'],
                name=user['name'],
                email=user.get('email'),
                phone=user.get('phone'),
                status=UserStatus(user.get('status', 'active')),
                created_at=user['created_at'],
                last_active=datetime.now(),
                profile=user.get('profile', {}),
                preferences=user.get('preferences', {}),
                settings=user.get('settings', {})
            ),
            "expires_in": 86400  # 24 hours
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {login_data.identifier}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

# Get user profile
@router.get("/profile/{user_id}", response_model=UserResponse)
async def get_user_profile(
    user_id: str = Path(..., description="User identifier"),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Get user profile information
    """
    try:
        # Check if user can access this profile
        if current_user['user_id'] != user_id:
            # Check if profile is public or user has permission
            if not await can_access_profile(current_user['user_id'], user_id, services):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this profile"
                )
        
        user = await get_user_by_id(user_id, services)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            user_id=user['user_id'],
            name=user['name'],
            email=user.get('email'),
            phone=user.get('phone'),
            status=UserStatus(user.get('status', 'active')),
            created_at=user['created_at'],
            last_active=user.get('last_active'),
            profile=user.get('profile', {}),
            preferences=user.get('preferences', {}),
            settings=user.get('settings', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

# Update user profile
@router.put("/profile/{user_id}")
async def update_user_profile(
    user_id: str = Path(..., description="User identifier"),
    profile_update: UserProfileUpdate = Body(...),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Update user profile information
    """
    try:
        # Check if user can update this profile
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own profile"
            )
        
        # Update profile
        updated_user = await update_user_profile_data(user_id, profile_update, services)
        
        return {
            "message": "Profile updated successfully",
            "user": updated_user,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

# Get user preferences
@router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: str = Path(..., description="User identifier"),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Get user preferences and settings
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only access your own preferences"
            )
        
        preferences = await get_user_preferences_data(user_id, services)
        
        return {
            "preferences": preferences,
            "user_id": user_id,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get preferences error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get preferences"
        )

# Update user preferences
@router.put("/preferences/{user_id}")
async def update_user_preferences(
    user_id: str = Path(..., description="User identifier"),
    preferences_update: UserPreferencesUpdate = Body(...),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Update user preferences
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own preferences"
            )
        
        updated_preferences = await update_user_preferences_data(
            user_id, preferences_update, services
        )
        
        return {
            "message": "Preferences updated successfully",
            "preferences": updated_preferences,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preferences update error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

# Update user settings
@router.put("/settings/{user_id}")
async def update_user_settings(
    user_id: str = Path(..., description="User identifier"),
    settings_update: UserSettingsUpdate = Body(...),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Update user application settings
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own settings"
            )
        
        updated_settings = await update_user_settings_data(
            user_id, settings_update, services
        )
        
        return {
            "message": "Settings updated successfully",
            "settings": updated_settings,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Settings update error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update settings"
        )

# Get user activity/stats
@router.get("/activity/{user_id}")
async def get_user_activity(
    user_id: str = Path(..., description="User identifier"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days"),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Get user activity statistics
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only access your own activity"
            )
        
        activity_data = await get_user_activity_data(user_id, days, services)
        
        return {
            "activity": activity_data,
            "user_id": user_id,
            "period_days": days,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activity error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get activity data"
        )

# Send OTP for phone verification
@router.post("/send-otp")
async def send_otp(
    phone: str = Body(..., embed=True),
    purpose: str = Body(default="login", embed=True),
    services: dict = Depends(get_services)
):
    """
    Send OTP to phone number for verification
    """
    try:
        # Validate phone number
        phone_pattern = r'^(\+91|91|0)?[6-9][0-9]{9}$'
        if not re.match(phone_pattern, phone.replace(' ', '').replace('-', '')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid phone number format"
            )
        
        # Send OTP
        otp_sent = await send_otp_to_phone(phone, purpose, services)
        
        return {
            "message": "OTP sent successfully",
            "phone": phone,
            "expires_in": 300,  # 5 minutes
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP send error for {phone}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP"
        )

# Verify OTP
@router.post("/verify-otp")
async def verify_otp(
    phone: str = Body(..., embed=True),
    otp: str = Body(..., embed=True),
    services: dict = Depends(get_services)
):
    """
    Verify OTP for phone number
    """
    try:
        is_valid = await verify_otp_code(phone, otp, services)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP"
            )
        
        return {
            "message": "OTP verified successfully",
            "phone": phone,
            "verified": True,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP verification error for {phone}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OTP verification failed"
        )

# Delete user account
@router.delete("/account/{user_id}")
async def delete_user_account(
    user_id: str = Path(..., description="User identifier"),
    confirmation: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks,
    services: dict = Depends(get_services)
):
    """
    Delete user account (with confirmation)
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only delete your own account"
            )
        
        if confirmation.lower() != "delete":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid confirmation. Type 'delete' to confirm"
            )
        
        # Schedule account deletion
        background_tasks.add_task(process_account_deletion, user_id, services)
        
        return {
            "message": "Account deletion initiated",
            "user_id": user_id,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )

# Export user data
@router.get("/export/{user_id}")
async def export_user_data(
    user_id: str = Path(..., description="User identifier"),
    current_user: dict = Depends(get_current_user),
    services: dict = Depends(get_services)
):
    """
    Export all user data (GDPR compliance)
    """
    try:
        if current_user['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only export your own data"
            )
        
        user_data = await export_user_data_complete(user_id, services)
        
        return {
            "user_data": user_data,
            "export_date": datetime.now(),
            "format": "json",
            "gdpr_compliant": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data export failed"
        )

# Helper functions (these would integrate with your database service)
async def check_existing_user(email: str, phone: str, services: dict) -> bool:
    """Check if user already exists"""
    # Mock implementation
    return False

async def create_user_profile(user_data: UserRegistration, services: dict) -> str:
    """Create new user profile"""
    # Mock implementation
    return f"user_{datetime.now().timestamp()}"

async def create_default_preferences(user_id: str, user_data: UserRegistration, services: dict):
    """Create default user preferences"""
    # Mock implementation
    pass

async def get_user_by_id(user_id: str, services: dict) -> dict:
    """Get user by ID"""
    # Mock implementation
    return {
        "user_id": user_id,
        "name": "Mock User",
        "email": "user@example.com",
        "status": "active",
        "created_at": datetime.now(),
        "profile": {},
        "preferences": {},
        "settings": {}
    }

async def authenticate_user(login_data: UserLogin, services: dict) -> dict:
    """Authenticate user credentials"""
    # Mock implementation
    return {
        "user_id": "mock_user",
        "name": "Mock User",
        "email": "user@example.com",
        "created_at": datetime.now()
    }

def generate_jwt_token(user_id: str, email: str) -> str:
    """Generate JWT token"""
    # Mock implementation
    return "mock_jwt_token"

async def send_welcome_notification(user_id: str, user_data: UserRegistration, services: dict):
    """Send welcome notification to new user"""
    logger.info(f"Welcome notification sent to user {user_id}")

async def can_access_profile(requesting_user: str, target_user: str, services: dict) -> bool:
    """Check if user can access another user's profile"""
    return False

async def update_last_active(user_id: str, services: dict):
    """Update user's last active timestamp"""
    pass

async def update_user_profile_data(user_id: str, profile_update: UserProfileUpdate, services: dict) -> dict:
    """Update user profile data"""
    return {"message": "Profile updated"}

async def get_user_preferences_data(user_id: str, services: dict) -> dict:
    """Get user preferences"""
    return {"language": "en", "timezone": "Asia/Kolkata"}

async def update_user_preferences_data(user_id: str, preferences: UserPreferencesUpdate, services: dict) -> dict:
    """Update user preferences"""
    return {"message": "Preferences updated"}

async def update_user_settings_data(user_id: str, settings: UserSettingsUpdate, services: dict) -> dict:
    """Update user settings"""
    return {"message": "Settings updated"}

async def get_user_activity_data(user_id: str, days: int, services: dict) -> dict:
    """Get user activity statistics"""
    return {"total_messages": 100, "active_days": 15}

async def send_otp_to_phone(phone: str, purpose: str, services: dict) -> bool:
    """Send OTP to phone number"""
    return True

async def verify_otp_code(phone: str, otp: str, services: dict) -> bool:
    """Verify OTP code"""
    return True

async def process_account_deletion(user_id: str, services: dict):
    """Process account deletion"""
    logger.info(f"Processing account deletion for user {user_id}")

async def export_user_data_complete(user_id: str, services: dict) -> dict:
    """Export complete user data"""
    return {"user_id": user_id, "data": "exported_data"}