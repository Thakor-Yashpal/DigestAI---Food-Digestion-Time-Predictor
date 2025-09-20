# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import uvicorn
import logging
from datetime import datetime, timedelta
import json
import os
from digestion_model import DigestAIPredictor
import asyncio
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DigestAI API",
    description="AI-powered food digestion time prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    food_name: str
    user_notes: Optional[str] = None
    meal_context: Optional[str] = None  # breakfast, lunch, dinner, snack
    
    @validator('food_name')
    def validate_food_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Food name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Food name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Food name must be less than 100 characters')
        return v.strip()

class PredictionResponse(BaseModel):
    success: bool
    food_name: str
    digestion_time_hours: Optional[float] = None
    digestion_time_minutes: Optional[int] = None
    confidence: Optional[str] = None
    description: Optional[str] = None
    nutritional_factors: Optional[Dict] = None
    timestamp: str
    prediction_id: str
    error: Optional[str] = None

class HistoryEntry(BaseModel):
    prediction_id: str
    food_name: str
    digestion_time_hours: float
    digestion_time_minutes: int
    confidence: str
    description: str
    user_notes: Optional[str]
    meal_context: Optional[str]
    timestamp: str
    nutritional_factors: Dict

class ComparisonRequest(BaseModel):
    food_names: List[str]
    
    @validator('food_names')
    def validate_food_names(cls, v):
        if len(v) < 2:
            raise ValueError('At least 2 foods required for comparison')
        if len(v) > 5:
            raise ValueError('Maximum 5 foods allowed for comparison')
        return v

# Global variables
predictor = None
prediction_history = []

# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Initialize the prediction model on startup"""
    global predictor
    try:
        logger.info("Loading DigestAI prediction model...")
        predictor = DigestAIPredictor()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # In production, you might want to fail startup here
        predictor = None

@lru_cache()
def get_predictor():
    """Dependency to get the predictor instance"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction model not available")
    return predictor

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to DigestAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy" if predictor else "model_loading"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_digestion_time(
    request: PredictionRequest,
    model: DigestAIPredictor = Depends(get_predictor)
):
    """Predict digestion time for a food item"""
    try:
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.food_name) % 10000}"
        
        # Make prediction
        result = model.predict(request.food_name)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400, 
                detail=result.get("error", "Prediction failed")
            )
        
        # Create response
        response = PredictionResponse(
            success=True,
            food_name=result["food_name"],
            digestion_time_hours=result["digestion_time_hours"],
            digestion_time_minutes=result["digestion_time_minutes"],
            confidence=result["confidence"],
            description=result["description"],
            nutritional_factors=result["nutritional_factors"],
            timestamp=datetime.now().isoformat(),
            prediction_id=prediction_id
        )
        
        # Add to history
        history_entry = HistoryEntry(
            prediction_id=prediction_id,
            food_name=result["food_name"],
            digestion_time_hours=result["digestion_time_hours"],
            digestion_time_minutes=result["digestion_time_minutes"],
            confidence=result["confidence"],
            description=result["description"],
            user_notes=request.user_notes,
            meal_context=request.meal_context,
            timestamp=datetime.now().isoformat(),
            nutritional_factors=result["nutritional_factors"]
        )
        
        prediction_history.append(history_entry.dict())
        
        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        logger.info(f"Prediction made for: {request.food_name} -> {result['digestion_time_minutes']} minutes")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/history")
async def get_prediction_history(
    limit: int = 20,
    offset: int = 0,
    food_filter: Optional[str] = None
):
    """Get prediction history with optional filtering"""
    try:
        # Filter by food name if provided
        filtered_history = prediction_history
        if food_filter:
            filtered_history = [
                entry for entry in prediction_history 
                if food_filter.lower() in entry["food_name"].lower()
            ]
        
        # Sort by timestamp (newest first)
        sorted_history = sorted(
            filtered_history, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        # Apply pagination
        paginated_history = sorted_history[offset:offset + limit]
        
        return {
            "history": paginated_history,
            "total_count": len(filtered_history),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.post("/compare")
async def compare_foods(
    request: ComparisonRequest,
    model: DigestAIPredictor = Depends(get_predictor)
):
    """Compare digestion times for multiple foods"""
    try:
        comparisons = []
        
        for food_name in request.food_names:
            result = model.predict(food_name)
            if result.get("success", False):
                comparisons.append({
                    "food_name": result["food_name"],
                    "digestion_time_hours": result["digestion_time_hours"],
                    "digestion_time_minutes": result["digestion_time_minutes"],
                    "confidence": result["confidence"],
                    "description": result["description"],
                    "nutritional_factors": result["nutritional_factors"]
                })
            else:
                comparisons.append({
                    "food_name": food_name,
                    "error": result.get("error", "Prediction failed")
                })
        
        # Sort by digestion time
        successful_comparisons = [c for c in comparisons if "error" not in c]
        failed_comparisons = [c for c in comparisons if "error" in c]
        
        successful_comparisons.sort(key=lambda x: x["digestion_time_minutes"])
        
        return {
            "success": True,
            "comparisons": successful_comparisons + failed_comparisons,
            "fastest": successful_comparisons[0] if successful_comparisons else None,
            "slowest": successful_comparisons[-1] if successful_comparisons else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/nutrition-tips")
async def get_nutrition_tips():
    """Get general nutrition and digestion tips"""
    tips = [
        {
            "category": "Macronutrients",
            "tip": "Fats and high fiber slow digestion. Lean proteins and simple carbs digest faster.",
            "icon": "🍽️"
        },
        {
            "category": "Hydration",
            "tip": "Hydration supports enzymatic activity—sip water before and after meals.",
            "icon": "💧"
        },
        {
            "category": "Movement",
            "tip": "Light movement post-meal can enhance gastric motility.",
            "icon": "🚶"
        },
        {
            "category": "Timing",
            "tip": "Late-night heavy meals may prolong digestion and affect sleep.",
            "icon": "⏰"
        },
        {
            "category": "Fiber",
            "tip": "Soluble fiber slows digestion more than insoluble fiber.",
            "icon": "🌾"
        },
        {
            "category": "Temperature",
            "tip": "Warm foods may digest slightly faster than cold foods.",
            "icon": "🌡️"
        }
    ]
    
    return {
        "tips": tips,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/food-categories")
async def get_food_categories():
    """Get information about different food categories and their typical digestion times"""
    categories = [
        {
            "name": "Liquids",
            "typical_time_range": "0-30 minutes",
            "examples": ["water", "juice", "milk", "smoothies"],
            "characteristics": "Fastest digestion, minimal processing required"
        },
        {
            "name": "Fruits",
            "typical_time_range": "30-60 minutes",
            "examples": ["apples", "bananas", "berries", "oranges"],
            "characteristics": "High water content, simple sugars, moderate fiber"
        },
        {
            "name": "Vegetables",
            "typical_time_range": "30-90 minutes",
            "examples": ["lettuce", "broccoli", "carrots", "spinach"],
            "characteristics": "High fiber content, low fat, good water content"
        },
        {
            "name": "Grains & Starches",
            "typical_time_range": "1-3 hours",
            "examples": ["rice", "bread", "pasta", "quinoa"],
            "characteristics": "Complex carbohydrates, moderate protein and fiber"
        },
        {
            "name": "Proteins",
            "typical_time_range": "2-4 hours",
            "examples": ["chicken", "fish", "eggs", "tofu"],
            "characteristics": "Requires significant digestive enzymes and processing"
        },
        {
            "name": "Nuts & Seeds",
            "typical_time_range": "3-4 hours",
            "examples": ["almonds", "walnuts", "chia seeds"],
            "characteristics": "High fat and protein content, dense nutrition"
        },
        {
            "name": "Mixed Meals",
            "typical_time_range": "2-5 hours",
            "examples": ["pizza", "burgers", "salads with protein"],
            "characteristics": "Combination of macronutrients, processing varies by composition"
        }
    ]
    
    return {
        "categories": categories,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str):
    """Delete a specific prediction from history"""
    global prediction_history
    
    original_length = len(prediction_history)
    prediction_history = [
        entry for entry in prediction_history 
        if entry["prediction_id"] != prediction_id
    ]
    
    if len(prediction_history) < original_length:
        return {"success": True, "message": "Prediction deleted"}
    else:
        raise HTTPException(status_code=404, detail="Prediction not found")

@app.delete("/history")
async def clear_history():
    """Clear all prediction history"""
    global prediction_history
    prediction_history.clear()
    return {"success": True, "message": "History cleared"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Something went wrong"}
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )