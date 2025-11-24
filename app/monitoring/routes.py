# app/monitoring/routes.py
from fastapi import APIRouter
from datetime import datetime, timedelta
from .monitoring import vocab_monitor

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/stats")
async def get_monitoring_stats():
    """Get monitoring statistics"""
    return vocab_monitor.get_stats()

@router.get("/top-words")
async def get_top_new_words(limit: int = 20):
    """Get top new words by frequency"""
    return {
        "top_new_words": vocab_monitor.get_top_new_words(limit),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/alerts")
async def get_alerts():
    """Get current alerts"""
    return {
        "alerts": vocab_monitor.alerts[-20:],
        "total_alerts": len(vocab_monitor.alerts)
    }

@router.get("/vocabulary-trend")
async def get_vocabulary_trend(days: int = 7):
    """Get vocabulary trend over time"""
    dates = []
    new_word_ratios = []
    prediction_counts = []
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        stats = vocab_monitor.daily_stats[date]
        
        total_predictions = stats["total_predictions"]
        new_word_count = stats["new_word_count"]
        
        if total_predictions > 0:
            ratio = new_word_count / (total_predictions * 10)
        else:
            ratio = 0
            
        dates.append(date)
        new_word_ratios.append(ratio)
        prediction_counts.append(total_predictions)
    
    return {
        "dates": list(reversed(dates)),
        "new_word_ratios": list(reversed(new_word_ratios)),
        "prediction_counts": list(reversed(prediction_counts))
    }

@router.post("/cleanup")
async def cleanup_old_data(days_to_keep: int = 30):
    """Clean up old monitoring data"""
    vocab_monitor.cleanup_old_data(days_to_keep)
    return {"message": f"Cleaned up data older than {days_to_keep} days"}

@router.get("/export")
async def export_monitoring_data():
    """Export monitoring data"""
    export_file = vocab_monitor.export_data()
    return {"message": f"Data exported to {export_file}", "file": export_file}

@router.get("/health")
async def monitoring_health():
    """Check monitoring system health"""
    stats = vocab_monitor.get_stats()
    return {
        "status": "healthy",
        "data_file": vocab_monitor.data_file,
        "total_unique_words": stats["daily_stats"]["unknown_vocabulary_size"],
        "data_last_updated": datetime.now().isoformat()
    }