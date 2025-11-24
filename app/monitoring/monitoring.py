# app/monitoring/monitoring.py
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import heapq
from typing import Dict, List
import asyncio
import json
import os

class EfficientVocabMonitor:
    def __init__(self, max_top_words=100, data_file="monitoring_data.json"):
        self.data_file = data_file
        self.max_top_words = max_top_words
        self.load_data()  # Load existing data on startup
        
    def load_data(self):
        """Load monitoring data from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to Counter and defaultdict
                self.new_words_counter = Counter(data.get('new_words_counter', {}))
                self.daily_stats = defaultdict(lambda: {
                    "total_predictions": 0,
                    "new_word_count": 0,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Load daily stats
                for date, stats in data.get('daily_stats', {}).items():
                    self.daily_stats[date] = stats
                
                self.alerts = data.get('alerts', [])
                
                # Rebuild heap from counter
                self.top_words_heap = []
                for word, count in self.new_words_counter.most_common(self.max_top_words):
                    heapq.heappush(self.top_words_heap, (count, word))
                    if len(self.top_words_heap) >= self.max_top_words:
                        break
                
                print(f"✓ Loaded monitoring data from {self.data_file}")
                print(f"  - {len(self.new_words_counter)} unique new words")
                print(f"  - {len(self.alerts)} alerts")
            else:
                self.initialize_empty_data()
                print("✓ Initialized new monitoring data")
                
        except Exception as e:
            print(f"✗ Error loading monitoring data: {e}")
            self.initialize_empty_data()
    
    def initialize_empty_data(self):
        """Initialize empty data structures"""
        self.new_words_counter = Counter()
        self.daily_stats = defaultdict(lambda: {
            "total_predictions": 0,
            "new_word_count": 0,
            "timestamp": datetime.now().isoformat()
        })
        self.alerts = []
        self.top_words_heap = []
    
    def save_data(self):
        """Save monitoring data to JSON file"""
        try:
            # Convert to JSON-serializable format
            data = {
                'new_words_counter': dict(self.new_words_counter),
                'daily_stats': dict(self.daily_stats),
                'alerts': self.alerts,
                'last_updated': datetime.now().isoformat(),
                'metadata': {
                    'total_unique_new_words': len(self.new_words_counter),
                    'total_alerts': len(self.alerts),
                    'data_file_version': '1.0'
                }
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved monitoring data to {self.data_file}")
        except Exception as e:
            print(f"✗ Error saving monitoring data: {e}")
        
    def track_new_words(self, text: str, vocab: dict) -> List[str]:
        """Efficiently track new words with O(1) lookups and auto-save"""
        from app.utils.text_cleaning import clean_text
        
        cleaned = clean_text(text)
        words = cleaned.split()
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily_stats = self.daily_stats[today]
        daily_stats["total_predictions"] += 1
        daily_stats["timestamp"] = datetime.now().isoformat()
        
        new_words = []
        should_save = False
        
        for word in words:
            # O(1) lookup in dict
            if word not in vocab:
                self.new_words_counter[word] += 1
                count = self.new_words_counter[word]
                
                # Efficient top-K maintenance
                if len(self.top_words_heap) < self.max_top_words:
                    heapq.heappush(self.top_words_heap, (count, word))
                else:
                    if count > self.top_words_heap[0][0]:
                        heapq.heappushpop(self.top_words_heap, (count, word))
                
                daily_stats["new_word_count"] += 1
                new_words.append(word)
                should_save = True  # Mark that we have new data to save
                
                # Check for alerts
                if count >= 20:
                    self._create_alert(word, count)
        
        # Auto-save only if new words were found
        if should_save:
            self.save_data()
        
        return new_words
    
    def _create_alert(self, word: str, frequency: int):
        """Create alert for frequent new word"""
        alert = {
            "word": word,
            "frequency": frequency,
            "timestamp": datetime.now().isoformat(),
            "type": "HIGH_FREQUENCY_NEW_WORD",
            "severity": "HIGH" if frequency > 50 else "MEDIUM"
        }
        
        if not any(a["word"] == word for a in self.alerts[-100:]):
            self.alerts.append(alert)
            self.alerts = self.alerts[-100:]  # Keep last 100 alerts
            self.save_data()  # Save when new alert is created
    
    def get_top_new_words(self, limit: int = 20) -> List[Dict]:
        """Get top new words efficiently using heap"""
        sorted_words = sorted(self.top_words_heap, reverse=True)
        return [
            {"word": word, "frequency": count}
            for count, word in sorted_words[:limit]
        ]
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_stats = self.daily_stats[today]
        
        total_predictions = daily_stats["total_predictions"]
        new_word_count = daily_stats["new_word_count"]
        
        avg_words_per_comment = 10
        total_words_processed = total_predictions * avg_words_per_comment
        new_word_ratio = new_word_count / total_words_processed if total_words_processed > 0 else 0
        
        # Calculate 7-day trend
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        weekly_new_words = sum(
            stats["new_word_count"] 
            for date, stats in self.daily_stats.items() 
            if date >= week_ago
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "daily_stats": {
                "total_predictions": total_predictions,
                "new_word_count": new_word_count,
                "new_word_ratio": round(new_word_ratio, 4),
                "unknown_vocabulary_size": len(self.new_words_counter)
            },
            "weekly_stats": {
                "total_new_words": weekly_new_words,
                "unique_new_words": len([
                    word for word, count in self.new_words_counter.items()
                    if count >= 1  # Words that appeared at least once
                ])
            },
            "alert_status": "ALERT" if new_word_ratio > 0.1 else "NORMAL"
        }
    
    def cleanup_old_data(self, days_to_keep=30):
        """Clean up data older than specified days"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
        dates_to_remove = [date for date in self.daily_stats.keys() if date < cutoff_date]
        
        for date in dates_to_remove:
            del self.daily_stats[date]
        
        if dates_to_remove:
            self.save_data()
            print(f"✓ Cleaned up {len(dates_to_remove)} days of old data")
    
    def export_data(self, export_file="monitoring_export.json"):
        """Export complete monitoring data for analysis"""
        data = {
            'new_words_counter': dict(self.new_words_counter),
            'daily_stats': dict(self.daily_stats),
            'alerts': self.alerts,
            'export_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_unique_new_words': len(self.new_words_counter),
                'total_alerts_generated': len(self.alerts),
                'total_days_tracked': len(self.daily_stats),
                'most_common_word': self.new_words_counter.most_common(1)[0] if self.new_words_counter else None
            }
        }
        
        with open(export_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return export_file

# Global monitor instance
vocab_monitor = EfficientVocabMonitor(data_file="data/monitoring_data.json")

async def track_prediction_for_monitoring(text: str, vocab: dict):
    """Async function to track prediction for monitoring"""
    try:
        vocab_monitor.track_new_words(text, vocab)
    except Exception as e:
        # Silently fail - monitoring shouldn't affect predictions
        pass