"""
Sentiment analysis module for Data Kitchen
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .logger import setup_logger
import time
import sqlite3
from pathlib import Path
import json
import hashlib

logger = setup_logger(__name__)

class CacheManager:
    """Cache manager for sentiment data"""
    
    def __init__(self, cache_dir: str):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / 'sentiment_cache.db'
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp FLOAT,
                    expires FLOAT
                )
            """)
            # Create index for faster cleanup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON sentiment_cache(expires)")
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, expires FROM sentiment_cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row and row[1] > time.time():
                    return json.loads(row[0])
                
                # Delete expired entry if exists
                if row:
                    conn.execute("DELETE FROM sentiment_cache WHERE key = ?", (key,))
                    
                return None
        except Exception as e:
            logger.error(f"Cache read error: {str(e)}")
            return None
    
    def set(self, key: str, data: Dict, expires_in: int):
        """Set cached data"""
        try:
            expires = time.time() + expires_in
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sentiment_cache (key, data, timestamp, expires) VALUES (?, ?, ?, ?)",
                    (key, json.dumps(data), time.time(), expires)
                )
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")
    
    def cleanup(self, max_age: int = 86400):
        """Clean up expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete expired entries
                conn.execute("DELETE FROM sentiment_cache WHERE expires < ?", (time.time(),))
                # Delete old entries
                conn.execute("DELETE FROM sentiment_cache WHERE timestamp < ?", 
                           (time.time() - max_age,))
                conn.execute("VACUUM")  # Optimize database size
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")

class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        """تهيئة محلل المشاعر"""
        self.config = config or {
            'max_articles': 100,
            'min_article_length': 50,
            'sources': {
                'reddit': ['cryptocurrency', 'bitcoin', 'ethereum'],
                'news': ['cointelegraph.com']
            }
        }
        self.cache = CacheManager('cache')  # استخدام CacheManager
        self.last_cleanup = time.time()
        
    def _should_cleanup_cache(self) -> bool:
        """Check if cache cleanup is needed"""
        return (time.time() - self.last_cleanup) > self.config.get('cache_cleanup_interval', 3600)
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if needed"""
        if self._should_cleanup_cache():
            self.cache.cleanup()
            self.last_cleanup = time.time()
    
    def get_sentiment_features(self, currency: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """استخراج ميزات المشاعر للعملة في الفترة المحددة"""
        try:
            dates = pd.date_range(start_date, end_date, freq='D')
            sentiments = []
            
            for date in dates:
                sentiment = self._get_daily_sentiment(currency, date)
                sentiment['date'] = date
                sentiments.append(sentiment)
            
            if not sentiments:
                return pd.DataFrame()
                
            df = pd.DataFrame(sentiments)
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {currency}: {str(e)}")
            return pd.DataFrame()

    def _get_daily_sentiment(self, currency: str, date: datetime) -> Dict:
        """حساب المشاعر اليومية"""
        try:
            # التحقق من التخزين المؤقت
            cache_key = self._get_cache_key(currency, date)
            cached_data = self._get_from_cache(cache_key)
        
            if cached_data:
                return cached_data
            
            # جمع النصوص
            texts = []
        
            # جمع النصوص من Reddit
            for subreddit in self.config['sources']['reddit']:
                reddit_texts = self._scrape_reddit(subreddit, currency, date)
                texts.extend(reddit_texts)
            
            # جمع النصوص من مصادر الأخبار
            for news_source in self.config['sources']['news']:
                news_texts = self._scrape_news(currency, news_source, date)
                texts.extend(news_texts)
            
            if not texts:
                return {
                    'sentiment_mean': 0.0,
                    'sentiment_std': 0.0,
                    'subjectivity_mean': 0.0,
                    'text_count': 0
                }
            
            # حساب المشاعر
            sentiments = []
            subjectivities = []
        
            for text in texts:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            
            result = {
                'sentiment_mean': float(np.mean(sentiments)),
                'sentiment_std': float(np.std(sentiments)),
                'subjectivity_mean': float(np.mean(subjectivities)),
                'text_count': len(texts)
            }
        
            # حفظ في التخزين المؤقت
            self._save_to_cache(cache_key, result)
        
            return result
        
        except Exception as e:
            logger.error(f"Error getting daily sentiment: {str(e)}")
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'subjectivity_mean': 0.0,
                'text_count': 0
            }

    def _get_cache_key(self, currency: str, date: datetime) -> str:
        """إنشاء مفتاح التخزين المؤقت"""
        return f"{currency}_{date.strftime('%Y-%m-%d')}"

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """استرجاع البيانات من التخزين المؤقت"""
        return self.cache.get(key)

    def _save_to_cache(self, key: str, data: Dict) -> None:
        """حفظ البيانات في التخزين المؤقت"""
        try:
            self.cache.set(key, data, expires_in=3600)  # Cache for 1 hour
        except Exception as e:
            logger.error(f"Cache write error: {str(e)}")

    def _scrape_reddit(self, subreddit: str, query: str, date: datetime) -> List[str]:
        """جمع النصوص من Reddit"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': query,
                'restrict_sr': 1,
                'sort': 'new',
                't': 'day'
            }
            response = requests.get(
                url, 
                params=params,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            data = response.json()
            texts = []
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                
                if len(title) > self.config['min_article_length']:
                    texts.append(title)
                if len(selftext) > self.config['min_article_length']:
                    texts.append(selftext)
                    
                if len(texts) >= self.config['max_articles']:
                    break
                    
            return texts
        
        except Exception as e:
            logger.warning(f"Error scraping Reddit {subreddit}: {str(e)}")
            return []

    def _scrape_news(self, currency: str, source: str, date: datetime) -> List[str]:
        """جمع النصوص من مصادر الأخبار"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            if source == 'cointelegraph.com':
                url = f"https://{source}/tags/{currency}"
            else:
                url = f"https://{source}/search?q={currency}"
                
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                # إذا فشل استخدام lxml، نستخدم html.parser كبديل
                soup = BeautifulSoup(response.text, 'html.parser')
                
            texts = []
            
            # البحث عن العناوين والمقالات
            if source == 'cointelegraph.com':
                articles = soup.find_all(['h1', 'h2', 'article'])
            else:
                articles = soup.find_all(['h1', 'h2', 'h3', 'p'])
            
            for article in articles:
                text = article.get_text().strip()
                if len(text) > self.config['min_article_length']:
                    texts.append(text)
                    
                if len(texts) >= self.config['max_articles']:
                    break
                    
            return texts
            
        except Exception as e:
            logger.warning(f"Error scraping {source}: {str(e)}")
            return []
