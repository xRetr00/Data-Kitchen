"""
Sentiment analysis module for Data Kitchen
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
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
        """
        Initialize SentimentAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'max_articles': 100,
            'cache_duration': 3600,
            'batch_size': 10,
            'min_article_length': 100,
            'max_workers': 4,
            'cache_cleanup_interval': 86400,  # 24 hours
            'sources': {
                'reddit': ['r/cryptocurrency', 'r/bitcoin', 'r/ethereum'],
                'news': ['cointelegraph.com', 'coindesk.com']
            }
        }
        self.cache = CacheManager('cache/sentiment')
        self.last_cleanup = time.time()
        
    def _should_cleanup_cache(self) -> bool:
        """Check if cache cleanup is needed"""
        return (time.time() - self.last_cleanup) > self.config['cache_cleanup_interval']
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if needed"""
        if self._should_cleanup_cache():
            self.cache.cleanup()
            self.last_cleanup = time.time()
    
    def get_sentiment_features(self, pair: str, start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """
        Get sentiment features for a trading pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date for sentiment analysis
            end_date: End date for sentiment analysis
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            self._cleanup_cache_if_needed()
            
            # Get base currency (e.g., 'BTC' from 'BTC/USDT')
            currency = pair.split('/')[0].lower()
            
            # Initialize sentiment data
            dates = pd.date_range(start_date, end_date, freq='D')
            sentiment_data = []
            
            # Process dates in parallel using ProcessPoolExecutor for CPU-bound tasks
            with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # Create tasks for each date
                future_to_date = {
                    executor.submit(self._get_daily_sentiment, currency, date): date
                    for date in dates
                }
                
                # Process results as they complete
                for future in future_to_date:
                    try:
                        sentiment_data.append(future.result())
                    except Exception as e:
                        logger.error(f"Error processing date {future_to_date[future]}: {str(e)}")
            
            # Create DataFrame
            df = pd.DataFrame(sentiment_data)
            if df.empty:
                return pd.DataFrame()
                
            df.set_index('date', inplace=True)
            
            # Resample to hourly data and forward fill
            df = df.resample('1H').ffill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {pair}: {str(e)}")
            return pd.DataFrame()
    
    def _get_daily_sentiment(self, currency: str, date: datetime) -> Dict:
        """Get sentiment data for a specific day"""
        cache_key = hashlib.md5(f"{currency}_{date.date()}".encode()).hexdigest()
        
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Collect text data from various sources using ThreadPoolExecutor for I/O-bound tasks
            texts = []
            with ThreadPoolExecutor(max_workers=self.config['batch_size']) as executor:
                # Reddit scraping tasks
                reddit_futures = [
                    executor.submit(self._scrape_reddit, currency, subreddit, date)
                    for subreddit in self.config['sources']['reddit']
                ]
                
                # News scraping tasks
                news_futures = [
                    executor.submit(self._scrape_news, currency, domain, date)
                    for domain in self.config['sources']['news']
                ]
                
                # Collect results
                for future in reddit_futures + news_futures:
                    try:
                        texts.extend(future.result())
                    except Exception as e:
                        logger.warning(f"Error in scraping task: {str(e)}")
            
            # Filter and clean texts
            texts = [
                text for text in texts 
                if len(text) >= self.config['min_article_length']
            ][:self.config['max_articles']]
            
            # Calculate sentiment metrics
            sentiments = [TextBlob(text).sentiment for text in texts]
            
            data = {
                'date': date,
                'sentiment_mean': np.mean([s.polarity for s in sentiments]) if sentiments else 0,
                'sentiment_std': np.std([s.polarity for s in sentiments]) if sentiments else 0,
                'subjectivity_mean': np.mean([s.subjectivity for s in sentiments]) if sentiments else 0,
                'text_count': len(texts)
            }
            
            # Cache the result
            self.cache.set(cache_key, data, self.config['cache_duration'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting daily sentiment for {currency} on {date}: {str(e)}")
            return {
                'date': date,
                'sentiment_mean': 0,
                'sentiment_std': 0,
                'subjectivity_mean': 0,
                'text_count': 0
            }
    
    def _scrape_reddit(self, currency: str, subreddit: str, date: datetime) -> List[str]:
        """Scrape Reddit posts (using public RSS feeds)"""
        texts = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.rss"
            params = {
                'q': currency,
                't': 'day',
                'restrict_sr': 'on'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')
                for entry in soup.find_all('entry'):
                    title = entry.find('title')
                    content = entry.find('content')
                    if title and content:
                        texts.append(f"{title.text} {content.text}")
            
        except Exception as e:
            logger.warning(f"Error scraping Reddit {subreddit}: {str(e)}")
        
        return texts
    
    def _scrape_news(self, currency: str, domain: str, date: datetime) -> List[str]:
        """Scrape news articles"""
        texts = []
        
        try:
            url = f"https://{domain}/search"
            params = {
                'q': currency,
                'date': date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = soup.find_all('article') or soup.find_all(class_=re.compile('article|post'))
                
                for article in articles:
                    text = article.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text)
                    if text:
                        texts.append(text)
        
        except Exception as e:
            logger.warning(f"Error scraping {domain}: {str(e)}")
        
        return texts
