"""
Sentiment analysis module for Data Kitchen
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
from .logger import setup_logger
import time

logger = setup_logger(__name__)

class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SentimentAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'max_articles': 100,  # Maximum number of articles to analyze per day
            'cache_duration': 3600,  # Cache duration in seconds
            'batch_size': 10,  # Number of articles to process in parallel
            'min_article_length': 100,  # Minimum article length in characters
            'sources': {
                'reddit': ['r/cryptocurrency', 'r/bitcoin', 'r/ethereum'],
                'news': ['cointelegraph.com', 'coindesk.com']
            }
        }
        self.cache = {}
        
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
            # Get base currency (e.g., 'BTC' from 'BTC/USDT')
            currency = pair.split('/')[0].lower()
            
            # Initialize sentiment data
            dates = pd.date_range(start_date, end_date, freq='D')
            sentiment_data = []
            
            with ThreadPoolExecutor(max_workers=self.config['batch_size']) as executor:
                # Process dates in parallel
                futures = [
                    executor.submit(self._get_daily_sentiment, currency, date)
                    for date in dates
                ]
                
                for future in futures:
                    sentiment_data.append(future.result())
            
            # Create DataFrame
            df = pd.DataFrame(sentiment_data)
            df.set_index('date', inplace=True)
            
            # Resample to hourly data and forward fill
            df = df.resample('1H').ffill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {pair}: {str(e)}")
            return pd.DataFrame()
    
    def _get_daily_sentiment(self, currency: str, date: datetime) -> Dict:
        """Get sentiment data for a specific day"""
        cache_key = f"{currency}_{date.date()}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.config['cache_duration']:
                return cached_data
        
        try:
            # Collect text data from various sources
            texts = []
            
            # Reddit posts
            for subreddit in self.config['sources']['reddit']:
                texts.extend(self._scrape_reddit(currency, subreddit, date))
            
            # News articles
            for domain in self.config['sources']['news']:
                texts.extend(self._scrape_news(currency, domain, date))
            
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
            self.cache[cache_key] = (data, time.time())
            
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
            # Use Reddit's public RSS feed
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
            # Use public RSS feeds or HTML scraping
            url = f"https://{domain}/search"
            params = {
                'q': currency,
                'date': date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract article text (customize selectors for each domain)
                articles = soup.find_all('article') or soup.find_all(class_=re.compile('article|post'))
                
                for article in articles:
                    # Get text content
                    text = article.get_text(separator=' ', strip=True)
                    # Clean text
                    text = re.sub(r'\s+', ' ', text)
                    if text:
                        texts.append(text)
        
        except Exception as e:
            logger.warning(f"Error scraping {domain}: {str(e)}")
        
        return texts
