"""
Tests for sentiment analysis module
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from ..sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "Bitcoin price surges to new all-time high as institutional investors show strong interest",
        "Market crash: Cryptocurrency prices plummet amid regulatory concerns",
        "Neutral news about blockchain technology development",
        "This is a very positive development for the future of digital currencies",
        "Investors remain cautious as market volatility continues"
    ]

@pytest.fixture
def mock_response():
    """Mock response for requests"""
    mock = MagicMock()
    mock.status_code = 200
    mock.text = """
    <?xml version="1.0" encoding="UTF-8"?>
    <feed>
        <entry>
            <title>Test Title</title>
            <content>Test Content</content>
        </entry>
    </feed>
    """
    return mock

def test_sentiment_analyzer_init():
    """Test SentimentAnalyzer initialization"""
    analyzer = SentimentAnalyzer()
    assert analyzer.config['max_articles'] > 0
    assert isinstance(analyzer.config['sources'], dict)
    assert analyzer.cache == {}

@patch('requests.get')
def test_scrape_reddit(mock_get, mock_response):
    """Test Reddit scraping"""
    mock_get.return_value = mock_response
    analyzer = SentimentAnalyzer()
    
    texts = analyzer._scrape_reddit('bitcoin', 'cryptocurrency', datetime.now())
    assert len(texts) > 0
    assert isinstance(texts[0], str)

@patch('requests.get')
def test_scrape_news(mock_get, mock_response):
    """Test news scraping"""
    mock_get.return_value = mock_response
    analyzer = SentimentAnalyzer()
    
    texts = analyzer._scrape_news('bitcoin', 'cointelegraph.com', datetime.now())
    assert isinstance(texts, list)

def test_get_daily_sentiment(sample_texts):
    """Test daily sentiment calculation"""
    analyzer = SentimentAnalyzer()
    date = datetime.now()
    
    # Mock text collection
    with patch.object(analyzer, '_scrape_reddit', return_value=sample_texts[:2]), \
         patch.object(analyzer, '_scrape_news', return_value=sample_texts[2:]):
        
        result = analyzer._get_daily_sentiment('bitcoin', date)
        
        assert isinstance(result, dict)
        assert 'sentiment_mean' in result
        assert 'sentiment_std' in result
        assert 'subjectivity_mean' in result
        assert 'text_count' in result
        assert result['text_count'] == len(sample_texts)

def test_get_sentiment_features():
    """Test sentiment feature generation"""
    analyzer = SentimentAnalyzer()
    start_date = datetime.now() - timedelta(days=2)
    end_date = datetime.now()
    
    # Mock daily sentiment
    mock_sentiment = {
        'date': datetime.now(),
        'sentiment_mean': 0.5,
        'sentiment_std': 0.1,
        'subjectivity_mean': 0.5,
        'text_count': 10
    }
    
    with patch.object(analyzer, '_get_daily_sentiment', return_value=mock_sentiment):
        df = analyzer.get_sentiment_features('BTC/USDT', start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'sentiment_mean' in df.columns
        assert 'sentiment_std' in df.columns
        assert 'subjectivity_mean' in df.columns
        assert 'text_count' in df.columns

def test_sentiment_caching():
    """Test sentiment caching mechanism"""
    analyzer = SentimentAnalyzer({'cache_duration': 10})
    date = datetime.now()
    currency = 'bitcoin'
    
    # Mock text collection
    with patch.object(analyzer, '_scrape_reddit', return_value=['test']), \
         patch.object(analyzer, '_scrape_news', return_value=['test']):
        
        # First call should compute sentiment
        result1 = analyzer._get_daily_sentiment(currency, date)
        
        # Second call should use cache
        result2 = analyzer._get_daily_sentiment(currency, date)
        
        assert result1 == result2
        assert f"{currency}_{date.date()}" in analyzer.cache

def test_error_handling():
    """Test error handling in sentiment analysis"""
    analyzer = SentimentAnalyzer()
    
    # Test with invalid dates
    df = analyzer.get_sentiment_features(
        'INVALID/PAIR',
        datetime.now(),
        datetime.now() - timedelta(days=1)  # End before start
    )
    assert df.empty  # Should return empty DataFrame on error

@patch('requests.get')
def test_request_error_handling(mock_get):
    """Test handling of request errors"""
    mock_get.side_effect = Exception("Network error")
    analyzer = SentimentAnalyzer()
    
    texts = analyzer._scrape_reddit('bitcoin', 'cryptocurrency', datetime.now())
    assert len(texts) == 0  # Should return empty list on error
