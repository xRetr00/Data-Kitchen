"""
اختبارات لوحدة تحليل المشاعر
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from ..sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def sample_texts():
    """نصوص نموذجية للاختبار"""
    return [
        "Bitcoin price surges to new all-time high as institutional investors show strong interest",
        "Market crash: Cryptocurrency prices plummet amid regulatory concerns",
        "Neutral news about blockchain technology development",
        "This is a very positive development for the future of digital currencies",
        "Investors remain cautious as market volatility continues"
    ]

@pytest.fixture
def mock_response():
    """استجابة وهمية للطلبات"""
    mock = MagicMock()
    mock.status_code = 200
    mock.text = """<?xml version="1.0" encoding="UTF-8"?>
    <feed>
        <entry>
            <title>Test Title</title>
            <content>Test Content</content>
        </entry>
    </feed>"""
    return mock

def test_sentiment_analyzer_init():
    """اختبار تهيئة محلل المشاعر"""
    analyzer = SentimentAnalyzer()
    assert analyzer.config['max_articles'] > 0
    assert isinstance(analyzer.config['sources'], dict)
    assert hasattr(analyzer, 'cache')

@patch('requests.get')
def test_scrape_reddit(mock_get):
    """اختبار جمع النصوص من Reddit"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'data': {
            'children': [
                {
                    'data': {
                        'title': 'Test Title ' * 10,  
                        'selftext': 'Test Content ' * 10
                    }
                }
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response
    
    analyzer = SentimentAnalyzer()
    texts = analyzer._scrape_reddit('cryptocurrency', 'bitcoin', datetime.now())
    
    assert len(texts) == 2  
    assert 'Test Title' in texts[0]
    assert 'Test Content' in texts[1]

@patch('requests.get')
def test_scrape_news(mock_get, mock_response):
    """اختبار جمع الأخبار"""
    mock_get.return_value = mock_response
    analyzer = SentimentAnalyzer()
    
    texts = analyzer._scrape_news('bitcoin', 'cointelegraph.com', datetime.now())
    assert isinstance(texts, list)

def test_get_daily_sentiment():
    """اختبار حساب المشاعر اليومية"""
    analyzer = SentimentAnalyzer()
    date = datetime.now().replace(microsecond=0)
    
    # تهيئة محلل مع مصادر محددة للاختبار
    analyzer.config['sources'] = {
        'reddit': ['cryptocurrency'],
        'news': ['cointelegraph.com']
    }
    
    reddit_texts = ['Bitcoin price surges to new all-time high']
    news_texts = [
        'Market crash: Cryptocurrency prices plummet',
        'Neutral news about blockchain development'
    ]
    
    with patch.object(analyzer, '_scrape_reddit', return_value=reddit_texts), \
         patch.object(analyzer, '_scrape_news', return_value=news_texts):
        
        result = analyzer._get_daily_sentiment('bitcoin', date)
        
        assert isinstance(result, dict)
        assert 'sentiment_mean' in result
        assert 'sentiment_std' in result
        assert 'subjectivity_mean' in result
        assert 'text_count' in result
        
        # طباعة القيم للتشخيص
        print(f"Reddit texts: {reddit_texts}")
        print(f"News texts: {news_texts}")
        print(f"Total expected: {len(reddit_texts + news_texts)}")
        print(f"Result text_count: {result['text_count']}")
        
        assert result['text_count'] == len(reddit_texts + news_texts)

def test_get_sentiment_features():
    """اختبار توليد ميزات المشاعر"""
    analyzer = SentimentAnalyzer()
    start_date = datetime.now().replace(microsecond=0) - timedelta(days=2)
    end_date = datetime.now().replace(microsecond=0)
    
    mock_sentiment = {
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
    """اختبار التخزين المؤقت للمشاعر"""
    analyzer = SentimentAnalyzer()
    date = datetime.now().replace(microsecond=0)
    currency = 'bitcoin'

    # وهمية لجمع النصوص والتخزين المؤقت
    mock_sentiment = {
        'sentiment_mean': 0.5,
        'sentiment_std': 0.1,
        'subjectivity_mean': 0.5,
        'text_count': 1
    }

    with patch.object(analyzer, '_scrape_reddit', return_value=['test']), \
         patch.object(analyzer, '_scrape_news', return_value=['test']):

        # أول استدعاء يجب أن يحسب المشاعر
        result1 = analyzer._get_daily_sentiment(currency, date)

        # ثاني استدعاء يجب أن يستخدم التخزين المؤقت
        result2 = analyzer._get_daily_sentiment(currency, date)

        assert result1 == result2

def test_error_handling():
    """اختبار معالجة الأخطاء"""
    analyzer = SentimentAnalyzer()
    
    # اختبار مع تواريخ غير صالحة
    df = analyzer.get_sentiment_features(
        'INVALID/PAIR',
        datetime.now(),
        datetime.now() - timedelta(days=1)  # النهاية قبل البداية
    )
    assert df.empty  # يجب أن يعيد DataFrame فارغ عند الخطأ

@patch('requests.get')
def test_request_error_handling(mock_get):
    """اختبار معالجة أخطاء الطلبات"""
    mock_get.side_effect = Exception("Network error")
    analyzer = SentimentAnalyzer()
    
    texts = analyzer._scrape_reddit('bitcoin', 'cryptocurrency', datetime.now())
    assert len(texts) == 0  # يجب أن يعيد قائمة فارغة عند الخطأ
