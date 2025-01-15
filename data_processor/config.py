"""
إعدادات معالج البيانات
"""

import os
from datetime import datetime, timedelta

# إعدادات API
BINANCE_API_KEY = ""  # أدخل مفتاح API الخاص بك هنا
BINANCE_SECRET_KEY = ""  # أدخل المفتاح السري هنا

# أزواج العملات والإطارات الزمنية
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']  # أزواج العملات الرئيسية
TIMEFRAMES = ['1h', '4h', '1d']  # الإطارات الزمنية الأكثر استخدامًا

# إعدادات فترة البيانات التاريخية
YEARS_OF_DATA = 3  # عدد السنوات المطلوبة
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA * 365)
END_DATE = datetime.now()

# حساب عدد الشموع المطلوبة لكل إطار زمني
TIMEFRAME_CHUNKS = {
    '1h': int((END_DATE - START_DATE).total_seconds() / 3600),  # عدد الساعات في 3 سنوات
    '4h': int((END_DATE - START_DATE).total_seconds() / (3600 * 4)),  # عدد فترات 4 ساعات
    '1d': int((END_DATE - START_DATE).days)  # عدد الأيام
}

# إعدادات معالجة البيانات
CHUNK_SIZE = max(TIMEFRAME_CHUNKS.values())  # تعيين حجم الدفعة ليناسب أكبر فترة زمنية
MAX_WORKERS = 4    # عدد العمليات المتزامنة

# مسارات الملفات
DATA_DIR = "data"
LOG_DIR = "logs"
PARQUET_DIR = "data/parquet"

# إعدادات المؤشرات الفنية
TECHNICAL_INDICATORS = {
    'RSI': {'period': 14},  # مؤشر القوة النسبية
    'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},  # مؤشر MACD
    'SMA': {'periods': [20, 50, 200]},  # المتوسط المتحرك البسيط
    'EMA': {'periods': [9, 21, 50]},  # المتوسط المتحرك الأسي
}

# إعدادات التسجيل
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/data_processor.log"

# تحقق من أن المسارات موجودة
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(PARQUET_DIR):
    os.makedirs(PARQUET_DIR)

# print time period information
print(f"""
Time period information:
--------------------------------
Start date: {START_DATE}
End date: {END_DATE}
Number of candles required for each time frame:
- 1h: {TIMEFRAME_CHUNKS['1h']} candles
- 4h: {TIMEFRAME_CHUNKS['4h']} candles
- 1d: {TIMEFRAME_CHUNKS['1d']} candles
""")