"""
وحدة التسجيل لاختبارات الوحدة
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_test_logger():
    """
    إعداد نظام التسجيل للاختبارات
    """
    # إنشاء مجلد للسجلات إذا لم يكن موجوداً
    log_dir = Path("logs/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # إنشاء اسم ملف السجل مع الطابع الزمني
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_results_{timestamp}.log"
    
    # تكوين التسجيل
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # للطباعة في وحدة التحكم أيضاً
        ]
    )
    
    return logging.getLogger("TestLogger")
