"""
Date Conversion Utilities
------------------------
Helpers for working with Persian/Shamsi calendar dates.
"""

import datetime
import re

def gregorian_to_shamsi(date=None):
    """
    Convert Gregorian date to Shamsi (Persian/Iranian) date.
    
    Args:
        date: datetime object (defaults to current date if None)
    
    Returns:
        str: Formatted Shamsi date string (e.g., "۱۴۰۳/۰۲/۱۱")
    """
    if date is None:
        date = datetime.datetime.now()
    
    # Gregorian year, month and day
    year = date.year
    month = date.month
    day = date.day
    
    # Convert to Shamsi (Persian) calendar
    # Algorithm based on conversion tables
    d_4 = year % 4
    g_a = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    doy_g = g_a[month] + day
    
    if d_4 == 0 and month > 2:
        doy_g += 1
    
    d_33 = int(((year - 16) % 132) * 0.0305)
    a = 286 if (d_33 == 3 or d_33 < (d_4 - 1) or d_4 == 0) else 287
    
    if (d_33 == 1 or d_33 == 2) and (d_33 == d_4 or d_4 == 1):
        b = 78
    else:
        b = 80 if (d_33 == 3 and d_4 == 0) else 79
    
    if int((year - 10) / 63) == 30:
        a -= 1
        b += 1
    
    if doy_g > b:
        jy = year - 621
        doy_j = doy_g - b
    else:
        jy = year - 622
        doy_j = doy_g + a
    
    if doy_j < 187:
        jm = int((doy_j - 1) / 31)
        jd = doy_j - (31 * jm)
        jm += 1
    else:
        jm = int((doy_j - 187) / 30)
        jd = doy_j - 186 - (jm * 30)
        jm += 7
    
    # Format with leading zeros
    shamsi_date = f"{jy:04d}/{jm:02d}/{jd:02d}"
    
    # Convert digits to Persian numerals if needed
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    for i in range(10):
        shamsi_date = shamsi_date.replace(str(i), persian_digits[i])
    
    return shamsi_date

def get_shamsi_month_name(date=None):
    """
    Get the name of the current Shamsi month.
    
    Args:
        date: datetime object (defaults to current date if None)
    
    Returns:
        str: Persian month name
    """
    if date is None:
        date = datetime.datetime.now()
    
    # Get the Shamsi date string
    shamsi_date = gregorian_to_shamsi(date)
    
    # Extract month number (remove Persian digits first)
    for i, digit in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        shamsi_date = shamsi_date.replace(digit, str(i))
    
    month_num = int(re.search(r'\d{4}/(\d{2})/', shamsi_date).group(1))
    
    # Persian month names
    month_names = [
        "فروردین", "اردیبهشت", "خرداد", 
        "تیر", "مرداد", "شهریور", 
        "مهر", "آبان", "آذر", 
        "دی", "بهمن", "اسفند"
    ]
    
    return month_names[month_num - 1]
def get_full_shamsi_date():
    """
    Returns the current date in multiple formats: Shamsi, Gregorian, and Hijri.
    
    Returns:
        str: Formatted date string with all three calendar systems
    """
    now = datetime.datetime.now()
    shamsi_date = gregorian_to_shamsi(now)
    
    # Extract Shamsi components
    for i, digit in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        shamsi_date = shamsi_date.replace(digit, str(i))
    
    year, month, day = shamsi_date.split('/')
    year_persian = "".join(["۰۱۲۳۴۵۶۷۸۹"[int(d)] for d in year])
    day_persian = "".join(["۰۱۲۳۴۵۶۷۸۹"[int(d)] for d in day])
    
    month_name = get_shamsi_month_name(now)
    shamsi_full = f"{day_persian} {month_name} {year_persian}"
    
    # Gregorian date
    gregorian_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    gregorian_full = f"{now.day} {gregorian_months[now.month - 1]} {now.year}"
    
    # Hijri date calculation (approximate)
    # Basic Hijri conversion (Julian day based)
    julian_day = int(now.toordinal() + 1948440.5)
    hijri_year = int((julian_day - 1948440.5 - 227015) / 354.367)
    remaining_days = julian_day - int(1948440.5 + 227015 + hijri_year * 354.367)
    
    # Hijri months (29/30 days alternating, with adjustments)
    hijri_months = [
        "محرم", "صفر", "ربیع الاول", "ربیع الثانی", "جمادی الاول", "جمادی الثانی",
        "رجب", "شعبان", "رمضان", "شوال", "ذیقعده", "ذیحجه"
    ]
    
    days_in_months = [30, 29, 30, 29, 30, 29, 30, 29, 30, 29, 30, 29]
    hijri_month = 1
    hijri_day = remaining_days
    
    for i, days in enumerate(days_in_months):
        if hijri_day <= days:
            hijri_month = i + 1
            break
        hijri_day -= days
    
    if hijri_day <= 0:
        hijri_day = 1
    
    hijri_year += 1444  # Approximate base year adjustment
    hijri_day_persian = "".join(["۰۱۲۳۴۵۶۷۸۹"[int(d)] for d in str(int(hijri_day))])
    hijri_year_persian = "".join(["۰۱۲۳۴۵۶۷۸۹"[int(d)] for d in str(hijri_year)])
    hijri_full = f"{hijri_day_persian} {hijri_months[hijri_month - 1]} {hijri_year_persian}"
    
    return f"شمسی: {shamsi_full} | میلادی: {gregorian_full} | قمری: {hijri_full}"