"""
Centralized date parsing utilities for EMem conversation processing.

This module provides a unified interface for parsing various date formats
commonly found in conversation datasets, with extensible support for new formats.
"""

from datetime import datetime
from typing import Optional, List, Union
import logging
import re

logger = logging.getLogger(__name__)


class DateFormatRegistry:
    """Registry for date format patterns with priority ordering."""
    
    def __init__(self):
        # Format patterns in order of preference (most specific first)
        self._formats = [
            # Standard LoCoMo format: "1:56 pm on 8 May, 2023"
            "%I:%M %p on %d %B, %Y",
            # Alternative formats with different time representations
            "%H:%M on %d %B, %Y",  # "13:56 on 8 May, 2023"
            "%I:%M%p on %d %B, %Y",  # "1:56pm on 8 May, 2023" (no space)
            # LongMemEval format: "2023/05/30 (Tue) 23:40" - put first for priority
            "%Y/%m/%d (%a) %H:%M",  # "2023/05/30 (Tue) 23:40"
            "%Y/%m/%d (%A) %H:%M",  # "2023/05/30 (Tuesday) 23:40"
            # Date-only formats
            "%d %B, %Y",  # "8 May, 2023"
            "%d %B %Y",   # "8 May 2023"
            "%Y-%m-%d",   # "2023-05-08"
            "%m/%d/%Y",   # "05/08/2023"
            "%d/%m/%Y",   # "08/05/2023"
            "%Y/%m/%d",   # "2023/05/30" (LongMemEval date-only)
            # ISO format
            "%Y-%m-%dT%H:%M:%S",  # "2023-05-08T13:56:00"
            "%Y-%m-%dT%H:%M:%SZ", # "2023-05-08T13:56:00Z"
            # Formats with weekdays (for future extension)
            "%A, %d %B, %Y",      # "Monday, 8 May, 2023"
            "%a, %d %B, %Y",      # "Mon, 8 May, 2023"
            "%A %I:%M %p on %d %B, %Y",  # "Monday 1:56 pm on 8 May, 2023"
            "%a %I:%M %p on %d %B, %Y",  # "Mon 1:56 pm on 8 May, 2023"
        ]
        
        # Regex patterns for preprocessing (to handle variations)
        self._preprocessing_patterns = [
            # Normalize AM/PM variations
            (r'\b([ap])\.?m\.?\b', r'\1m'),  # "a.m." -> "am", "p.m." -> "pm"
            # Normalize whitespace
            (r'\s+', ' '),
            # Handle ordinal numbers (1st, 2nd, 3rd, etc.)
            (r'\b(\d+)(?:st|nd|rd|th)\b', r'\1'),
        ]
    
    def add_format(self, format_pattern: str, priority: int = None):
        """Add a new date format pattern to the registry.
        
        Args:
            format_pattern: strptime format string
            priority: Position to insert (0 = highest priority). If None, appends to end.
        """
        if priority is None:
            self._formats.append(format_pattern)
        else:
            self._formats.insert(priority, format_pattern)
        logger.info(f"Added date format pattern: {format_pattern}")
    
    def get_formats(self) -> List[str]:
        """Get all registered format patterns in priority order."""
        return self._formats.copy()
    
    def preprocess_date_string(self, date_str: str) -> str:
        """Apply preprocessing patterns to normalize date string."""
        normalized = date_str.strip()
        
        for pattern, replacement in self._preprocessing_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized.strip()


# Global date format registry
_date_registry = DateFormatRegistry()


def parse_date_string(date_str: Union[str, datetime, None], 
                     fallback_formats: Optional[List[str]] = None) -> Optional[datetime]:
    """
    Parse a date string into a datetime object using registered formats.
    
    Args:
        date_str: Date string to parse, datetime object, or None
        fallback_formats: Additional format patterns to try if registry fails
        
    Returns:
        Parsed datetime object or None if parsing fails
        
    Examples:
        >>> parse_date_string("1:56 pm on 8 May, 2023")
        datetime.datetime(2023, 5, 8, 13, 56)
        
        >>> parse_date_string("8 May, 2023")
        datetime.datetime(2023, 5, 8, 0, 0)
        
        >>> parse_date_string("Mon 2:30 pm on 15 June, 2024")
        datetime.datetime(2024, 6, 15, 14, 30)
    """
    # Handle None or empty strings
    if not date_str:
        return None
    
    # If already a datetime object, return as-is
    if isinstance(date_str, datetime):
        return date_str
    
    # Ensure we have a string
    date_str = str(date_str)
    
    # Preprocess the date string
    normalized_date = _date_registry.preprocess_date_string(date_str)
    
    # Try registered formats first
    formats_to_try = _date_registry.get_formats()
    
    # Add fallback formats if provided
    if fallback_formats:
        formats_to_try.extend(fallback_formats)
    
    # Attempt parsing with each format
    for fmt in formats_to_try:
        try:
            parsed_date = datetime.strptime(normalized_date, fmt)
            logger.debug(f"Successfully parsed '{date_str}' using format '{fmt}' -> {parsed_date}")
            return parsed_date
        except ValueError:
            logger.debug(f"Failed to parse '{date_str}' using format '{fmt}'. Trying with another date format pattern.")
            continue
    
    # Log warning if all formats failed
    logger.warning(f"Failed to parse date string: '{date_str}'. Consider adding a new format pattern.")
    return None


def format_date_for_display(dt: Optional[datetime], format_str: str = "%d %B, %Y") -> str:
    """
    Format a datetime object for display purposes.
    
    Args:
        dt: Datetime object to format
        format_str: strftime format string
        
    Returns:
        Formatted date string or "Unknown" if dt is None
    """
    if dt is None:
        return "Unknown"
    
    return dt.strftime(format_str)


def add_date_format(format_pattern: str, priority: int = None):
    """
    Add a new date format pattern to the global registry.
    
    Args:
        format_pattern: strptime format string
        priority: Position to insert (0 = highest priority). If None, appends to end.
    """
    _date_registry.add_format(format_pattern, priority)


def get_supported_formats() -> List[str]:
    """Get list of all supported date format patterns."""
    return _date_registry.get_formats()


def is_date_parseable(date_str: str) -> bool:
    """
    Check if a date string can be parsed with current formats.
    
    Args:
        date_str: Date string to test
        
    Returns:
        True if parseable, False otherwise
    """
    return parse_date_string(date_str) is not None


# Convenience functions for common operations
def parse_session_date(session_date_str: str) -> Optional[datetime]:
    """Parse a session date string (convenience wrapper)."""
    return parse_date_string(session_date_str)


def parse_edu_date(edu_date_str: str) -> Optional[datetime]:
    """Parse an EDU date string (convenience wrapper)."""
    return parse_date_string(edu_date_str)


def compare_dates(date1: Union[str, datetime, None], 
                 date2: Union[str, datetime, None]) -> Optional[int]:
    """
    Compare two dates and return -1, 0, or 1.
    
    Args:
        date1, date2: Date strings or datetime objects to compare
        
    Returns:
        -1 if date1 < date2, 0 if equal, 1 if date1 > date2, None if either is unparseable
    """
    dt1 = parse_date_string(date1)
    dt2 = parse_date_string(date2)
    
    if dt1 is None or dt2 is None:
        return None
    
    if dt1 < dt2:
        return -1
    elif dt1 > dt2:
        return 1
    else:
        return 0


def filter_dates_in_range(dates: List[Union[str, datetime]], 
                         start_date: Union[str, datetime, None] = None,
                         end_date: Union[str, datetime, None] = None) -> List[datetime]:
    """
    Filter dates within a specified range.
    
    Args:
        dates: List of date strings or datetime objects
        start_date: Start of range (inclusive), None for no lower bound
        end_date: End of range (inclusive), None for no upper bound
        
    Returns:
        List of datetime objects within the specified range
    """
    parsed_dates = []
    start_dt = parse_date_string(start_date) if start_date else None
    end_dt = parse_date_string(end_date) if end_date else None
    
    for date in dates:
        dt = parse_date_string(date)
        if dt is None:
            continue
            
        # Check range constraints
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue
            
        parsed_dates.append(dt)
    
    return sorted(parsed_dates)


def convert_date_format(date_str: str, 
                       source_format: str = "longmemeval", 
                       target_format: str = "locomo") -> Optional[str]:
    """
    Convert date string from one format to another.
    
    Args:
        date_str: Date string to convert
        source_format: Source format type ("longmemeval" or "locomo")
        target_format: Target format type ("longmemeval" or "locomo")
        
    Returns:
        Converted date string or None if conversion fails
        
    Examples:
        >>> convert_date_format("2023/05/30 (Tue) 23:40", "longmemeval", "locomo")
        "11:40 pm on 30 May, 2023"
        
        >>> convert_date_format("1:56 pm on 8 May, 2023", "locomo", "longmemeval")
        "2023/05/08 (Mon) 13:56"
    """
    # Parse the input date
    dt = parse_date_string(date_str)
    if dt is None:
        logger.warning(f"Failed to parse date string: {date_str}")
        return None
    
    # Convert to target format
    if target_format == "locomo":
        # LoCoMo format: "4:10 pm on 26 October, 2023" - preserve exact format
        formatted = dt.strftime("%I:%M %p on %d %B, %Y")
        # Fix formatting to match LoCoMo exactly:
        # 1. Remove leading zero from hour
        if formatted.startswith('0'):
            formatted = formatted[1:]
        # 2. Convert AM/PM to lowercase
        formatted = formatted.replace(' AM', ' am').replace(' PM', ' pm')
        # 3. Remove leading zero from day (e.g., "08" -> "8")
        formatted = re.sub(r' on 0(\d)', r' on \1', formatted)
        return formatted
    elif target_format == "longmemeval":
        # LongMemEval format: "2023/05/30 (Tue) 23:40"
        return dt.strftime("%Y/%m/%d (%a) %H:%M")
    else:
        logger.error(f"Unsupported target format: {target_format}")
        return None


def format_date_for_dataset(dt: Optional[datetime], 
                           dataset_type: str = "locomo") -> str:
    """
    Format a datetime object for a specific dataset format.
    
    Args:
        dt: Datetime object to format
        dataset_type: Dataset format type ("locomo" or "longmemeval")
        
    Returns:
        Formatted date string or "Unknown" if dt is None
    """
    if dt is None:
        return "Unknown"
    
    if dataset_type == "locomo":
        # LoCoMo format: "4:10 pm on 26 October, 2023" - preserve exact format
        formatted = dt.strftime("%I:%M %p on %d %B, %Y")
        # Fix formatting to match LoCoMo exactly:
        # 1. Remove leading zero from hour
        if formatted.startswith('0'):
            formatted = formatted[1:]
        # 2. Convert AM/PM to lowercase
        formatted = formatted.replace(' AM', ' am').replace(' PM', ' pm')
        # 3. Remove leading zero from day (e.g., "08" -> "8")
        formatted = re.sub(r' on 0(\d)', r' on \1', formatted)
        return formatted
    elif dataset_type == "longmemeval":
        # LongMemEval format: "2023/05/30 (Tue) 23:40" - preserve full date info including weekday and time
        return dt.strftime("%Y/%m/%d (%a) %H:%M")
    else:
        logger.warning(f"Unknown dataset type: {dataset_type}, using default format")
        return format_date_for_display(dt)
