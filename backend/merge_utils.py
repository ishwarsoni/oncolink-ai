"""
=============================================================================
merge_utils.py - Reusable Helper Functions for Data Merging
=============================================================================

What this module provides:
    Small, focused helper functions that perform one merge operation each.
    These are used by the harmonizer to combine data from multiple documents.
    
    Each function:
    - Does ONE thing
    - Is small (usually 3-10 lines)
    - Has clear input and output
    - Handles edge cases (None values, empty lists, etc.)

Why every field needs a different merge strategy:
    You cannot merge all fields the same way. Consider:
    
    - Patient Name: Should be the SAME across all documents. Take any valid value.
    - Medications: Each document might list DIFFERENT meds. Merge into a list.
    - Biomarkers: If Doc A says "ER+" and Doc B says "ER-", that's a CONFLICT.
    - Age: All documents should agree. Take the first valid value.
    
    Different data types (identity, list, status) need different merge logic.

=============================================================================
"""

# Import logging
import logging

# Create a logger
logger = logging.getLogger(__name__)


def first_non_null(values):
    """
    Return the first non-None value from a list.
    
    What this does:
        Takes a list of values (some may be None) and returns the first
        one that is not None. If all are None, returns None.
    
    When to use:
        For fields where all documents should have the SAME value.
        Example: patient_name, age, gender.
        If Doc A says "Meera" and Doc B also says "Meera",
        we just take the first one found.
    
    Parameters:
        values (list): A list of values (may contain None)
    
    Returns:
        The first non-None value, or None if all are None
    
    Python concepts used:
        - For loop with return (exit early when found)
        - 'is not None' check (identity check, not equality)
    """
    for value in values:
        if value is not None:
            return value
    
    return None


def first_non_empty_string(strings):
    """
    Return the first non-empty string from a list.
    
    What this does:
        Similar to first_non_null but also filters out empty strings.
        Takes a list of strings and returns the first one that has
        actual content (after stripping whitespace).
    
    When to use:
        For text fields where empty strings should be ignored.
        Example: diagnosis, cancer_stage.
    
    Parameters:
        strings (list): A list of strings (may contain None, "", "   ")
    
    Returns:
        The first non-empty string, or None if all are empty
    """
    for text in strings:
        if text and isinstance(text, str) and text.strip():
            return text.strip()
    
    return None


def prefer_longest_string(strings):
    """
    Return the longest non-empty string from a list.
    
    What this does:
        Among all non-empty strings, return the longest one.
        Longer strings usually have more detail.
    
    When to use:
        For fields where a more detailed value is better.
        Example: diagnosis. "Invasive ductal carcinoma of left breast"
        is better than just "Invasive ductal carcinoma" because it
        contains more clinical detail.
    
    Parameters:
        strings (list): A list of strings
    
    Returns:
        The longest non-empty string, or None if all are empty
    """
    # Filter to only non-empty strings
    valid_strings = []
    for text in strings:
        if text and isinstance(text, str) and text.strip():
            valid_strings.append(text.strip())
    
    # Return the longest one
    if len(valid_strings) > 0:
        longest = max(valid_strings, key=len)
        return longest
    
    return None


def merge_string_lists(lists):
    """
    Merge multiple lists of strings into one deduplicated list.
    
    What this does:
        Takes several lists (each from a different document) and combines
        them into one list without duplicates.
    
    When to use:
        For fields where each document might contribute different items.
        Example: medications, where Doc A says "Trastuzumab" and Doc B
        says "Letrozole". The merged list should have both.
    
    Parameters:
        lists (list of lists): Multiple lists to merge
    
    Returns:
        list: A single deduplicated list
    
    Python concepts used:
        - Nested loop (for list in lists, for item in list)
        - Case-insensitive duplicate detection
        - Building a result list with .append()
    """
    
    # This will store unique items (tracking lowercase versions)
    seen = set()
    result = []
    
    # Loop through each input list
    for item_list in lists:
        # Make sure the item is actually a list
        if not isinstance(item_list, list):
            continue
        
        # Loop through each item in the list
        for item in item_list:
            # Convert to string for comparison
            item_str = str(item).strip()
            
            # Check if we've seen this item before (case-insensitive)
            if item_str.lower() not in seen:
                seen.add(item_str.lower())
                result.append(item)
    
    return result


def merge_unique_strings(strings):
    seen = set()
    all_items = []

    for text in strings:
        if not text or not isinstance(text, str):
            continue

        parts = text.replace(";", ",").replace("\n", ",").split(",")

        for part in parts:
            cleaned = part.strip()
            if cleaned and cleaned.lower() not in seen:
                # Check if this item is a near-duplicate of an existing one
                is_duplicate = False
                for existing in all_items:
                    # If one is a substring of the other (case-insensitive), skip shorter
                    el = existing.lower()
                    cl = cleaned.lower()
                    if len(cl) >= 10 and len(el) >= 10:
                        if cl in el or el in cl:
                            is_duplicate = True
                            # Keep the longer one
                            if len(cl) > len(el):
                                all_items.remove(existing)
                                seen.discard(el)
                                all_items.append(cleaned)
                                seen.add(cl)
                            break
                if not is_duplicate:
                    seen.add(cleaned.lower())
                    all_items.append(cleaned)

    if len(all_items) > 0:
        return "; ".join(all_items)

    return None


def merge_biomarker_lists(biomarker_lists):
    """
    Merge biomarker lists from multiple documents.
    
    What this does:
        Each document might report different biomarkers:
        - Doc A (Pathology): ER+, PR+, HER2-
        - Doc B (Lab): Ki-67 25%
        
        The merged list should include ALL unique biomarkers.
        If the same biomarker appears in multiple docs with different
        values, we keep all of them (for conflict detection in Phase 6).
    
    Parameters:
        biomarker_lists (list of lists): Multiple biomarker lists
    
    Returns:
        list: A merged list of unique biomarkers
    
    Each biomarker is a dict with: name, value, status
    Two biomarkers are considered the "same" if they have the same name.
    """
    
    seen_names = set()
    result = []
    
    for bm_list in biomarker_lists:
        if not isinstance(bm_list, list):
            continue
        
        for biomarker in bm_list:
            if not isinstance(biomarker, dict):
                continue
            
            # Get the biomarker name (case-insensitive)
            bm_name = biomarker.get("name", "")
            bm_name_lower = bm_name.lower().strip()
            
            if bm_name_lower and bm_name_lower not in seen_names:
                seen_names.add(bm_name_lower)
                result.append(biomarker)
            elif bm_name_lower in seen_names:
                # Same biomarker name already seen — log this
                logger.info(f"Duplicate biomarker '{bm_name}' found in multiple documents")
    
    return result


def merge_numbers(numbers):
    """
    Merge numeric values, preferring valid numbers and ignoring None.
    
    What this does:
        Takes a list of numbers (possibly None) and returns the first
        valid non-None number. All documents should agree on numeric
        values like age or ECOG score.
    
    Parameters:
        numbers (list): A list of numeric values or None
    
    Returns:
        The first valid number, or None if all are None
    """
    for num in numbers:
        if num is not None:
            return num
    
    return None
