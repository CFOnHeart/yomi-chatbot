import re

def clean_markdown_format(text):
    """
    Remove markdown code block formatting from the beginning and end of text.

    Args:
        text (str): Text that may contain markdown code blocks

    Returns:
        str: Cleaned text with markdown formatting removed
    """
    if not text or not isinstance(text, str):
        return text

    text = text.strip()

    # Pattern to match code blocks at the beginning: ```language_name
    # and at the end: ```
    # e.g., ```json ```
    start_pattern = r'^```\w*\s*'
    end_pattern = r'\s*```$'
    text = re.sub(start_pattern, '', text)
    text = re.sub(end_pattern, '', text)
    return text.strip()