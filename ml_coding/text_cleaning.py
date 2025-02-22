    """
    Adds columns to df indicating potential 'messy' text patterns.
    Returns the original df with additional boolean flags.
    """
    # 1. Check for missing / null
    df['is_null'] = df[text_col].isnull()

    # 2. Check for whitespace only
    df['is_whitespace'] = df[text_col].fillna('').str.strip().eq('')

    # 3. Check for HTML tags
    df['contains_html'] = df[text_col].fillna('').str.contains(r'<[^>]+>', regex=True)

    # 4. Check for URLs (http, https, www)
    df['contains_url'] = df[text_col].fillna('').str.contains(r'(http|www)', case=False, regex=True)

    # 5. Check if text is only punctuation
    def is_only_punct(text):
        if not isinstance(text, str):
            return False
        stripped = re.sub(r'\s+', '', text)  # remove spaces
        return not bool(re.search(r'\w', stripped))  # no word chars => only punct

    df['only_punct'] = df[text_col].apply(is_only_punct)

    # 6. Check for non-ASCII (possible emojis, foreign text) 
    # Better to keep them tho as they are important
    df['contains_non_ascii'] = df[text_col].fillna('').apply(lambda x: not x.isascii() if isinstance(x, str) else False)

df_checked = check_messiness(df.copy(), text_col='prompt')
df_checked[['prompt','is_null','is_whitespace','contains_html','contains_url','only_punct','contains_non_ascii']]
for col in ['is_null','is_whitespace','contains_html','contains_url','only_punct','contains_non_ascii']:
    count_flagged = df_checked[col].sum()
    print(f"{col}: {count_flagged} flagged rows")


def clean_text(text):
    """
    Cleans up text for typical bag-of-words or TF-IDF approaches.
    1. Strip whitespace
    2. Remove HTML tags
    3. Replace URLs with <URL>
    4. Remove bizarre punctuation sequences
    5. Lowercase
    """
    if not isinstance(text, str):
        return text

    # a) Strip whitespace
    text = text.strip()

    # b) Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # c) Replace URLs
    text = re.sub(r'(http\S+|www\S+)', '<URL>', text)

    # d) Possibly remove or normalize punctuation
    #    Example: keep basic punctuation like . ! ? but remove weird stuff like #$%^
    #    Adjust as you see fit.
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)

    # e) Lowercase
    text = text.lower()

    return text
  
df_clean = df_checked.copy()

# 4.1 Drop rows that are null or just whitespace
df_clean = df_clean[~df_clean['is_null'] & ~df_clean['is_whitespace']]

# 4.2 Drop rows that are only punctuation (completely uninformative)
df_clean = df_clean[~df_clean['only_punct']]

# 4.3 Apply our cleaning function
df_clean['prompt_cleaned'] = df_clean['prompt'].apply(clean_text)

# (Optional) You could choose to remove or transform non-ASCII (like emojis) if you want
# For instance:
# df_clean = df_clean[~df_clean['contains_non_ascii']]
