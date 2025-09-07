# -*- coding: utf-8 -*-
# Text Preprocessing & Geospatial Indexing with H3

"""
This script is part of the supplementary material for the paper:

    Assessing Place Similarity by Inferring and Modeling Place Semantics 
    from Geosocial Media Data

It preprocesses geosocial media data by:
- Removing duplicates,
- Cleaning and normalizing text fields (tags, titles, bodies),
- Adding spatial indexing using the H3 hexagonal hierarchical geospatial index.

Data collection:

Instagram data were gathered in July 2018 using the Netlytic platform 
(Grudz, 2016), just before Instagram discontinued its public API. 
Flickr and Twitter data were collected via their respective public APIs.

Data availability:

The original geosocial media data used in this study, comprising geotagged posts from 
Instagram, Flickr, and X (formerly Twitter), cannot be publicly shared. All data
utilized in this study was collected through official APIs or authorized services
that, at the time of collection, explicitly prohibited the redistribution of 
user-generated content in accordance with their respective terms of service
(e.g., the Twitter Developer Agreement and Instagram's API Terms of Use). 
While these specific agreements are no longer publicly available due to platform 
ownership changes and service restructuring, their restrictions were in effect and
adhered to during the data acquisition period.

"""

import os
import regex as re
import emoji
import pandas as pd
import regex
import h3


def fix_mojibake(text):
    """Fix mojibake text artifacts."""
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text
    return text


def remove_emojis(text):
    """Remove emojis from text."""
    return emoji.replace_emoji(text, replace='')


# Precompiled regex patterns
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_MENTION_NUMERIC_PATTERN = re.compile(r'https?://\S+|www\.\S+|@\w+|\b\w*\d\w*\b')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
NON_LETTER_PATTERN = re.compile(r'[^\p{L}\s]', flags=regex.UNICODE)
MULTI_SPACE_PATTERN = re.compile(r'\s+')


def preprocess_text(text):
    """Clean and normalize social media text."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = fix_mojibake(text)
    text = remove_emojis(text)
    text = HTML_TAG_PATTERN.sub('', text)
    text = HASHTAG_PATTERN.sub(r'\1', text)
    text = URL_MENTION_NUMERIC_PATTERN.sub('', text)
    text = NON_LETTER_PATTERN.sub(' ', text)
    text = text.lower()
    text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
    return text


def add_h3_column(dataframe, lat_column, long_column, resolution):
    """Add H3 spatial index column to DataFrame."""
    h3_column_name = f'h3_index_{resolution}'
    dataframe[h3_column_name] = dataframe.apply(
        lambda row: h3.geo_to_h3(row[long_column], row[lat_column], resolution),
        axis=1
    )


def main():
    """Main function to preprocess the dataset."""
    folder_path = './'
    input_path = os.path.join(folder_path, '01_Input')
    output_path = os.path.join(folder_path, '02_Output')
    input_file = os.path.join(input_path, 'YFCC_Dresden_Subset.csv')
    output_file = os.path.join(output_path, 'Dresden_GSMDataset_preprocessed.csv')

    # Load dataset
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Remove bulk uploads
    df.drop_duplicates(
        subset=['origin_id', 'user_guid', 'tags', 'post_title', 'post_body'],
        keep='first',
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    # Drop rows with all text fields missing
    df.dropna(subset=['tags', 'post_title', 'post_body'], how='all', inplace=True)

    # Combine text columns
    df['post_text'] = (
        df['tags'].fillna('') + ' ' +
        df['post_title'].fillna('') + ' ' +
        df['post_body'].fillna('')
    )

    # Clean and preprocess text
    df['text'] = df['post_text'].apply(preprocess_text)

    # Remove empty or too short texts
    df = df[df['text'].str.split().apply(len) >= 5]
    df.reset_index(drop=True, inplace=True)

    # Add H3 index (resolution 9)
    add_h3_column(df, lat_column='longitude', long_column='latitude', resolution=9)

    # Save to file
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
