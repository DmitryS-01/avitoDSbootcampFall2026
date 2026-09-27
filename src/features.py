"""Повторяемые item- и query–item-признаки, без скрытой разметки."""
import re
import numpy as np
import pandas as pd
from .data import normalize


def prepare_items(frame):
    items = frame.copy().reset_index(drop=True)
    for col in ['item_price', 'item_latitude', 'item_longitude', 'item_rating', 'item_rating_reviews_count']:
        items[col] = pd.to_numeric(items[col], errors='coerce').astype(float)
    for col, name in [('item_title_raw', 'title_text'), ('item_infm_params_text', 'params_text'), ('item_description_raw', 'description_text')]:
        items[name] = items[col].map(normalize)
    # Lexical видит больше описания, dense укладывается в ограниченный token budget.
    items['lexical_text'] = items.title_text + ' ' + items.title_text + ' ' + items.params_text.str[:800] + ' ' + items.description_text.str[:2400]
    items['dense_text'] = items.title_text + '. ' + items.params_text.str[:350] + '. ' + items.description_text.str[:650]
    return items


def item_features(items):
    result = pd.DataFrame(index=items.index)
    for col in ['item_price', 'item_rating_reviews_count']:
        value = pd.to_numeric(items[col], errors='coerce')
        result[f'{col}_missing'] = value.isna().astype('int8')
        if col == 'item_price':
            result['item_price_unknown'] = (value.isna() | value.lt(0)).astype('int8')
            value = value.where(value.ge(0))
        result[f'{col}_log1p'] = np.log1p(value.clip(lower=0))
        result[f'{col}_zero'] = value.eq(0).astype('int8')
    result['rating'] = items.item_rating
    result['rating_missing'] = items.item_rating.isna().astype('int8')
    reviews = items.item_rating_reviews_count.fillna(0).clip(lower=0)
    # Prior считается по каталогу, а не по выбранным пользователями объявлениям.
    prior = items.item_rating.median()
    result['rating_shrunk'] = (reviews * items.item_rating.fillna(prior) + 20 * prior) / (reviews + 20)
    for col in ['title_text', 'params_text', 'description_text']:
        result[col + '_loglen'] = np.log1p(items[col].str.len())
        result[col + '_empty'] = items[col].eq('').astype('int8')
    result['phone_hidden'] = items.item_is_phone_hidden.astype('int8')
    result['message_forbidden'] = items.item_is_message_forbidden.astype('int8')
    result['both_contacts_restricted'] = result.phone_hidden * result.message_forbidden
    price = items.item_price.where(items.item_price.ge(0))
    median = price.groupby(items.item_microcat_id).transform('median')
    result['price_vs_microcat_log'] = np.log1p(price) - np.log1p(median)
    result['latitude'] = items.item_latitude.where(items.item_latitude.between(-90, 90))
    result['longitude'] = items.item_longitude.where(items.item_longitude.between(-180, 180))
    return result.astype('float32')


def pair_features(query, positions, items, numeric, counts=None, centers=None):
    subset = items.iloc[positions]
    result = numeric.iloc[positions].reset_index(drop=True).copy()
    tokens = set(query.text_key.split())
    denom = max(1, len(tokens))
    for field in ['title_text', 'params_text', 'description_text']:
        result[field + '_coverage'] = [sum((' '+t+' ') in (' '+x+' ') for t in tokens) / denom for x in subset[field]]
    result['query_tokens'] = len(tokens)
    result['query_chars'] = len(query.text_key)
    result['filter_chars'] = len(str(query.search_infm_params_text))
    result['same_location'] = subset.item_location_id.eq(query.search_location_id).to_numpy().astype('int8')
    result['same_category'] = subset.item_category_id.eq(query.search_category).to_numpy().astype('int8')
    result['delivery'] = int(query.search_is_delivery_search)
    result['item_pop_log'] = np.log1p([counts.get(i, 0) for i in subset.item_id]) if counts is not None else 0.
    result['distance_km_proxy'] = np.nan
    if centers is not None and query.search_location_id in centers.index:
        center = centers.loc[query.search_location_id]
        lat1, lon1 = np.radians([center.item_latitude, center.item_longitude])
        lat2 = np.radians(subset.item_latitude.to_numpy())
        lon2 = np.radians(subset.item_longitude.to_numpy())
        a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
        result['distance_km_proxy'] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    match = re.search(r'(\d(?:[.,]\d)?)\s*звезд', str(query.search_infm_params_text).lower())
    result['rating_threshold_requested'] = int(match is not None)
    result['rating_margin'] = subset.item_rating.to_numpy() - float(match[1].replace(',', '.')) if match else np.nan
    return result.replace([np.inf, -np.inf], np.nan).astype('float32')
