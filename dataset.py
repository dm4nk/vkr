import re
import time
from datetime import datetime

import emoji as emoji
import numpy as np
import pandas as pd
import vk_api
import yaml
from tqdm import tqdm

from utils import read_config, normalize_columns

VK = 'vk'
dull = lambda x: x
t_count = lambda x: x.get('count') if x else None
features = {
    'text': dull,
    'date': dull,
    'is_pinned': dull,
    'attachments': lambda x: len(x) if x else None,
    'post_source': lambda x: x.get('type') if x else None,
    'domain': dull,
    'comments': t_count,
    'likes': t_count,
    'reposts': t_count,
    'views': t_count
}


def auth_handler():
    return str(input('Enter authentication code: ').strip()), True


def captcha_handler(captcha):
    key = input("Enter captcha code {0}: ".format(captcha.get_url())).strip()
    return captcha.try_again(key)


def read_token():
    with open('token.yaml', 'r') as stream:
        return yaml.safe_load(stream)


def check_time_bounds(post, from_time_timestamp, to_time_timestamp):
    return from_time_timestamp < float(post['date']) < to_time_timestamp


def get_posts_from_group(batch, domain, size, sleep, vk, from_time_timestamp, to_time_timestamp):
    items = []
    posts = []
    for i in tqdm(range(0, size, batch)):
        dose = None
        while (not dose) or ('error' in dose):
            dose = vk.wall.get(domain=domain, count=batch, filter='owner', offset=i)

            # with open('data/raw_posts.json', 'a', encoding='utf-8') as raw_posts_file:
            #     json.dump(dose, raw_posts_file, indent=4, ensure_ascii=False)

            time.sleep(sleep)

        if 'items' in dose:
            tmp = [post for post in dose['items'] if
                   check_time_bounds(post, from_time_timestamp, to_time_timestamp)]
            if len(tmp) == 0:
                break
            items.extend(tmp)

    for i, item in enumerate(items):
        row = {'index': i}
        for key, f in features.items():
            row[key] = f(item.get(key))
        row['domain'] = domain
        posts.append(row)

    return posts


def get_posts_from_vk(config, token):
    domains = config['dataset']['vk']['domains']
    size = config['dataset']['size']
    batch = config['dataset']['batch']
    sleep = config['dataset']['vk']['sleep']
    token = token['vk'] if token else None
    from_time = datetime.strptime(config['dataset']['dates']['from'], '%d.%m.%y')
    to_time = datetime.strptime(config['dataset']['dates']['to'], '%d.%m.%y')

    if token:
        vk_session = vk_api.VkApi(token=token)
    else:
        login = config['credentials']['login']
        password = config['credentials']['password']
        vk_session = vk_api.VkApi(login, password, auth_handler=auth_handler, captcha_handler=captcha_handler)
        vk_session.auth()

    vk = vk_session.get_api()

    flag = True
    for domain in domains:
        print(f'\n\n<{"=" * 3} Getting data from {domain} {"=" * 3}>')
        posts = get_posts_from_group(batch, domain, size, sleep, vk, from_time.timestamp(), to_time.timestamp())
        df = get_data_frame(posts)
        df.to_csv('data/dataset.csv', mode='a', index=False, header=flag)
        flag = False


def get_data_frame(posts):
    df = pd.DataFrame(posts, columns=features.keys())
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['len_text'] = df.text.str.len()
    df['post_source_id'] = (df['post_source'] == 'vk').astype(int)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df.drop(df[df.len_text == 0].index, inplace=True)
    df.fillna(0, inplace=True)

    return df


def extract_hashtags(df):
    regex = ' #\S+'
    df['hashtag'] = df.text.apply(lambda txt: re.findall(regex, str(txt)))
    df['hashtag_count'] = df.hashtag.apply(lambda lst: len(lst))


def extract_urls(df):
    regex = 'http[s]?://\S+|\S+.ru|\S+.com'
    df['url'] = df.text.apply(lambda txt: re.findall(regex, str(txt)))
    df['url_count'] = df.url.apply(lambda lst: len(lst))


def extract_emojis(df):
    df['emoji'] = df.text.apply(lambda txt: list(map(lambda o: o['emoji'], emoji.emoji_list(txt))))
    df['emoji_count'] = df.emoji.apply(lambda lst: len(lst))


def get_time_window(df, night, morning, day):
    def get_time(hour):
        if hour <= night:
            return 'night'
        if hour <= morning:
            return 'morning'
        if hour <= day:
            return 'day'
        return 'evening'

    def get_time_id(hour):
        if hour <= night:
            return 0
        if hour <= morning:
            return 1
        if hour <= day:
            return 2
        return 3

    df['time_window'] = df.hour.apply(lambda h: get_time(h))
    df['time_window_id'] = df.hour.apply(lambda h: get_time_id(h))


def run():
    config = read_config()
    token = read_token()
    source = config['dataset']['source']
    get_posts_from_vk(config, token) if source == VK else None


def extend_dataset():
    config = read_config()
    df = pd.read_csv('data/dataset.csv')
    df.drop(df[df.len_text < 30].index, inplace=True)
    df.drop(df[df.len_text == 'len_text'].index, inplace=True)

    domains = df.domain.unique()
    domains_dict = dict(zip(domains, range(len(domains))))
    df['domain_id'] = df.domain.map(domains_dict)

    windows = config['dataset']['time']['windows']
    night, morning, day = windows['night'], windows['morning'], windows['day']

    print('calculating hashtags')
    extract_hashtags(df)

    print('calculating urls')
    extract_urls(df)

    print('calculating emojis')
    extract_emojis(df)

    print('calculating time_window')
    get_time_window(df, night, morning, day)

    print('calculating log1p')
    df['log1p_comments'] = np.log1p(df.comments)
    df['log1p_likes'] = np.log1p(df.likes)
    df['log1p_reposts'] = np.log1p(df.reposts)
    df['log1p_views'] = np.log1p(df.views)

    df.to_csv('data/dataset_extended.csv', index=False)

    cols_to_normalize = ['len_text', 'domain_id', 'log1p_views', 'log1p_likes', 'log1p_reposts', 'log1p_comments',
                         'hashtag_count', 'url_count', 'emoji_count', 'time_window_id', 'dayofweek', 'attachments']

    print('normalizing cloumns')
    normalize_columns(df, cols_to_normalize)

    cols = ['text', 'len_text', 'domain', 'domain_id', 'views', 'log1p_views', 'likes', 'log1p_likes', 'reposts',
            'log1p_reposts', 'comments', 'log1p_comments', 'is_pinned', 'post_source', 'post_source_id', 'hashtag',
            'hashtag_count', 'url', 'url_count', 'emoji', 'emoji_count', 'date', 'time_window', 'time_window_id',
            'year', 'month', 'dayofweek', 'hour', 'attachments'] + [col + '_normalized' for col in cols_to_normalize]

    df = df[cols]
    df.info()

    df.to_csv('data/dataset_extended.csv', index=False)


def cut_data():
    df = pd.read_csv('data/dataset_preprocessed.csv')
    df.sample(frac=0.05).to_csv('data/dataset_preprocessed_5_percent.csv')
    df.sample(frac=0.1).to_csv('data/dataset_preprocessed_10_percent.csv')
    df.sample(frac=0.3).to_csv('data/dataset_preprocessed_30_percent.csv')
    df.sample(frac=0.5).to_csv('data/dataset_preprocessed_50_percent.csv')
    df[['text', 'preprocessed_text']].to_csv('data/dataset_text_only.csv')


def add_semantics():
    df1 = pd.read_csv('data/dataset_extended.csv')
    cols = ['is_pinned', 'text', 'len_text', 'domain', 'domain_id', 'views', 'likes', 'reposts',
            'comments', 'log1p_views', 'log1p_likes', 'log1p_reposts', 'log1p_comments', 'post_source',
            'post_source_id', 'hashtag', 'hashtag_count', 'url', 'url_count', 'date', 'time_window', 'time_window_id',
            'year', 'month', 'dayofweek', 'hour', 'attachments']

    df1 = df1[cols]

    df2 = pd.read_csv('data/dataset_with_semantic.csv')
    result = pd.concat([df1, df2], axis=1, join="inner")

    result.drop('Unnamed: 0', axis=1, inplace=True)
    result.fillna(0, inplace=True)
    result.to_csv('data/dataset_extended_semantics.csv')


def clean():
    df = pd.read_csv('data/dataset_preprocessed.csv')
    df.drop(df[df.preprocessed_text.str.count(' ') + 1 < 2].index, inplace=True)
    df = df[df.preprocessed_text.notna()]
    print(df.info())
    df.to_csv('data/dataset_preprocessed.csv', mode='w')


# run()
# extend_dataset()
cut_data()
# add_semantics()
# clean()
