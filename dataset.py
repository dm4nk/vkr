import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import vk_api
import yaml
from tqdm import tqdm

VK = 'vk'
dull = lambda x: x
t_count = lambda x: x.get('count') if x else None
features = {
    'date': dull,
    'text': dull,
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


def read_config():
    with open('config.yaml', 'r') as stream:
        return yaml.safe_load(stream)


def read_token():
    with open('token.yaml', 'r') as stream:
        return yaml.safe_load(stream)


def check_time_bounds(post, from_time_timestamp, to_time_timestamp):
    return from_time_timestamp < float(post['date']) < to_time_timestamp


def get_posts_from_group(batch, domain, items, posts, size, sleep, vk, from_time_timestamp, to_time_timestamp):
    with open('data/posts.json', 'a', encoding='utf-8') as file:
        for i in tqdm(range(0, size, batch)):
            dose = None
            while (not dose) or ('error' in dose):
                dose = vk.wall.get(domain=domain, count=batch, filter='owner', offset=i)

                with open('data/raw_posts.json', 'a', encoding='utf-8') as raw_posts_file:
                    json.dump(dose, raw_posts_file, indent=4, ensure_ascii=False)

                time.sleep(sleep)
            if 'items' in dose:
                dose['items'] = [post for post in dose['items'] if
                                 check_time_bounds(post, from_time_timestamp, to_time_timestamp)]
                if len(dose['items']) == 0:
                    break
                items.extend(dose['items'])

        for i, item in enumerate(items):
            row = {'index': i}
            for key, f in features.items():
                row[key] = f(item.get(key))
            row['domain'] = domain
            json.dump(row, file, indent=4, ensure_ascii=False)
            posts.append(row)


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

    print(vk_session.token)
    vk = vk_session.get_api()
    items = []
    posts = []

    for domain in domains:
        print(f'\n\n<{"=" * 3} Getting data from {domain} {"=" * 3}>')
        get_posts_from_group(batch, domain, items, posts, size, sleep, vk, from_time.timestamp(), to_time.timestamp())

    return posts


def get_data_frame(posts):
    df = pd.DataFrame(posts, columns=features.keys())
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['len_text'] = df.text.str.len()
    df['post_source_id'] = (df['post_source'] == 'vk').astype(int)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df.fillna(0, inplace=True)

    return df


def extract_hashtags(text):
    regex = "#\S+"
    hashtag_list = re.findall(regex, str(text))
    hashtag_str = ' '.join(hashtag_list)
    return hashtag_str, len(hashtag_list)


def extract_urls(text):
    regex = 'http[s]?://\S+|\S+.ru|\S+.com'
    url_list = re.findall(regex, str(text))
    url_str = ' '.join(url_list)
    return url_str, len(url_list)


def get_time_window(hour, night, morning, day):
    if hour < night:
        return 'night', 0
    if hour < morning:
        return 'morning', 1
    if hour < day:
        return 'day', 2
    return 'evening', 3


def run():
    config = read_config()
    token = read_token()
    source = config['dataset']['source']
    posts = get_posts_from_vk(config, token) if source == VK else None
    df = get_data_frame(posts)
    df.to_csv('data/dataset.csv')


def extend():
    config = read_config()
    df = pd.read_csv('data/2_try/dataset.csv')[:10]
    domains = df.domain.unique()
    domains_dict = dict(zip(domains, range(len(domains))))
    windows = config['dataset']['time']['windows']
    night, morning, day = windows['night'], windows['morning'], windows['day']

    hash_tags = [extract_hashtags(text) for text in df.text]
    df = df.join(pd.DataFrame(hash_tags, columns=['hashtags', 'hashtag_count']))

    urls = [extract_urls(text) for text in df.text]
    df = df.join(pd.DataFrame(urls, columns=['urls', 'url_count']))

    time_windows = [get_time_window(hour, night, morning, day) for hour in df.hour]
    df = df.join(pd.DataFrame(time_windows, columns=['time_window', 'time_window_id']))

    df['domain_id'] = df.domain.map(domains_dict)
    df['log1p_comments'] = np.log1p(df.comments)
    df['log1p_likes'] = np.log1p(df.likes)
    df['log1p_reposts'] = np.log1p(df.reposts)
    df['log1p_views'] = np.log1p(df.views)

    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv('data/dataset_united_extended.csv')


def unite():
    CHUNK_SIZE = 5000
    csv_file_list = ['data/1_try/dataset.csv', 'data/2_try/dataset.csv']
    output_file = 'data/dataset_united.csv'

    first_one = True
    for csv_file_name in csv_file_list:

        if not first_one:  # if it is not the first csv file then skip the header row (row 0) of that file
            skip_row = [0]
        else:
            skip_row = []

        chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE, skiprows=skip_row)
        for chunk in chunk_container:
            chunk.to_csv(output_file, mode="a", index=False)
        first_one = False


# run()
# extend()
unite()
