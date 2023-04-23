import re

import pandas as pd
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from stop_words import get_stop_words
from tqdm import tqdm

from utils import read_config


def run():
    tqdm.pandas()
    config = read_config()
    convert = config['preprocessing']['options']['convert']['numbers']
    garbage = config['preprocessing']['garbage']
    words = config['preprocessing']['words']
    words = words if words else {}

    rt = RegexpTokenizer(r'\w+')
    morph = pymorphy2.MorphAnalyzer()
    hashed_stop_words = set([hash(word) for word in get_stop_words('ru') + garbage])

    df = pd.read_csv('data/dataset_extended.csv')
    print(f'Initial size: {df.shape[0]}')
    df.drop(df[df.len_text == 0].index, inplace=True)
    print(f'Size after first drop: {df.shape[0]}')

    df["preprocessed_text"] = df.text \
        .progress_apply(lambda lst: rt.tokenize(re.sub('http[s]?://\S+|\S+.ru|\S+.com|#\S+', '', lst))) \
        .progress_apply(lambda lst: [word for word in lst if word.isalpha() or convert]) \
        .progress_apply(lambda lst: [word.lower() for word in lst]) \
        .progress_apply(lambda lst: [word for word in lst if hash(word) not in hashed_stop_words]) \
        .progress_apply(lambda lst: [morph.parse(word)[0].normal_form for word in lst]) \
        .progress_apply(lambda lst: [words.get(word, word) for word in lst]) \
        .progress_apply(lambda lst: [num2words(word, lang='ru') if word.isdecimal() else word for word in lst]) \
        .progress_apply(lambda lst: " ".join(lst))

    print(f'{"=" * 15}====Raw====={"=" * 15}\n\n{df["text"][0]}\n\n'
          f'{"=" * 15}Preprocessed{"=" * 15}\n\n{df["preprocessed_text"][0]}')

    df.preprocessed_text.dropna(inplace=True)
    print(f'Size after second drop: {df.shape[0]}')
    df.to_csv('data/dataset_preprocessed.csv', index=False)


run()
