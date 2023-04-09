import re

import pandas as pd
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from stop_words import get_stop_words

from utils import read_config


def run():
    config = read_config()
    convert = config['preprocessing']['options']['convert']['numbers']
    garbage = config['preprocessing']['garbage']
    words = config['preprocessing']['words']
    rt = RegexpTokenizer(r'\w+')
    morph = pymorphy2.MorphAnalyzer()
    hashed_stop_words = set([hash(word) for word in get_stop_words('ru') + garbage])

    df = pd.read_csv('data/dataset.csv')

    df["preprocessed_text"] = df.text \
        .apply(lambda lst: rt.tokenize(re.sub('http[s]?://\S+|\S+.ru|\S+.com|#\S+', '', lst))) \
        .apply(lambda lst: [word for word in lst if word.isalpha() or convert]) \
        .apply(lambda lst: [word.lower() for word in lst]) \
        .apply(lambda lst: [word for word in lst if hash(word) not in hashed_stop_words]) \
        .apply(lambda lst: [morph.parse(word)[0].normal_form for word in lst]) \
        .apply(lambda lst: [words.get(word, word) for word in lst]) \
        .apply(lambda lst: [num2words(word, lang='ru') if word.isdecimal() else word for word in lst]) \
        .apply(lambda lst: " ".join(lst))

    print(f'{"=" * 20}====Raw====={"=" * 20}\n\n{df["text"][0]}\n\n'
          f'{"=" * 20}Preprocessed{"=" * 20}\n\n{df["preprocessed_text"][0]}')

    df.to_csv('data/dataset_extended_semantics_preprocessed')


run()
