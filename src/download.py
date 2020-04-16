import io
import os
import random
import requests
import pandas as pd


def clean_df(df):
    df.reset_index(inplace=True, drop=True)
    df = df[['question1', 'question2', 'is_duplicate']]

    return df


def split_data(data, validataion_size=10000, test_size=10000):
    validataion_size = validataion_size // 2
    test_size = test_size // 2

    bad_indices = random.sample(list(data[data.is_duplicate == 0].index), validataion_size + test_size)
    good_indices = random.sample(list(data[data.is_duplicate == 1].index), validataion_size + test_size)

    validation_indices = bad_indices[:validataion_size] + good_indices[:validataion_size]
    test_indices = bad_indices[validataion_size:] + good_indices[validataion_size:]

    train_data = clean_df(data[~data.index.isin(bad_indices + good_indices)])
    validation_data = clean_df(data[data.index.isin(validation_indices)])
    test_data = clean_df(data[data.index.isin(test_indices)])

    return train_data, validation_data, test_data


def get_dataset(url, data_dir='./data/'):

    request = requests.get(url=url)
    data = pd.read_csv(io.StringIO(request.content.decode('utf-8')), sep='\t')

    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    for col in ['question1', 'question2']:
        data[col] = data[col].map(lambda x: x.replace('\n', ' ').replace('\r', ' '))

    train_data, validation_data, test_data = split_data(data)

    data.to_csv(os.path.join(data_dir, 'raw_data.tsv'), sep='\t')
    train_data.to_csv(os.path.join(data_dir, 'train.tsv'), index=None, header=None, sep='\t')
    validation_data.to_csv(os.path.join(data_dir, 'validation.tsv'), index=None, header=None, sep='\t')
    test_data.to_csv(os.path.join(data_dir, 'test.tsv'), index=None, header=None, sep='\t')
