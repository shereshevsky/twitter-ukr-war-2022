from typing import Tuple, Any, List

from dotenv import load_dotenv
load_dotenv()

import os
import io
import boto3
import prefect
from tqdm import tqdm
from prefect import task, Flow, Parameter, unmapped
from prefect.engine.results import LocalResult
import spacy
import re
import ujson
import pandas as pd
from pathlib import Path
from gzip import GzipFile
from emoji import UNICODE_EMOJI
from collections import Counter, defaultdict
from gensim.models.phrases import Phrases, Phraser, FrozenPhrases

nlp = spacy.load('en_core_web_sm')

text_in_brackets = re.compile("\[(.*?)\]")
non_alpha = re.compile("[^A-Za-z']+")

HOME = Path.home()

S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

results = LocalResult(dir=str(HOME / os.environ.get("RESULTS_PREFIX")))

s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)


@task(name="Read Text From Tweeter Files", checkpoint=True, result=results, target="{data_file}_network_tuples.df")
def read_data(data_file: str) -> list:
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    network_tuples = []

    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().strip("\n"))
                for tweet in data.get("data", []):
                    _referenced_tweets = tweet.get("referenced_tweets", [])
                    _id = tweet.get("id")
                    _author_id = tweet.get("author_id")
                    _retweets = [i["id"] for i in _referenced_tweets if i["type"] == "retweeted"]
                    _created_at = tweet.get("created_at")
                    network_tuples.append((_id, _author_id, _retweets, _created_at))
        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")

    return network_tuples


@task(name="Collect Files")
def collect_files(s3_path: str) -> list:
    logger = prefect.context.get("logger")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files]
    logger.info(f"Discovered files - {files}")
    return files


@task(name="read tuples", checkpoint=True, result=results, target="authors_network_tuples.df")
def read_tuples(texts):
    authors = defaultdict(list)
    for i in texts:
        for tw in i:
            authors[tw[1]].append(tw[0])
    return authors


def twitter_flow():
    with Flow("Twitter") as f:
        p_path = Parameter(name='p_path', required=True)
        files = collect_files(p_path)
        texts = read_data.map(files)
        tuples = read_tuples(texts)

        # word_freq_and_texts = calc_word_frequency_and_clean_texts.map(files, texts)
        # trigram_model = calc_ngrams(word_freq_and_texts)
        # ngram_frequency = tokens_frequency_with_ngrams.map(files, word_freq_and_texts, unmapped(trigram_model))
        # global_ngram_freq = calculate_global_frequency(files, ngram_frequency)
        # check_ngrams(global_ngram_freq)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run(parameters={"p_path": ""})

