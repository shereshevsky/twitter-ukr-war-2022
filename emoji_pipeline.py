from typing import Tuple, Any, List

from dotenv import load_dotenv
load_dotenv()

import os
import io
import boto3
import ujson
import prefect
import pandas as pd
from gzip import GzipFile
from tqdm import tqdm
from emoji import UNICODE_EMOJI
from pathlib import Path
from datetime import datetime
from prefect import task, Flow, Parameter, unmapped
from collections import defaultdict, Counter
from prefect.engine.results import LocalResult
from gensim.models.phrases import Phrases, Phraser, FrozenPhrases


HOME = Path.home()
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

results = LocalResult(dir=str(HOME / os.environ.get("RESULTS_PREFIX")))

s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)


@task(name="Read Data Files", checkpoint=True, result=results, target="{data_file}_emoji_tuples.df")
def read_twits(data_file: set) -> list:
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    res = []
    total_tweets = 0
    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().strip("\n"))
                for tweet in data["data"]:
                    total_tweets += 1
                    res_tuple, is_empty = extract_emoji_tuples(tweet)
                    if not is_empty:
                        res.append(res_tuple)
        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")
    logger.info(f"Total twits parsed - {total_tweets}")
    logger.info(f"Collected {len(res)} day / lang pairs")

    with (HOME / os.environ.get("RESULTS_PREFIX") / "totals.lst").open("ta") as r:
        print(total_tweets, file=r)
    return res


def extract_emoji_tuples(tweet):
    created_at = datetime.strptime(tweet['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
    is_empty = False
    res_tuple = None
    try:
        emojis = [emoji for emoji in UNICODE_EMOJI['en'] if emoji in tweet["text"]]
        if emojis:
            res_tuple: tuple[datetime, Any, list[Any]] = (
                created_at.replace(minute=0, second=0, microsecond=0),
                tweet['lang'],
                emojis)
        else:
            is_empty = True
    except KeyError:
        is_empty = True
    return res_tuple, is_empty


@task(name="Build Emoji Matrix")
def build_matrix(tuples):
    merged = defaultdict(list)
    for i in tqdm(tuples):
        for j in i:
            merged[j[0]].extend(j[2])

    for i in tqdm(merged.keys()):
        merged[i] = Counter(merged[i])

    df = pd.DataFrame(merged)
    df.to_csv(HOME / "prefect_results" / "hashtags.csv")


@task(name="Collect Files")
def collect_files(s3_path: str) -> list:
    logger = prefect.context.get("logger")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files]
    logger.info(f"Discovered files - {files}")
    return files


@task(name="calc_ngrams", checkpoint=True, result=results, target="trigram_emoji.df")
def calc_ngrams(chunks: List):
    emojis = []
    for chunk in chunks:
        emojis.extend([i[2] for i in chunk])
    bigram_phrases = Phrases(emojis, min_count=30, progress_per=10000, max_vocab_size=200000)
    bigram = Phraser(bigram_phrases)
    trigram_phrases = Phrases(bigram[emojis], min_count=30, progress_per=10000, max_vocab_size=200000)
    trigram = Phraser(trigram_phrases)
    trigram.save(str(HOME / os.environ.get("RESULTS_PREFIX") / "trigram_emoji.phrases"))
    return trigram


@task(name="tokens_frequency_with_ngrams", result=results, target="{file}_ngram_frequency_emoji.df")
def tokens_frequency_with_ngrams(file: str, word_freq_and_texts: dict, phraser: FrozenPhrases) -> Counter:
    phraser = Phrases.load(str(HOME / os.environ.get("RESULTS_PREFIX") / "trigram_emoji.phrases"))
    return Counter([j for i in word_freq_and_texts for j in phraser[i[2]]])


@task(name="calc_ngrams_and_clean_texts", checkpoint=True, result=results, target="global_ngram_frequency_emoji.df")
def calculate_global_frequency(files: list, ngram_freq: List[Counter]):
    k = 10
    merged = defaultdict(list)
    for file, freq in zip(files, ngram_freq):
        merged[file[-22:-9]] = freq
    df = pd.DataFrame(merged)
    terms_over_k = df.sum(axis=1)[df.sum(axis=1) > k].index
    df2 = df.loc[terms_over_k]
    return df2


def twitter_flow():
    with Flow("Twitter") as f:
        p_path = Parameter(name='p_path', required=True)
        files = collect_files(p_path)
        emoji_tuples = read_twits.map(files)
        trigram_model = calc_ngrams(emoji_tuples)
        ngram_frequency = tokens_frequency_with_ngrams.map(files, emoji_tuples, unmapped(trigram_model))
        global_ngram_freq = calculate_global_frequency(files, ngram_frequency)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run(parameters={"p_path": ""})

