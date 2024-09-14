from dotenv import load_dotenv

load_dotenv()

import os
import io
import boto3
import ujson
import prefect
import pickle
from tqdm import tqdm
from pathlib import Path
from gzip import GzipFile
from prefect import task, Flow, unmapped
from collections import defaultdict, Counter
from prefect.executors import DaskExecutor
from typing import Tuple, Any, List
from datetime import datetime
import pandas as pd
import spacy
from prefect.engine.results import LocalResult

HOME = Path.home()
BASE_DIR = HOME / 'prefect_results/twits/'
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

sentiment = pd.read_csv(HOME / 'Documents/osi/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', delimiter='\t',
                        header=None, names=['word', 'emotion', 'score'])

most_f_60k = set(pd.read_csv(HOME / "twitter-uk-ru-2022/data/wordlist_60k_g803.txt", sep="\t").lemma.values)


results = LocalResult(dir=str(HOME / os.environ.get("RESULTS_PREFIX")))


@task(name="Collect Files")
def collect_files(s3_path: str = "twits/") -> list:
    # TODO: uk|ru pattern
    return list(BASE_DIR.glob("*/*.jsonl.gz-all.pkl"))


@task(name="Clean corpus", checkpoint=True, result=results, target="{file}_all_clean_text.df")
def clean_texts(file):
    corpus = pickle.load(file.open("rb"))
    nlp = spacy.load('en_core_web_sm')
    def cleaning(doc):
        _txt = [token.lemma_.lower().strip()
                if token.lemma_ != "-PRON-"
                else token.lower_
                for token in doc
                if not token.is_stop
                and len(token.lemma_) > 1
                and token.lemma_ != 'amp'
                ]  # tokenize, remove stopwords, lowercase
        if len(_txt) > 1:
            return ' '.join([i for i in _txt if i in most_f_60k])

    _results = []
    for clean_doc in [cleaning(doc) for doc in nlp.pipe(texts=corpus, batch_size=600)]:
        if clean_doc:
            _results.append(clean_doc)

    return _results


@task(name="Analyze Texts", checkpoint=True, result=results, target="{file}_all_emotions.df")
def analyze_texts(file, texts):
    anger = list(sentiment[(sentiment.emotion == 'anger') & (sentiment.score != 0)].word.values)
    anticipation = list(sentiment[(sentiment.emotion == 'anticipation') & (sentiment.score != 0)].word.values)
    disgust = list(sentiment[(sentiment.emotion == 'disgust') & (sentiment.score != 0)].word.values)
    fear = list(sentiment[(sentiment.emotion == 'fear') & (sentiment.score != 0)].word.values)
    joy = list(sentiment[(sentiment.emotion == 'joy') & (sentiment.score != 0)].word.values)
    sadness = list(sentiment[(sentiment.emotion == 'sadness') & (sentiment.score != 0)].word.values)
    surprise = list(sentiment[(sentiment.emotion == 'surprise') & (sentiment.score != 0)].word.values)
    trust = list(sentiment[(sentiment.emotion == 'trust') & (sentiment.score != 0)].word.values)

    all_tokens = [j for i in texts if i for j in i.split(" ") if j]
    known_only = [i for i in all_tokens if i in most_f_60k]
    cnt = Counter(known_only)

    return {
        "anger": sum([cnt[i] for i in anger]) / len(known_only),
        "anticipation": sum([cnt[i] for i in anticipation]) / len(known_only),
        "disgust": sum([cnt[i] for i in disgust]) / len(known_only),
        "fear": sum([cnt[i] for i in fear]) / len(known_only),
        "joy": sum([cnt[i] for i in joy]) / len(known_only),
        "sadness": sum([cnt[i] for i in sadness]) / len(known_only),
        "surprise": sum([cnt[i] for i in surprise]) / len(known_only),
        "trust": sum([cnt[i] for i in trust]) / len(known_only),
    }


EXECUTOR = DaskExecutor(
    cluster_kwargs={"n_workers": 6, "threads_per_worker": 2}
)


def twitter_flow():
    with Flow("User Stats",
              # run_config=RUN_CONFIG
              executor=EXECUTOR
              ) as f:
        files = collect_files()
        texts = clean_texts.map(files)
        analyze_texts.map(files, texts)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run()
