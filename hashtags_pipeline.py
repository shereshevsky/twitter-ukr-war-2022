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
from prefect import task, Flow, Parameter
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from prefect.engine.results import LocalResult


HOME = Path.home()
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

results = LocalResult(dir=str(HOME / "prefect_results"))

s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)


@task(name="Read Data Files", checkpoint=True, result=results, target="{data_file}_tuples.df")
def read_twits(data_file: set) -> list:
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    res = []
    total_tweets = 0
    twits_wo_hashtag = 0
    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        for line in tqdm(f.readlines()):
            data = ujson.loads(line.decode().strip("\n"))
            for tweet in data["data"]:
                total_tweets += 1
                res_tuple, is_empty = extract_hashtag_tuples(tweet)
                if not is_empty:
                    res.append(res_tuple)
                else:
                    twits_wo_hashtag += 1

    logger.info(f"Total twits parsed - {total_tweets}")
    logger.info(f"Twits w/o hashtags - {twits_wo_hashtag}")
    logger.info(f"Collected {len(res)} day / lang pairs")
    return res


def extract_hashtag_tuples(tweet):
    created_at = datetime.strptime(tweet['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
    is_empty = False
    try:
        hash_tags = tweet['entities']['hashtags']
        if hash_tags:
            res_tuple: tuple[datetime, Any, list[Any]] = (
                created_at.replace(minute=0, second=0, microsecond=0),
                tweet['lang'],
                [i["tag"].lower() for i in hash_tags])
        else:
            is_empty = True
    except KeyError:
        is_empty = True
    return res_tuple, is_empty


@task(name="Generate Language Plot")
def plot_lang(groups: list,) -> None:
    lang = Counter(j[1] for i in groups for j in i).most_common(20)
    y = [count for tag, count in lang]
    x = [tag for tag, count in lang]
    plt.figure(figsize=(20, 10))
    plt.fontsize = 30
    plt.bar(x, y, color='crimson')
    plt.title(f"Top 20 languages in the dataset")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    for i, (tag, count) in enumerate(lang):
        plt.text(i, count, f' {count} ', rotation=90,
                 ha='center', va='top' if i < 10 else 'bottom', color='white' if i < 10 else 'black')
    plt.xlim(-0.6, len(x) - 0.4)  # optionally set tighter x lims
    plt.tight_layout()  # change the whitespace such that all labels fit nicely
    plt.savefig("lang.png")


@task(name="Generate Hash Tag Plot")
def plot_hashtags(groups: list,) -> None:
    logger = prefect.context.get("logger")
    merged = defaultdict(list)
    for i in tqdm(groups):
        for j in i:
            merged[j[0]].extend(j[2])

    for i in tqdm(merged.keys()):
        merged[i] = Counter(merged[i])

    for day in tqdm(merged.keys()):
        logger.info(f"generate plot for {day.isoformat()} hashtags")
        tf = merged[day].most_common(40)
        y = [count for tag, count in tf]
        x = [tag for tag, count in tf]
        plt.figure(figsize=(20, 10))
        plt.fontsize = 30
        plt.bar(x, y, color='crimson')
        plt.title(f"Hash Tag frequencies in Twitter Data for {day.isoformat()}")
        plt.ylabel("Frequency (log scale)")
        plt.yscale('log')  # optionally set a log scale for the y-axis
        plt.xticks(rotation=90)
        for i, (tag, count) in enumerate(tf):
            plt.text(i, count, f' {count} ', rotation=90,
                     ha='center', va='top' if i < 10 else 'bottom', color='white' if i < 10 else 'black')
        plt.xlim(-0.6, len(x) - 0.4)  # optionally set tighter x lims
        plt.tight_layout()  # change the whitespace such that all labels fit nicely
        plt.savefig(f"hashtags-{day.isoformat()}.png")


@task(name="Hashtag Dynamic")
def hashtags_dynamic(groups: list, ) -> None:
    merged = defaultdict(list)
    for i in tqdm(groups):
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


def twitter_flow():
    with Flow("Twitter") as f:
        p_path = Parameter(name='p_path', required=True)
        files = collect_files(p_path)
        data = read_twits.map(files)
        # plot_lang(data)
        # plot_hashtags(data)
        hashtags_dynamic(data)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run(parameters={"p_path": ""})

