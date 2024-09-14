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
from prefect import task, Flow
from collections import defaultdict, Counter
from prefect.executors import DaskExecutor
from typing import Tuple, Any, List, Dict

HOME = Path.home()
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")


@task(name="Read User Data", checkpoint=False)
def read_twits(data_file: str) -> None:
    res_file = (HOME / os.environ.get("RESULTS_PREFIX") / f"{data_file}-graph.pkl")
    if res_file.exists():
        return
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    graph = defaultdict(lambda: defaultdict(int))
    year = data_file[29:33]
    month = data_file[34:36]
    day = data_file[37:39]
    tweet2author = ujson.load(open(
        f"/Users/anodot_1/prefect_results/twits/{year}-{month}/ukraine_russia-{year}-{month}-{day}.jsonl.gz-tweet2author.pkl",
        "rt"))
    # f"/Users/anodot_1/prefect_results/twits/{month}/ukraine_russia-{year}-{month}-{day}.jsonl.gz-tweet2author.pkl"
    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        # try:
        for line in tqdm(f.readlines()):
            data = ujson.loads(line.decode().strip("\n"))
            for tweet in data["data"]:
                if "referenced_tweets" in tweet:
                    for ref in tweet["referenced_tweets"]:
                        if ref["id"] in tweet2author:
                            graph[tweet["author_id"]][tweet2author[ref["id"]]] += 1
        # except Exception as e:
        #     logger.error(f"Error reading file {data_file}, error: {e}")
    ujson.dump(graph, res_file.open("wt"))


@task(name="Collect Files")
def collect_files(s3_path: str = "twits/") -> list:
    logger = prefect.context.get("logger")
    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files if i["Key"].endswith(".gz")]
    logger.info(f"Discovered files - {files}")
    return files



EXECUTOR = DaskExecutor(
    cluster_kwargs={"n_workers": 4, "threads_per_worker": 1}
)


def twitter_flow():
    with Flow("User Stats",
              # run_config=RUN_CONFIG
              executor=EXECUTOR
              ) as f:
        files = collect_files()
        read_twits.map(files)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run()
