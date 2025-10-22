from typing import Any, Callable, Iterable, List
from torchvision import transforms as F
from dataclasses import dataclass
from lbm.config import BaseConfig

# A callable that takes batchsize, collation_fn and returns a function that takes an iterable of samples and yields batches
BatchedFn = Callable[
    [int, Callable], # arguments: batchsize, collation_fn
    Callable[ # returns: a function that takes an iterable of samples and yields batches
        [Iterable[dict[str, Any]]], Iterable[dict[str, Any]]
    ]
]

# Inspired by https://github.com/webdataset/webdataset/blob/0773837ecd298587fc89c4f944ef346ef1a6b619/src/webdataset/filters.py#L764
# and https://github.com/NVlabs/Sana/blob/48ec430646bc524bbf0bb5d68bf092739b0f5082/diffusion/utils/data_sampler.py#L81-L99
def bucketing_batch(
    bucket_key: str,
    partial: bool,
    max_buckets: int = 300,
) -> BatchedFn:
    """
    Aspect ratio bucketing for webdataset
    To be used instead of wds.batched in the DataPipeline

    Args:

        bucket_key (str):
            Key in the sample that contains the image, to use to get sample size. Defaults to "image".
        
        partial (bool):
            Whether to send the last batch if it's smaller than the specified batch size. Defaults to False

        max_buckets (int):
            Maximum number of different aspect ratios to allow. Defaults to 300.

    Returns:
        BatchedFn: A function that behaves like wds.batched but groups samples by aspect ratio
    """
    def batched_fn(batchsize: int, collation_fn: Callable) -> Callable:

        _buckets: dict[str, List[dict[str, Any]]] = {}

        def bucketing_fn(data: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
            for batch in data:
                data_bucket_key = str(batch[bucket_key])

                if data_bucket_key not in _buckets:
                    _buckets[data_bucket_key] = []
                    n_buckets = len(_buckets)
                    assert n_buckets <= max_buckets, (
                        f"Too many different aspect ratios, "
                        f"please make sure the images sizes are standardized"
                        f"Current number of aspect ratios: {n_buckets}"
                    )

                bucket = _buckets[data_bucket_key]
                bucket.append(batch)

                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == batchsize:
                    yield collation_fn(bucket)
                    bucket.clear()

            for bucket in _buckets.values():
                if bucket:  # do not collate_fn empty buckets
                    if not partial or len(bucket) == batchsize:
                        yield collation_fn(bucket)
                    bucket.clear()

        return bucketing_fn

    return batched_fn