ROUND = 1   # round of testing query
TOPK = 10   # knn
EF = 400    # EF for indexing

datasets = {
    # "SSN": "l2",
    # "SPACE1B": "l2",
    "sift1B": "l2",
    # "deep1M": "l2",
    # "gist": "l2",
    # "msong": "l2",
    # "tiny5m": "l2",
    # "imagenet": "l2",
    # "deep100M": "l2",
    # "msturing100M": "l2"
}


degrees = {
    "sift1B": [32],
    # "SSN": [32],
    # "SPACE1B": [32],
    # "deep1M": [32, 64, 128],
    # "gist": [32, 64, 128],
    # "msong": [32, 64, 128],
    # "tiny5m": [32, 64, 128],
    # "imagenet": [32, 64, 128],
    # "deep100M": [32],
    # "msturing100M": [32],
}

iter = {
    "sift1B": 3,
    # "SSN": 3,
    # "SPACE1B": 3,
    # "deep1M": 3,
    # "gist": 3,
    # "msong": 3,
    # "tiny5m": 3,
    # "imagenet": 3,
    # "deep100M": 3,
    # "msturing100M": 4
}