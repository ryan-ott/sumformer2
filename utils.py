from datasets import load_dataset, concatenate_datasets, Dataset


def load_reddit(train_split, val_split, min_len=50):
    """Concatenate the short and long reddit TIFU datasets and split into train, validation, and test sets. Keep only the docs and their summary."""
    dataset_short = load_dataset("reddit_tifu", "short")
    dataset_short = dataset_short.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'tldr'])
    dataset_short = dataset_short.rename_columns({'documents': 'document', 'title': 'summary'})

    dataset_long = load_dataset("reddit_tifu", "long")
    dataset_long = dataset_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'title'])
    dataset_long = dataset_long.rename_columns({'documents': 'document', 'tldr': 'summary'})

    dataset = concatenate_datasets([dataset_short["train"], dataset_long["train"]])

    # Filtering out too short documents and summaries
    dataset = filter(min_len, dataset)

    # Split the dataset into train, validation, and test sets after shuffling
    train_dataset, val_dataset, test_dataset = split_data(dataset, train_split, val_split)

    # Turn the dictionaries into huggingface datasets
    train_dataset = Dataset.from_dict(train_dataset)
    val_dataset = Dataset.from_dict(val_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    return train_dataset, val_dataset, test_dataset


def split_data(dataset, train_split, val_split):
    # Shuffle the dataset
    dataset = dataset.shuffle()
    dataset = dataset.flatten_indices()  # rewrite the shuffled dataset on disk again as contiguous chunks for speed

    # Split the dataset
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    return train_dataset, val_dataset, test_dataset


def filter(min_len, dataset):
    return dataset.filter(lambda x:
                             len(x["document"]) > min_len
                             and len(x["summary"]) > min_len
                             and len(x["document"]) > len(x["summary"]))
