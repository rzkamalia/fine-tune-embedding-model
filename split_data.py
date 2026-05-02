from datasets import DatasetDict, load_dataset


# load dataset
ds = load_dataset("sentence-transformers/natural-questions")

# split dataset
train_test_split = ds['train'].train_test_split(test_size=0.3)
eval_test_split = train_test_split['test'].train_test_split(test_size=0.1)
ds_split = DatasetDict({
    'train': train_test_split['train'],
    'eval': eval_test_split['train'],
    'test': eval_test_split['test']
})

# save dataset splits so they can be reused later
ds_split["train"].save_to_disk("dataset/train-data")
ds_split["eval"].save_to_disk("dataset/eval-data")
ds_split["test"].save_to_disk("dataset/test-benchmark")