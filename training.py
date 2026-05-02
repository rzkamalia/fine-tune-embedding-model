from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers import sentence_transformer
from sentence_transformers.sentence_transformer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback


# load saved dataset splits
train_dataset = load_from_disk("dataset/train-data")
eval_dataset = load_from_disk("dataset/eval-data")

# load mode
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# define loss
loss = sentence_transformer.losses.MultipleNegativesRankingLoss(
    model,
    directions=("query_to_doc",),  # default
    partition_mode="joint",  # default
)

# define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="nq-e5-large-instruct/",

    # optional training parameters
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=0.0000000001,
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # optional tracking parameters
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,

    logging_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# define training function
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
        )
    ],
)

# training process
trainer.train()
trainer.evaluate()

# save the final model
model.save_pretrained("nq-e5-large-instruct/final")
