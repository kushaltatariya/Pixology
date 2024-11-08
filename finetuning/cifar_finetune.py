from sklearn.metrics import accuracy_score
import argparse
import numpy as np
from datasets import load_dataset, load_metric
import torch
from transformers import TrainingArguments, Trainer

from pixel import ViTForImageClassification, get_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Team-PIXEL/pixel-base')
args = parser.parse_args()

print(f"Finetuning {args.model} on CIFAR 100")

metric = load_metric("accuracy")

train_ds = load_dataset("uoft-cs/cifar100", split='train')
valid_ds = load_dataset("uoft-cs/cifar100", split='test')


id2label = {id:label for id, label in enumerate(train_ds.features['fine_label'].names)}
label2id = {label:id for id,label in id2label.items()}

transforms = get_transforms(
    do_resize=True,
    rgb=True,
    do_squarify=True,
    size=(224, 224)
)
def image_preprocess(examples):
    examples["pixel_values"] = [transforms(image) for image in examples["img"]]
    return examples

train_ds.set_transform(image_preprocess)
valid_ds.set_transform(image_preprocess)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def resize_model_embeddings(model):
    old_pos_embeds = model.vit.embeddings.position_embeddings[:, : 197, :]
    model.vit.embeddings.position_embeddings.data = old_pos_embeds.clone()
    model.config.image_size = [224, 224]
    model.image_size = [224, 224]
    model.vit.embeddings.patch_embeddings.image_size = [224, 224]


model = ViTForImageClassification.from_pretrained(args.model,
                                                  id2label=id2label,
                                                  label2id=label2id)
model.to("cuda")

if "pixel" in args.model:
    resize_model_embeddings(model)

metric_name = "accuracy"

batch_size = 32
lr = 2e-5
epochs = 20

args = TrainingArguments(f"{args.model}-cifar-100-{batch_size}-{lr}-{epochs}",
                         save_strategy="epoch",
                         save_total_limit=1,
                         warmup_steps=100,
                         evaluation_strategy="epoch",
                         learning_rate=lr,
                         per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=8,
                         num_train_epochs=epochs,
                         weight_decay=0.01,
                         load_best_model_at_end=True,
                         metric_for_best_model=metric_name,
                         logging_dir='logs',
                         remove_unused_columns=False,
                         bf16=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
