from splitclassifier import SplitClassifier
from datasets import load_dataset
import argparse
import numpy as np
import torch
import pickle
from tqdm import tqdm
from pixel import ViTModel, get_transforms, PangoCairoTextRenderer, get_attention_mask, PIXELConfig, PangoCairoBigramsRenderer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Team-PIXEL/pixel-base')
parser.add_argument("--layer", type=str, default='all',
                    help="Layer to extract embeddings from, or 'all' for all layers")
parser.add_argument("--output_file", type=str, default="results",
                    help='Output file name for results, without the file extension')
args = parser.parse_args()

def resize_model_embeddings(model):
    old_pos_embeds = model.embeddings.position_embeddings[:, : 2, :]
    model.embeddings.position_embeddings.data = old_pos_embeds.clone()
    model.config.image_size = [16, 16]
    model.image_size = [16, 16]
    model.embeddings.patch_embeddings.image_size = [16, 16]

def batcher(model, batch):
    transforms = get_transforms(
        do_resize=True,
        size=(16, 16),
        rgb=False)

    encodings = [transforms(b) for b in batch]
    attention_mask = [get_attention_mask(1, 1) for i in encodings]
    mask = torch.stack(attention_mask).cuda()
    batch = torch.stack(encodings).cuda()
    print("Computing embeddings...")
    with torch.no_grad():
        outputs, pooled_output, hidden_states, _ = model(batch, attention_mask=mask, return_dict=False)

    extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)

    embeddings = {}
    for layer in range(1,13):
        output = hidden_states[int(layer)]
        output = extended_mask * output
        output = torch.sum(output, -2) / torch.sum(mask, -1).unsqueeze(-1)
        embeddings[layer] = output.data.cpu().numpy()

    return embeddings



dataset = load_dataset("mnist")
dataset_val = dataset['train'].train_test_split(test_size=10000)
dataset["train"] = dataset_val["train"]
dataset["validation"] = dataset_val["test"]
dataset["test"] = dataset["test"]
print(f"Loaded MNIST \n {dataset}")

config = PIXELConfig.from_pretrained(args.model)
config.output_hidden_states = True
config.output_attentions = True

model = ViTModel.from_pretrained(args.model, config=config)

model.to("cuda")

resize_model_embeddings(model)
model.eval()

task_embed = {'train': {}, 'validation': {}, 'test': {}}
batch_size = 5

for split in dataset:
    indexes = list(range(len(dataset[split]['label'])))
    layer_embs = {1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [],
                  8 : [], 9 : [], 10 : [], 11 : [], 12 : []}

    task_embed[split]['X'] = {}
    for i in tqdm(range(0,len(dataset[split]['label']), batch_size)):
        batch = dataset[split]['image'][i:i+batch_size]
        embs = batcher(model, batch)
        for k, v in embs.items():
            layer_embs[k].append(embs[k])
    for layer in range(1,13):
        task_embed[split]['X'][layer] = np.vstack(layer_embs[layer])
        task_embed[split]['y'] = np.array(dataset[split]['label'])
        task_embed[split]['idx'] = np.array(indexes)

assert task_embed['train']['X'][1].shape[0] == task_embed['train']['y'].shape[0] == task_embed['train']['idx'].shape[0]

params_classifier = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}
config_classifier = {'nclasses': 10, 'seed': 1223,
                             'usepytorch': True,
                             'classifier': params_classifier}
results = {}
if args.layer == 'all':
    for layer in tqdm(range(1, 13)):
        print(f"Training classifier on embeddings from layer {layer}...")
        clf = SplitClassifier(X={'train': task_embed['train']['X'][layer],
                                     'valid': task_embed['validation']['X'][layer],
                                     'test': task_embed['test']['X'][layer]},
                                  y={'train': task_embed['train']['y'],
                                     'valid': task_embed['validation']['y'],
                                     'test': task_embed['test']['y']},
                                  config=config_classifier)

        devacc, testacc, predictions = clf.run()
        results[layer] = (devacc, testacc, predictions)
        print(f"Dev acc : {devacc} Test acc : {testacc} on {layer} for MNIST classification")
else:
    print(f"Training classifier on embeddings from layer {args.layer}...")
    clf = SplitClassifier(X={'train': task_embed['train']['X'][args.layer],
                             'valid': task_embed['validation']['X'][args.layer],
                             'test': task_embed['test']['X'][args.layer]},
                          y={'train': task_embed['train']['y'],
                             'valid': task_embed['validation']['y'],
                             'test': task_embed['test']['y']},
                          config=config_classifier)
    devacc, testacc, predictions = clf.run()
    results[args.layer] = (devacc, testacc, predictions)
    print(f"Dev acc : {devacc} Test acc : {testacc} on layer {args.layer} for MNIST classification")


with open(f'{args.output_file}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
