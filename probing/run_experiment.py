from PIL import Image
import torch
import pickle
import argparse
import math
from pixel import ViTModel, get_transforms, PangoCairoTextRenderer, get_attention_mask, PIXELConfig, \
    PangoCairoBigramsRenderer, glue_strip_spaces
from transformers import AutoTokenizer, AutoModel, AutoConfig
from probing import Probing


def resize_model_embeddings(model: ViTModel, max_seq_length: int) -> None:
    """
    Checks whether position embeddings need to be resized. If the specified max_seq_length is longer than
    the model's number of patches per sequence, the position embeddings will be interpolated.
    If max_seq_length is shorter, the position embeddings will be truncated

    Args:
        model (`ViTModel`):
            The model for which position embeddings may be resized.
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """
    patch_size = model.config.patch_size
    if isinstance(model.config.image_size, tuple) or isinstance(model.config.image_size, list):
        old_height, old_width = model.config.image_size
    else:
        old_height, old_width = (model.config.image_size, model.config.image_size)

    # ppr means patches per row (image is patchified into grid of [ppr * ppr])
    old_ppr = math.sqrt(old_height * old_width) // patch_size
    new_ppr = math.sqrt(max_seq_length)

    if old_ppr < new_ppr:
        # Interpolate position embeddings
        # logger.info(f"Interpolating position embeddings to {max_seq_length}")
        model.config.interpolate_pos_encoding = True
    elif old_ppr > new_ppr:
        # logger.info(f"Truncating position embeddings to {max_seq_length}")
        # Truncate position embeddings
        old_pos_embeds = model.embeddings.position_embeddings[:, : max_seq_length + 1, :]
        model.embeddings.position_embeddings.data = old_pos_embeds.clone()
        # Update image_size
        new_height = int(new_ppr * patch_size) if old_height == old_width else int(patch_size)
        new_width = int(new_ppr * patch_size) if old_height == old_width else int(patch_size * new_ppr ** 2)
        model.config.image_size = [new_height, new_width]
        model.image_size = [new_height, new_width]
        model.embeddings.patch_embeddings.image_size = [new_height, new_width]


def batcher(params, batch, task):
    model = params["model"]
    model_name = params["model_name"]

    if "bert" in model_name:
        tokenizer = params["tokenizer"]
        # batch = [[token for token in sent] for sent in batch]
        # batch = [" ".join(sent) if sent != [] else "." for sent in batch]
        if "xlm" in model.config._name_or_path:
            batch = [["<s>"] + tokenizer.tokenize(sent) + ["</s>"] for sent in batch]
        else:
            if task in ["count_character_words_binned-4", "count_character_sentences_binned-4"]:
                batch = [["[CLS]"] + tokenizer.tokenize(sent.split("|")[0]) + ["[SEP]"] + tokenizer.tokenize(
                    sent.split("|")[1]) + ["[SEP]"] for sent in batch]
            else:
                batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
        batch = [b[:512] for b in batch]
        seq_length = max([len(sent) for sent in batch])
        mask = [[1] * len(sent) + [0] * (seq_length - len(sent)) for sent in batch]
        segment_ids = [[0] * seq_length for _ in batch]
        batch = [tokenizer.convert_tokens_to_ids(sent) + [0] * (seq_length - len(sent)) for sent in batch]
        with torch.no_grad():
            batch = torch.tensor(batch).cuda()
            mask = torch.tensor(mask).cuda()  # bs * seq_length
            segment_ids = torch.tensor(segment_ids).cuda()
            outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask,
                                                             return_dict=False)

        # extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)
        extended_mask = mask.unsqueeze(-1)

    else:
        processor = params["tokenizer"]
        format_fn = glue_strip_spaces
        if "vit" in model.config._name_or_path:
            transforms = get_transforms(
                do_resize=False,
                do_squarify=True
            )
        else:
            transforms = get_transforms(
                do_resize=True,
                size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
            )

        # batch = [[token for token in sent] for sent in batch]
        # batch = [" ".join(sent) if sent != [] else "." for sent in batch]
        if model_name in ["pixel-words", "pixel_r", "pixel-words-ud"]:
            encodings = [processor(text=a.split()) for a in batch]
        elif model_name in ["pixel-bigrams", "pixel-bigrams-ud", "pixel-small-bigrams"]:
            encodings = [processor(a, preprocessor="whitespace_only") for a in batch]
        else:
            if task in ["count_character_words_binned-4", "count_character_sentences_binned-4"]:
                encodings = [processor(text=(format_fn(a.split("|")[0]), a.split("|")[1])) for a in batch]
            encodings = [processor(text=format_fn(a)) for a in batch]
        pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        attention_mask = [
            get_attention_mask(e.num_text_patches, seq_length=processor.max_seq_length) for e in encodings
        ]

        with torch.no_grad():
            # batch = torch.stack(pixel_values)
            # mask = torch.stack(attention_mask)# bs * seq_length
            batch = torch.stack(pixel_values).cuda()
            mask = torch.stack(attention_mask).cuda()  # bs * seq_length
            outputs, pooled_output, hidden_states, _ = model(batch, attention_mask=mask, return_dict=False)

        extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)

    embeddings = {}
    for layer in range(1, 13):
        output = hidden_states[int(layer)]
        output = extended_mask * output
        output = torch.sum(output, -2) / torch.sum(mask, -1).unsqueeze(-1)
        embeddings[layer] = output.data.cpu().numpy()

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default="pixel", type=str,
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--probe", default=None, type=str,
                        choices=["visual", "linguistic"])
    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Path to probing data directory")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="which max length to use")
    parser.add_argument("--auth", default=None, type=str,
                        help="hf authentication token")
    args = parser.parse_args()

    model_dict = {"pixel": "Team-PIXEL/pixel-base",
                  "vit-mae": "facebook/vit-mae-base",
                  "pixel-words": "Team-PIXEL/pixel-small-words",
                  "pixel-bigrams": "Team-PIXEL/pixel-base-bigrams",
                  "pixel-ud": "Team-PIXEL/pixel-base-finetuned-parsing-ud-english-ewt",
                  "pixel-small": "Team-PIXEL/pixel-small-continuous",
                  "pixel-small-bigrams": "Team-PIXEL/pixel-small-bigrams",
                  "pixel-sst2": "Team-PIXEL/pixel-base-finetuned-sst2",
                  "pixel-mnli": "Team-PIXEL/pixel-base-finetuned-mnli",
                  "pixel-bigrams-mnli": "examples/pixel-bigrams-on-mnli",
                  "pixel-bigrams-ud": "UD_English-EWT-dep-pixel-base-bigrams-196-64-1-5e-5-15000-42",
                  "pixel-words-ud": "UD_English-EWT-dep-pixel-small-words-196-64-1-5e-5-15000-42",
                  "vit-mae-ud": "UD_English-EWT-dep-vit-mae-base-196-64-1-5e-5-15000-42",
                  "bert": "bert-base-cased"}

    if "bert" in args.model_name:
        config = AutoConfig.from_pretrained(model_dict[args.model_name])
        config.output_hidden_states = True
        config.output_attentions = True
        tokenizer = AutoTokenizer.from_pretrained(model_dict[args.model_name])
        model = AutoModel.from_pretrained(model_dict[args.model_name], config=config).cuda()
        model.eval()

    else:

        access_token = args.auth
        if args.model_name == "pixel-words-ud":
            args.max_seq_length = 196

        if args.model_name in ["pixel-bigrams", "pixel-bigrams-ud", "pixel-bigrams-mnli"]:
            renderer_cls = PangoCairoBigramsRenderer
        else:
            renderer_cls = PangoCairoTextRenderer

        if args.model_name in ["vit-mae", "vit-mae-ud"]:
            tokenizer = renderer_cls.from_pretrained(
                model_dict["pixel"],
                rgb=False,
                max_seq_length=196,
                fallback_fonts_dir="fallback_fonts",
                use_auth_token=access_token
            )
        elif args.model_name in ["pixel-bigrams"]:
            tokenizer = renderer_cls.from_pretrained(
                "test_text_renderer_config.json",
                rgb=False,
                max_seq_length=args.max_seq_length,
                fallback_fonts_dir="fallback_fonts"
            )
        elif args.model_name in ["pixel-bigrams-ud", "pixel-bigrams-mnli"]:
            tokenizer = renderer_cls.from_pretrained(
                f"{model_dict[args.model_name]}/text_renderer_config.json",
                rgb=False,
                max_seq_length=256,
                fallback_fonts_dir="fallback_fonts"
            )
        else:
            tokenizer = renderer_cls.from_pretrained(
                model_dict[args.model_name],
                rgb=False,
                max_seq_length=args.max_seq_length,
                fallback_fonts_dir="fallback_fonts",
                use_auth_token=access_token
            )

        config = PIXELConfig.from_pretrained(model_dict[args.model_name], use_auth_token=access_token)
        config.output_hidden_states = True
        config.output_attentions = True
        model = ViTModel.from_pretrained(model_dict[args.model_name], config=config, use_auth_token=access_token).cuda()
        # model = ViTModel.from_pretrained(model_dict[args.model_name], config=config)
        if "pixel" in args.model_name:
            resize_model_embeddings(model, args.max_seq_length)
        model.eval()

    tasks = [
        [
            'sentence_length',
            'word_content',
            'tree_depth',
            'top_constituents',
            'bigram_shift',
            'past_present',
            'subj_number',
            'obj_number',
            'odd_man_out',
            'coordination_inversion'
        ],
        ["count_character_sentences_binned-4", "count_character_words_binned-4",
         # "max_count_words_binned-4", "max_count_sentences_binned-4",
         # "argmax_count_sentences_binned-5_limited-8000", "argmax_count_sentences_binned-5",
         # "argmax_count_words_binned-5_limited-9900", "argmax_count_words_binned-5"
         ]
    ]

    if args.probe == "linguistic":
        task_index = 0
    elif args.probe == "visual":
        task_index = 1

    params = {'task_dir': args.data_dir, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'tokenizer': tokenizer, 'model': model,
              'model_name': args.model_name, 'seed': args.seed}

    for task in tasks[task_index]:
        probe = Probing(params=params, batcher=batcher, task=task)
        results = probe.run(params=params, batcher=batcher, task=task)

        # with open(f"{args.model_name}_{task}.pickle", 'wb') as handle:
        #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)






