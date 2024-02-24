import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import numpy as np
import os
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, DataCollatorForTokenClassification
from tokenizers import AddedToken
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import random
from utils import AverageMeter, compute_metrics
import json
from datasets import Dataset, concatenate_datasets
import warnings
from huggingface_hub import HfApi, login, create_repo
import subprocess
import shutil
from utils import postprocess_labels, get_error_row_ids, generate_visualization_df
import spacy
from collections import OrderedDict
from accelerate import Accelerator
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


# Initialize hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()

#Add labels as global variable
LABELS = ['B-EMAIL',
        'B-ID_NUM',
        'B-NAME_STUDENT',
        'B-PHONE_NUM',
        'B-STREET_ADDRESS',
        'B-URL_PERSONAL',
        'B-USERNAME',
        'I-ID_NUM',
        'I-NAME_STUDENT',
        'I-PHONE_NUM',
        'I-STREET_ADDRESS',
        'I-URL_PERSONAL',
        'O']

@hydra.main(config_path="config", config_name="config_accelerate")
def main(cfg: DictConfig):

    def seed_everything(seed=cfg.seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def generate_gt_df(ds):
        reference_data = []
        for doc, tokens, labels in zip(ds["document"], ds["tokens"], ds["provided_labels"]):
            reference_data.append({"document": doc, "tokens": tokens, "labels": labels})

        df = pd.DataFrame(reference_data)[['document', 'tokens', 'labels']].copy()
        df = df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
        df['token'] = df.groupby('document').cumcount()

        label_list = df['label'].unique().tolist()

        reference_df = df[df['label'] != 'O'].copy()
        reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
        reference_df = reference_df[['row_id', 'document', 'token', 'label']].copy()

        return reference_df
    
    def build_flatten_ds(ds):
        features = list(ds.features.keys())
        dataset_dict = {feature: [] for feature in features}

        for example in tqdm(ds, total=len(ds)):
            #Also make sure everything is a list
            for feature in features:
                assert isinstance(example[feature], list), f"Feature {feature} is not a list"
            for feature in features:
                dataset_dict[feature].extend(example[feature])

        return Dataset.from_dict(dataset_dict)
    
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis = -1, keepdims = True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def get_merged_preds(predictions, token_idxs_mapping):
        if token_idxs_mapping is not None:
            # Initialize averaged_predictions with the same shape as predictions
            averaged_predictions = np.zeros_like(predictions)
            
            # Iterate over each unique token index to average predictions
            for token_idx in set(token_idxs_mapping).difference(set([-1])):
                # Find the indices in token_idxs_mapping that match the current token_idx
                indices = np.where(np.array(token_idxs_mapping) == token_idx)[0]
                
                # Average the predictions for these indices and assign to the correct positions
                averaged_predictions[indices] = np.mean(predictions[indices], axis=0)
            
            # Use the averaged predictions for further processing
            predictions = averaged_predictions
        
        return predictions
    
    def get_processed_preds(preds, ds):
        if cfg.stride > 0:
            # Average the predictions of overlapping tokens
            groupby_preds = OrderedDict()
            for p, doc, offsets in zip(preds, ds["document"], ds["offset_mapping"]):
                if doc not in groupby_preds:
                    groupby_preds[doc] = []
                groupby_preds[doc].append(softmax(p[:len(offsets)]))
            merged_preds = []
            for doc in groupby_preds:
                doc_preds = groupby_preds[doc]
                if len(doc_preds) > 1:
                    for i in range(len(doc_preds) - 1):
                        total_stride = cfg.stride + 1  # Add extra token for special tokens
                        # Calculate the mean for the overlapping predictions
                        mean_preds = (np.array(doc_preds[i][-total_stride:]) + np.array(doc_preds[i + 1][:total_stride])) / 2
                        # Update the current and next predictions with the averaged values
                        doc_preds[i][-total_stride:] = mean_preds.tolist()
                        doc_preds[i + 1][:total_stride] = mean_preds.tolist()
                    # After processing all overlaps, append all modified predictions
                    merged_preds.extend(doc_preds)
                else:
                    # If there's only one set of predictions, append it directly
                    merged_preds.extend(doc_preds)
            
            preds = merged_preds.copy()
        else:
            preds = [softmax(p) for p in preds]

        preds = [get_merged_preds(p, token_idxs_mapping if cfg.merge_token_preds else None) for p, token_idxs_mapping in zip(preds, ds["token_idxs_mapping"])]

        return preds

    def generate_pred_df(preds, ds, threshold=0.9):
        id2label = {i: label for i, label in enumerate(LABELS)}
        triplets = set()
        row, document, token, label, token_str = [], [], [], [], []
        softmax_preds = preds.copy()
        preds = [postprocess_labels(p, None, threshold=threshold) for p, token_idxs_mapping in zip(preds, ds["token_idxs_mapping"])]

        for i, (p, token_map, offsets, tokens, doc) in enumerate(zip(preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"])):

            for token_pred, (start_idx, end_idx) in zip(p, offsets):
                label_pred = id2label[int(token_pred)]

                if start_idx + end_idx == 0: continue

                if token_map[start_idx] == -1: 
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map): break

                token_id = token_map[start_idx]

                # ignore "O" predictions and whitespace preds
                if label_pred != "O" and token_id != -1:
                    triplet = (doc, token_id)

                    if triplet not in triplets:
                        row.append(i)
                        document.append(doc)
                        token.append(token_id)
                        label.append(label_pred)
                        token_str.append(tokens[token_id])
                        triplets.add(triplet)


        df = pd.DataFrame({
            "eval_row": row,
            "document": document,
            "token": token,
            "label": label,
            "token_str": token_str
        })
        df = df.drop_duplicates().reset_index(drop=True)

        df["row_id"] = list(range(len(df)))

        return df, softmax_preds
    

    class Collate:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, batch):
            batch_len = max([len(sample["input_ids"]) for sample in batch])
            
            output = dict()
            output["input_ids"] = [sample["input_ids"] for sample in batch]
            output["attention_mask"] = [sample["attention_mask"] for sample in batch]
            output["labels"] = [sample["labels"] for sample in batch]


            output["input_ids"] = torch.tensor([input_ids + [self.tokenizer.pad_token_id] * (batch_len - len(input_ids)) for input_ids in output["input_ids"]])
            output["attention_mask"] = torch.tensor([attention_mask + [0] * (batch_len - len(attention_mask)) for attention_mask in output["attention_mask"]])
            output["labels"] = torch.tensor([labels + [-100] * (batch_len - len(labels)) for labels in output["labels"]])

            output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
            output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
            output["labels"] = torch.tensor(output["labels"], dtype=torch.long)

            return output


    def tokenize(example, tokenizer, label2id, max_length):
        """tokenize the examples"""
        text = []
        labels = []
        token_map = [] # Character index to spacy token mapping

        token_map_idx = 0
        for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
            text.append(t)
            labels.extend([l]*len(t))
            token_map.extend([token_map_idx] * len(t))
            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

            token_map_idx += 1


        tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation = True, max_length=max_length, return_overflowing_tokens=True, stride = cfg.stride)
        
        labels = np.array(labels)
        
        text = "".join(text)
        token_labels = []
        token_idxs_mapping = [] # Represents the mapping of deberta token idx to spacy token idx. We can potentially merge the predictions of these tokens
        num_sequences = len(tokenized["input_ids"])
        for sequence_idx in range(num_sequences):
            offset_mapping_sequence = tokenized["offset_mapping"][sequence_idx]
            token_labels_sequence = []
            token_idxs_mapping_sequence = []
            for start_idx, end_idx in offset_mapping_sequence:
                
                # CLS token
                if start_idx == 0 and end_idx == 0: 
                    token_idxs_mapping_sequence.append(-1)
                    token_labels_sequence.append(label2id["O"])
                    continue
                
                # case when token starts with whitespace
                if text[start_idx].isspace():
                    start_idx += 1
                
                while start_idx >= len(labels):
                    start_idx -= 1
                    
                token_labels_sequence.append(label2id[labels[start_idx]])
                token_idxs_mapping_sequence.append(token_map[start_idx])
            
            token_labels.append(token_labels_sequence)
            token_idxs_mapping.append(token_idxs_mapping_sequence)
        #preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]
        token_map = [token_map for _ in range(num_sequences)]
        document = [example["document"] for _ in range(num_sequences)]
        fold = [example["fold"] for _ in range(num_sequences)]
        tokens = [example["tokens"] for _ in range(num_sequences)]
            
        return {
            **tokenized,
            "labels": token_labels,
            "token_map": token_map,
            "document": document,
            "fold": fold,
            "tokens": tokens,
            "token_idxs_mapping": token_idxs_mapping
        }

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.model_name = cfg.model_name
            config = AutoConfig.from_pretrained(self.model_name)

            config.update({
                    "hidden_dropout_prob": cfg.hidden_dropout_prob,
                    "attention_probs_dropout_prob" : cfg.hidden_dropout_prob,
                    "layer_norm_eps": cfg.layer_norm_eps,
                    "add_pooling_layer": False,
                    "num_labels": len(LABELS)
            })

            self.transformer = AutoModel.from_pretrained(self.model_name, config=config)
            self.linear = nn.Linear(config.hidden_size, len(LABELS))

            if cfg.gradient_checkpointing_enable:
                self.transformer.gradient_checkpointing_enable()

            if cfg.freeze:
                self.freeze()
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
        def freeze(self, start_freeze = 0, end_freeze = 6):
            for i in range(start_freeze, end_freeze):
                for n,p in self.transformer.encoder.layer[i].named_parameters():
                    p.requires_grad = False

        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.linear(outputs.last_hidden_state)
            return logits

    def criterion(outputs, targets):
        return nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    
    def get_optimizer_scheduler(model, num_train_steps):
        """get optimizer and scheduler"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
                "lr" : cfg.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr" : cfg.lr
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
                num_training_steps=num_train_steps,
                last_epoch=-1,
        )
        return optimizer, scheduler
    
    def train(accelerator, epoch, model, train_loader, optimizer, scheduler):
        """training pass"""
        model.train()
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(train_loader), total = len(train_loader), disable=not accelerator.is_main_process):   
            labels = batch.pop("labels")
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                with accelerator.autocast():
                    loss = criterion(outputs, labels)
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            losses.update(loss.item(), cfg.batch_size)

            
            accelerator.log({
                "train/loss": losses.val,
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": epoch * len(train_loader) + batch_idx,

            })
        
        return losses.avg

    def evaluate(accelerator, epoch, model, valid_loader):
        """evaluate pass"""
        model.eval()
        all_preds = []
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(valid_loader), total = len(valid_loader), disable=not accelerator.is_main_process):
            labels = batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
            
            with accelerator.autocast():
                loss = criterion(outputs, labels)

            losses.update(loss.item(), cfg.valid_batch_size)

            gathered_outputs = accelerator.gather_for_metrics(outputs)
            all_preds.extend([np.squeeze(output, 0) for output in np.split(gathered_outputs.detach().cpu().numpy(), len(gathered_outputs))])

        return all_preds, losses.avg
    
    def prepare_data(accelerator):
        with open("../data/train.json") as f:
            data = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        if cfg.add_new_line_token:
            tokenizer.add_tokens(AddedToken("\n", normalized=False))


        with accelerator.main_process_first():
            ds = Dataset.from_dict({
                "full_text": [x["full_text"] for x in data],
                "document": [x["document"] for x in data],
                "tokens": [x["tokens"] for x in data],
                "trailing_whitespace": [x["trailing_whitespace"] for x in data],
                "provided_labels": [x["labels"] for x in data],
                "fold": [x["document"] % 4 for x in data]
            })

            if cfg.use_external_data:
                with open("../data/mixtral-8x7b-v1.json") as f:
                    external_data = json.load(f)

                external_ds = Dataset.from_dict(
                    {
                        "full_text": [x["full_text"] for x in external_data],
                        "document": [doc for doc, x in enumerate(external_data, 30000)],
                        "tokens": [x["tokens"] for x in external_data],
                        "trailing_whitespace": [x["trailing_whitespace"] for x in external_data],
                        "provided_labels": [x["labels"] for x in external_data],
                        "fold": [fold + 1 for _ in external_data]
                    }
                )

                ## Check that the features are the same and then concatenate
                assert ds.features.type == external_ds.features.type
                ds = concatenate_datasets([ds, external_ds])

            original_ds = ds.copy()

            label2id = {label: i for i, label in enumerate(LABELS)}

            ds = ds.map(
                tokenize, 
                fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg.max_length},
                num_proc=4
            ).remove_columns(["full_text", "trailing_whitespace", "provided_labels"])

            ds = build_flatten_ds(ds)
        return ds, original_ds, tokenizer
    
    def prepare_fold_data(accelerator, ds, original_ds, fold):
        with accelerator.main_process_first():
            valid_reference_df = generate_gt_df(original_ds.filter(lambda x: x["fold"] == fold, num_proc=4))
            train_ds = ds.filter(lambda x: x["fold"] != fold, num_proc=4)
            valid_ds = ds.filter(lambda x: x["fold"] == fold, num_proc=4)

        return train_ds, valid_ds, valid_reference_df
    
    def main_fold(accelerator, fold, train_ds, valid_ds, tokenizer, valid_reference_df):
        """Main loop"""
        best_score = 0
        best_pred_df = None
        thresholds_to_validate = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
        threshold_to_idx_mapping = {threshold: idx for idx, threshold in enumerate(thresholds_to_validate)}
        accelerator.print(threshold_to_idx_mapping)
        # Seed everything
        seed_everything(seed=cfg.seed)
        accelerator.init_trackers(project_name = cfg.project_name, config = dict(cfg), init_kwargs = {"wandb": {"group": cfg.model_name, "reinit": True}})

        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        if accelerator.is_main_process:
            wandb_tracker.define_metric("train/step")
            wandb_tracker.define_metric("valid/step")
            # define which metrics will be plotted against it
            wandb_tracker.define_metric("train/*", step_metric="train/step")
            wandb_tracker.define_metric("valid/*", step_metric="valid/step")

        collator = Collate(tokenizer)

        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            collate_fn=collator
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=cfg.valid_batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            collate_fn=collator
        )

        model = Model()
        if cfg.add_new_line_token:
            model.transformer.resize_token_embeddings(len(tokenizer))
        
        num_train_steps = (len(train_loader) * cfg.epochs) // cfg.gradient_accumulation_steps
        optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)

        model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, valid_loader, scheduler
        )

        for epoch in range(cfg.epochs):
            accelerator.print(f"Training FOLD : {fold}, EPOCH: {epoch + 1}")
            # Training loop
            train_loss = train(accelerator, epoch, model, train_loader, optimizer, scheduler)
            # Validation loop
            preds, valid_loss = evaluate(accelerator, epoch, model, valid_loader)
            threshold_f5 = []
            accelerator.print("Postprocessing predictions for validation...")
            # Processing predictions for validation
            preds = get_processed_preds(preds, valid_ds)

            accelerator.print("Validating for various thresholds...")
            for threshold in thresholds_to_validate:
                pred_df, softmax_preds = generate_pred_df(preds, valid_ds, threshold=threshold)
                eval_dict = compute_metrics(pred_df, valid_reference_df)
                threshold_f5.append(eval_dict['ents_f5'])
                accelerator.print(f"Threshold: {threshold}, Validation f5 score: {eval_dict['ents_f5']}")
            
            f5_score = threshold_f5[threshold_to_idx_mapping[float(cfg.threshold)]]

            accelerator.print(f"\nValidation f5 score: {f5_score:.4f}")

            accelerator.log({"valid/train_loss_avg": train_loss, 
                    "valid/valid_loss_avg": valid_loss, 
                    "valid/f5": f5_score,
                    **({f"valid/f5_{threshold}": f5 for threshold, f5 in zip(thresholds_to_validate, threshold_f5)}),
                    "valid/step": epoch})

            if f5_score > best_score:
                best_score = f5_score
                best_pred_df = pred_df

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_fold{fold}_seed{cfg.seed}.bin")

                if accelerator.is_main_process:
                    torch.save(unwrapped_model.state_dict(), save_path)
        # Prepare data to visualize errors and log them as a Weights & Biases table
        print('Visualizing errors...')
        if cfg.visualize:
            error_row_ids = get_error_row_ids(valid_reference_df, best_pred_df)
            viz_df = pd.read_json("../data/train.json")
            viz_df = viz_df[viz_df.document.isin(error_row_ids)][["document", "full_text"]]
            nlp = spacy.blank("en")
            viz_df = generate_visualization_df(viz_df, valid_reference_df, best_pred_df, nlp)
            errors_table = wandb.Table(dataframe=viz_df)
            if accelerator.is_main_process:
                accelerator.log({'errors_table': errors_table})
        
        accelerator.end_training()
        accelerator.free_memory()
        return best_score

    wandb.login(key = os.environ['WANDB_API_KEY']) # Enter your API key here
    # Create the main accelerator object
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=int(cfg.gradient_accumulation_steps), log_with = "wandb")

    # Create the output directory if it doesn't exist
    if accelerator.is_main_process:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir, exist_ok=True)

    fold_scores = []
    num_folds = min(max(cfg.num_folds, 0), 4)
    ds, original_ds, tokenizer = prepare_data(accelerator)
    for fold in range(num_folds):
        # Prepare the data for the fold for training and eval
        train_ds, valid_ds, valid_reference_df = prepare_fold_data(accelerator, ds, original_ds, fold)
        fold_score = main_fold(accelerator, fold, train_ds, valid_ds, tokenizer, valid_reference_df)
        fold_scores.append(fold_score)

    cv = np.mean(fold_scores)
    accelerator.print(f"CV SCORE: {cv:.4f}")

    if cfg.upload_models and accelerator.is_main_process:
        login(os.environ.get("HF_HUB_TOKEN"))
        api = HfApi()
        cfg.repo_id = f"jashdalvi/pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.5f}"
        # Creating a model repository in baseplate
        create_repo(cfg.repo_id, private= True, exist_ok=True)
        # Pushing the model to the hub
        api.upload_folder(
            folder_path = cfg.output_dir,
            path_in_repo = "/",
            repo_id = cfg.repo_id,
            repo_type = "model"
        )

        # Commenting out the kaggle api dataset upload code
        subprocess.run(["kaggle", "datasets", "init", "-p", cfg.output_dir], check=True)
        kaggle_dataset_metadata = {
            "title": f"pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.5f}",
            "id": f"jashdalvi99/pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.5f}".replace(".", ""),
            "licenses": [
                {
                "name": "CC0-1.0"
                }
            ]
        }
        # Overwriting the dataset metadata file
        with open(os.path.join(cfg.output_dir, "dataset-metadata.json"), "w") as f:
            json.dump(kaggle_dataset_metadata, f)
        # Uploading the dataset to kaggle
        subprocess.run(["kaggle", "datasets", "create", "-p", cfg.output_dir], check=True)

        # Deleting the output directory to save some space
        shutil.rmtree(cfg.output_dir)
        # Remove the local wandb dir to save some space
        shutil.rmtree("wandb")


if __name__ == "__main__":
    main()