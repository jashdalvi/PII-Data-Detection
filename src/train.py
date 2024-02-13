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
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


# declare the two GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" if torch.cuda.is_available() else "-1"

# avoids some issues when using more than one worker
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        wandb_api_key = os.environ['WANDB_API_KEY']
        wandb.login(key = wandb_api_key) # Enter your API key here
    except:
        print('Setting wandb usage to False')
        print('Add your wandb key in secrets so that you can use it')
        cfg.use_wandb = False

    def seed_everything(seed=cfg.seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed=cfg.seed)

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


    def generate_pred_df(preds, ds):
        id2label = {i: label for i, label in enumerate(LABELS)}
        triplets = set()
        document, token, label, token_str = [], [], [], []
        for p, token_map, offsets, tokens, doc in zip(preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):

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
                    triplet = (label_pred, token_id, tokens[token_id])

                    if triplet not in triplets:
                        document.append(doc)
                        token.append(token_id)
                        label.append(label_pred)
                        token_str.append(tokens[token_id])
                        triplets.add(triplet)


        df = pd.DataFrame({
            "document": document,
            "token": token,
            "label": label,
            "token_str": token_str
        })
        df = df.drop_duplicates().reset_index(drop=True)

        df["row_id"] = list(range(len(df)))

        return df[["row_id", "document", "token", "label"]].copy()
    

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


        tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation = True, max_length=max_length, return_overflowing_tokens=True, stride = 0)
        
        labels = np.array(labels)
        
        text = "".join(text)
        token_labels = []
        num_sequences = len(tokenized["input_ids"])
        for sequence_idx in range(num_sequences):
            offset_mapping_sequence = tokenized["offset_mapping"][sequence_idx]
            token_labels_sequence = []
            for start_idx, end_idx in offset_mapping_sequence:
                
                # CLS token
                if start_idx == 0 and end_idx == 0: 
                    token_labels_sequence.append(label2id["O"])
                    continue
                
                # case when token starts with whitespace
                if text[start_idx].isspace():
                    start_idx += 1
                
                while start_idx >= len(labels):
                    start_idx -= 1
                    
                token_labels_sequence.append(label2id[labels[start_idx]])
            
            token_labels.append(token_labels_sequence)
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
            "tokens": tokens
        }

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.model_name = cfg.model_name
            config = AutoConfig.from_pretrained(self.model_name)

            config.update({
                    "output_hidden_states": True,
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
                "weight_decay": 0.001,
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
    
    def train(epoch, model, train_loader, optimizer, scheduler, device, scaler):
        """training pass"""
        model.train()
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(train_loader), total = len(train_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            labels = batch.pop("labels")
            
            with autocast():
                outputs = model(**batch)
                loss = criterion(outputs, labels)
            
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            
            losses.update(loss.item() * cfg.gradient_accumulation_steps , cfg.batch_size)
            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if cfg.use_wandb:
                wandb.log({
                    "train/loss": losses.val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": epoch * len(train_loader) + batch_idx,

                })
        
        return losses.avg

    def postprocess_labels(outputs, threshold = 0.9):
        predictions = outputs.cpu().numpy()
        pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis = 2).reshape(predictions.shape[0],predictions.shape[1],1)
        id2label = {i: label for i, label in enumerate(LABELS)}
        preds = predictions.argmax(-1)
        preds_without_O = pred_softmax[:,:,:12].argmax(-1)
        O_preds = pred_softmax[:,:,12]

        preds_final = np.where(O_preds < threshold, preds_without_O , preds)
        return preds_final


    @torch.inference_mode()
    def evaluate(epoch, model, valid_loader, device):
        """evaluate pass"""
        model.eval()
        all_preds = []
        losses = AverageMeter()

        for batch_idx, (batch) in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            labels = batch.pop("labels")
            
            outputs = model(**batch)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), cfg.batch_size)
            all_preds.extend(postprocess_labels(outputs, threshold=cfg.threshold).tolist())

        return all_preds, losses.avg

    @torch.inference_mode()
    def predict(model, test_loader, device):
        """predict pass for calculating oof values"""
        model.eval()
        all_outputs = []

        for batch_idx, (batch) in tqdm(enumerate(test_loader), total = len(test_loader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            outputs = model(**batch)
            all_outputs.extend(outputs.cpu().numpy())
            
        all_outputs = np.vstack(all_outputs)
        return all_outputs
    
    def main_fold(fold):
        """Main loop"""
        best_score = 0
        # Seed everything
        seed_everything(seed=cfg.seed)
        if cfg.use_wandb:
            run = wandb.init(project=cfg.project_name, 
                            config=dict(cfg), 
                            group = cfg.model_name, 
                            reinit=True)
            wandb.define_metric("train/step")
            wandb.define_metric("valid/step")
            # define which metrics will be plotted against it
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("valid/*", step_metric="valid/step")

        with open("../data/train.json") as f:
            data = json.load(f)

        

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


        valid_reference_df = generate_gt_df(ds.filter(lambda x: x["fold"] == fold, num_proc=4))

        label2id = {label: i for i, label in enumerate(LABELS)}
        id2label = {i: label for i, label in enumerate(LABELS)}

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))


        ds = ds.map(
            tokenize, 
            fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg.max_length}, 
            num_proc=4,
        ).remove_columns(["full_text", "trailing_whitespace", "provided_labels"])

        ds = build_flatten_ds(ds)

        train_ds = ds.filter(lambda x: x["fold"] != fold, num_proc=4)
        valid_ds = ds.filter(lambda x: x["fold"] == fold, num_proc=4)
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
        # Do not resize the token embeddings
        model.transformer.resize_token_embeddings(len(tokenizer))
        model.to(cfg.device)

        if cfg.multi_gpu:
            print("Using Multi-GPU Setup")
            model = nn.DataParallel(model)
        
        num_train_steps = int(len(train_ds) / cfg.batch_size / cfg.gradient_accumulation_steps * cfg.epochs)

        if cfg.multi_gpu:
            optimizer, scheduler = get_optimizer_scheduler(model.module, num_train_steps)
        else:
            optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)

        scaler = GradScaler()

        for epoch in range(cfg.epochs):
            print(f"Training FOLD : {fold}, EPOCH: {epoch + 1}")

            train_loss = train(epoch, model, train_loader, optimizer, scheduler, cfg.device, scaler)
            preds, valid_loss = evaluate(epoch, model, valid_loader, cfg.device)
            pred_df = generate_pred_df(preds, valid_ds)
            eval_dict = compute_metrics(pred_df, valid_reference_df)
            print(f"\nValidation f5 score: {eval_dict['ents_f5']}")
            if cfg.use_wandb:
                wandb.log({"valid/train_loss_avg": train_loss, 
                        "valid/valid_loss_avg": valid_loss, 
                        "valid/f5": eval_dict['ents_f5'],
                        "valid/step": epoch})

            if eval_dict['ents_f5'] > best_score:
                best_score = eval_dict['ents_f5']
                if cfg.multi_gpu:
                    torch.save(model.module.state_dict(), os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_fold{fold}_seed{cfg.seed}.bin"))
                else:
                    torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_fold{fold}_seed{cfg.seed}.bin"))
        
        
        if cfg.use_wandb:
            run.finish()
        
        torch.cuda.empty_cache()

        return best_score
    
    def train_whole_dataset():
        """Main loop"""
        # Seed everything
        seed_everything(seed=cfg.seed)
        if cfg.use_wandb:
            run = wandb.init(project=cfg.project_name, 
                            config=dict(cfg), 
                            group = cfg.model_name, 
                            reinit=True)
            wandb.define_metric("train/step")
            wandb.define_metric("valid/step")
            # define which metrics will be plotted against it
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("valid/*", step_metric="valid/step")

        with open("../data/train.json") as f:
            data = json.load(f)

        ds = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data],
            "document": [x["document"] for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
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
                }
            )

            ## Check that the features are the same and then concatenate
            assert ds.features.type == external_ds.features.type
            ds = concatenate_datasets([ds, external_ds])

        label2id = {label: i for i, label in enumerate(LABELS)}
        id2label = {i: label for i, label in enumerate(LABELS)}

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # lots of newlines in the text
        # adding this should be helpful
        tokenizer.add_tokens(AddedToken("\n", normalized=False))


        ds = ds.map(
            tokenize, 
            fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg.max_length}, 
            num_proc=4,
        ).remove_columns(["full_text", "trailing_whitespace", "provided_labels"])

        ds = build_flatten_ds(ds)
        collator = Collate(tokenizer)

        train_loader = torch.utils.data.DataLoader(
            ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            collate_fn=collator
        )

        model = Model()
        # Do not resize the token embeddings
        model.transformer.resize_token_embeddings(len(tokenizer))
        model.to(cfg.device)

        if cfg.multi_gpu:
            print("Using Multi-GPU Setup")
            model = nn.DataParallel(model)
        
        num_train_steps = int(len(ds) / cfg.batch_size / cfg.gradient_accumulation_steps * cfg.epochs)

        if cfg.multi_gpu:
            optimizer, scheduler = get_optimizer_scheduler(model.module, num_train_steps)
        else:
            optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)

        scaler = GradScaler()

        for epoch in range(cfg.epochs):
            print(f"Training FULL FOLD: EPOCH: {epoch + 1}")

            train_loss = train(epoch, model, train_loader, optimizer, scheduler, cfg.device, scaler)
            if cfg.use_wandb:
                wandb.log({"valid/train_loss_avg": train_loss, 
                        "valid/step": epoch})
            
            if cfg.multi_gpu:
                torch.save(model.module.state_dict(), os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_full_fold_seed{cfg.seed}.bin"))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"{cfg.model_name.split(os.path.sep)[-1]}_full_fold_seed{cfg.seed}.bin"))
        
        
        if cfg.use_wandb:
            run.finish()
        
        torch.cuda.empty_cache()

    # fold_scores = []
    # for fold in range(4):
    #     fold_score = main_fold(fold)
    #     fold_scores.append(fold_score)
    
    if cfg.train_whole_dataset:
        for seed in [41,42]:
            cfg.seed = seed
            train_whole_dataset()


    # cv = np.mean(fold_scores)
    cv = 0.9488
    print(f"CV SCORE: {cv:.4f}")

    login(os.environ.get("HF_HUB_TOKEN"))

    api = HfApi()
    cfg.repo_id = f"jashdalvi/pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}"
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
        "title": f"pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}-full",
        "id": f"jashdalvi99/pii-data-detection-{cfg.model_name.split(os.path.sep)[-1]}-cv-{cv:.4f}-full".replace(".", ""),
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