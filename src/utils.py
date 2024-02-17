import numpy as np
import torch
import torch.nn as nn
import re
from collections import defaultdict
from typing import Dict
import numpy as np
import pandas as pd
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from spacy.tokens import Span
from spacy import displacy
import wandb


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    
    references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1] # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:] # avoid B- and I- prefix
            
        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:] # avoid B- and I- prefix
        
        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()
    
    for prf in score_per_type.values():
        totals += prf

    return {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k!= 'O'},
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Mapping of named colors to their RGB equivalents
named_color_to_rgb = {
    "aqua": (0, 255, 255),
    "skyblue": (135, 206, 235),
    "limegreen": (50, 205, 50),
    "lime": (0, 255, 0),
    "hotpink": (255, 105, 180),
    "lightpink": (255, 182, 193),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "salmon": (250, 128, 114),
    "silver": (192, 192, 192),
    "lightgray": (211, 211, 211),
    "brown": (165, 42, 42),
    "chocolate": (210, 105, 30),
    # Add the rest of your named colors and their RGB values here
}

def get_rgba(color_name, opacity):
    """Convert a named color and opacity to an rgba string."""
    rgb = named_color_to_rgb[color_name]  # Get the RGB values for the named color
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

def postprocess_labels(predictions, token_idxs_mapping = None, threshold = 0.9):
    if token_idxs_mapping is not None:
        # Initialize averaged_predictions with the same shape as predictions
        averaged_predictions = np.zeros_like(predictions)
        
        # Iterate over each unique token index to average predictions
        for token_idx in set(token_idxs_mapping):
            # Find the indices in token_idxs_mapping that match the current token_idx
            indices = [i for i, x in enumerate(token_idxs_mapping) if x == token_idx]
            
            # Average the predictions for these indices and assign to the correct positions
            averaged_predictions[indices] = np.mean(predictions[indices], axis=0)
        
        # Use the averaged predictions for further processing
        predictions = averaged_predictions
    preds = predictions.argmax(-1)
    preds_without_O = predictions[:,:12].argmax(-1)
    O_preds = predictions[:,12]

    preds_final = np.where(O_preds < threshold, preds_without_O , preds)
    return preds_final

def visualize(row, nlp, labels):
    options = {
        "colors": {
            "B-NAME_STUDENT": "aqua",
            "I-NAME_STUDENT": "skyblue",
            "B-EMAIL": "limegreen",
            "I-EMAIL": "lime",
            "B-USERNAME": "hotpink",
            "I-USERNAME": "lightpink",
            "B-ID_NUM": "purple",
            "I-ID_NUM": "rebeccapurple",
            "B-PHONE_NUM": "red",
            "I-PHONE_NUM": "salmon",
            "B-URL_PERSONAL": "silver",
            "I-URL_PERSONAL": "lightgray",
            "B-STREET_ADDRESS": "brown",
            "I-STREET_ADDRESS": "chocolate",
        }
    }
    doc = nlp(row.full_text)
    doc.ents = [
        Span(doc, token_idx, token_idx + 1, label=label)
        for token_idx, label in labels
    ]
    html = displacy.render(doc, style="ent", jupyter=False, options=options)
    return html

def convert_for_upload(viz_df):
    for col in viz_df.columns:
        if "viz" not in col:
            viz_df[col] = viz_df[col].astype(str)
    return viz_df

def get_error_row_ids(valid_df, pred_df):
    doc_to_label_map_valid = dict()
    for i, row in valid_df.iterrows():
        if row.document not in doc_to_label_map_valid:
            doc_to_label_map_valid[row.document] = []
        doc_to_label_map_valid[row.document].append((int(row.token), row.label))
    
    doc_to_label_map_pred= dict()
    for i, row in pred_df.iterrows():
        if row.document not in doc_to_label_map_pred:
            doc_to_label_map_pred[row.document] = []
        doc_to_label_map_pred[row.document].append((int(row.token), row.label))

    error_row_ids = []
    for doc in doc_to_label_map_valid.keys():
        valid_labels = sorted(doc_to_label_map_valid.get(doc, []), key = lambda x: x[0])
        pred_labels = sorted(doc_to_label_map_pred.get(doc, []), key = lambda x: x[0])
        if valid_labels != pred_labels:
            error_row_ids.append(doc)

    return error_row_ids

def get_unique_pairs(pairs):
    unique_tokens = []
    unique_pairs = []
    for pair in pairs:
        if pair[0] not in unique_tokens:
            unique_pairs.append(pair)
            unique_tokens.append(pair[0])
    return unique_pairs

def generate_visualization_df(viz_df, valid_df, pred_df, nlp):
    doc_to_label_map_valid = dict()
    for i, row in valid_df.iterrows():
        if row.document not in doc_to_label_map_valid:
            doc_to_label_map_valid[row.document] = []
        doc_to_label_map_valid[row.document].append((int(row.token), row.label))
    
    doc_to_label_map_pred= dict()
    for i, row in pred_df.iterrows():
        if row.document not in doc_to_label_map_pred:
            doc_to_label_map_pred[row.document] = []
        doc_to_label_map_pred[row.document].append((int(row.token), row.label))

    for i, row in tqdm(viz_df.iterrows(), total = len(viz_df)):
        valid_labels = get_unique_pairs(sorted(doc_to_label_map_valid.get(row.document, []), key = lambda x: x[0]))
        pred_labels = get_unique_pairs(sorted(doc_to_label_map_pred.get(row.document, []), key = lambda x: x[0]))
        gt_html = wandb.Html(visualize(row, nlp, valid_labels))
        pred_html = wandb.Html(visualize(row, nlp, pred_labels))
        viz_df.at[i, 'gt_viz'] = gt_html
        viz_df.at[i, 'pred_viz'] = pred_html

    viz_df.fillna("", inplace=True)
    viz_df = convert_for_upload(viz_df)
    return viz_df

