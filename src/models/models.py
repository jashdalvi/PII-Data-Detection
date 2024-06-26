from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.llama.modeling_llama import (LlamaModel,
                                                      LlamaPreTrainedModel)
from transformers.models.mistral.modeling_mistral import (
    MistralModel, MistralPreTrainedModel)
from transformers.models.phi.modeling_phi import PhiModel, PhiPreTrainedModel


class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out
    
class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.lstm_head = nn.LSTM(in_features,
                            in_features//2,
                            1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.classification_head = nn.Linear(in_features, self.num_labels, bias=False)

    def forward(self, x):
        x, (_, _) = self.lstm_head(x) # Apply LSTM for bidirectional context
        logits = self.classification_head(x) # (bs, num_labels)
        return logits


class MistralForTokenClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple,  TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = transformer_outputs[0]  # (bs, seq_len, dim)

        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=transformer_outputs.attentions,
        )
    

class PhiForTokenClassification(PhiPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiModel(config)

        self.lstm_head = nn.LSTM(config.hidden_size,
                            config.hidden_size//2,
                            1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.classification_head = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple,  TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = transformer_outputs[0]  # (bs, seq_len, dim)

        sequence_output, (_, _) = self.lstm_head(sequence_output) # Apply LSTM for bidirectional context
        logits = self.classification_head(sequence_output) # (bs, num_labels)

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=transformer_outputs.attentions,
        )
    
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)

        # self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple,  TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = transformer_outputs[0]  # (bs, seq_len, dim)
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=transformer_outputs.attentions,
        )