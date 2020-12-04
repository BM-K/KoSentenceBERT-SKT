from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel, BertConfig
import json
from typing import List, Dict, Optional
import os
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch

class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param tokenizer_args: Dict with parameters which are passed to the tokenizer.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, isKor=False, isLoad=False):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length
        
        # for Korea BERT
        if isKor:
            bert_model, vocab = get_pytorch_kobert_model()
            tokenizer = get_tokenizer()
            bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
                                    
            self.auto_model = bert_model
            self.tokenizer = bert_tokenizer
            self.vocab = vocab

            if isLoad:
                print("Load Model")
                self.auto_model.load_state_dict(torch.load(model_name_or_path+'/result.pt'))
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir,
                                                       **tokenizer_args)


    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.auto_model(**features)
        output_tokens = output_states[0]
        
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: str):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        tokens = self.tokenizer(text)
        tokens = [self.vocab.token_to_idx[token] for token in tokens]
        return tokens

    def get_segment_ids_vaild_len(self, inputs):
        v_len_list = [0] * len(inputs)
        segment_ids = torch.zeros_like(inputs).long()
        valid_length = torch.tensor(v_len_list, dtype=torch.int32)
        return segment_ids, valid_length

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length) : attention_mask[i][:v] = 1
        return attention_mask

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 #Add space for special tokens
        
        cls_token = self.vocab.cls_token
        sep_token = self.vocab.sep_token
        sep_token_idx = self.vocab.token_to_idx[sep_token]
        cls_token_idx = self.vocab.token_to_idx[cls_token]
        
        #input_sentence = [cls_token_idx] + tokens + [sep_token_idx]
        tokens = torch.cat([torch.tensor([cls_token_idx]), torch.tensor(tokens)], dim=-1)
        tokens = torch.cat([tokens, torch.tensor([sep_token_idx])], dim=-1)
        
        segment_ids, valid_len = self.get_segment_ids_vaild_len(tokens)
        attention_mask = self.gen_attention_mask(tokens, valid_len)
        
        result = {'input_ids':tokens.unsqueeze(0), 'token_type_ids':segment_ids.unsqueeze(0), 'attention_mask':attention_mask.unsqueeze(0)}
        
        return result
        #return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation=True, prepend_batch_axis=True)


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        torch.save(self.auto_model.state_dict(), os.path.join(output_path+'/result.pt'))
        #self.auto_model.save_pretrained(output_path)
        #self.tokenizer.save_pretrained(output_path)

        #with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
        #    json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
        #    config = json.load(fIn)
        
        return Transformer(model_name_or_path=input_path, isKor=True, isLoad=True)






