from transformers import AutoConfig, RoFormerConfig, MT5Config, BartConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerSinusoidalPositionalEmbedding
from transformers.models.mt5.modeling_mt5 import MT5Model, MT5Model, MT5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartModel, BartDecoder ,BartForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
# from transformers import GPT3Model, GPT3Tokenizer
import time
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
    mean_flat,
)

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1, # default 线性头?
        self_conditions=0, ## add self_conditions
        mask=0, ## add mask controls
        use_en_de=0, ### if use encoder decoder
    ):
        super().__init__()

        if config is None:
            if config_name == 'bert-base-chinese':
                print("### 001 Net's Arch use bert-cn ")
                config = AutoConfig.from_pretrained('bert-base-chinese')
                config.hidden_dropout_prob = dropout
            elif config_name == 'bert-base-chinese-m':
                print("### 002 Net's Arch use bert-cn medium ")
                config = AutoConfig.from_pretrained("bert-base-chinese") # you should manually modify the config in bert-base-chinese, head=8, layer=8,hiddensize=512
                config.hidden_dropout_prob = dropout
            elif config_name == "bart":
                raise NotImplementedError("Net's Arch use BART  ")
                config = BartConfig.from_pretrained("bart-base-chinese")
            elif config_name == 't5-pegasus':
                raise NotImplementedError("Net's Arch use mt5  ")
                config = MT5Config.from_pretrained("imxly/t5-pegasus")
            else:
                raise NotImplementedError("Net's others")
                config = AutoConfig.from_pretrained(config_name)
                
            config.hidden_dropout_prob = dropout
        
        ### while use t5-pegasus, hidden_dim == 768
        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.self_conditions = self_conditions ### add self_conditions
        self.mask = mask ### add mask controls
        self.config_name = config_name
        self.use_en_de = use_en_de ### whether en-decoder
        
        print("### the input_dim is ", input_dims)
        print("### the h_dim is ", self.hidden_size)
        
        if self.use_en_de == 1:
            ##
            decoder_config = BartConfig.from_pretrained("bart-base-chinese")
            # print(decoder_config)
            tmp_bart = BartForConditionalGeneration(decoder_config) # embed, encoder, decoder, lm_head
            # print(tmp_bart)
            
            # with th.no_grad():
            self.decoder = tmp_bart.model.decoder # 只用decoder
                # self.decoder = tmp_bart.model.encoder
                # self.lm_head2 = tmp_bart.lm_head
                # self.decoder_word_embedding = tmp_bart.model.decoder.embed_tokens
                # print("### lm_head2 is ", self.lm_head2)
                # print("### decoder_word_embedding is ", self.decoder_word_embedding)
            del tmp_bart
            del self.decoder.embed_tokens
            # self.decoder.embed_tokens.weight.requires_grad = False # freeze 没用上
            
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims) 
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad(): # 对齐  embedding 和 lm 头 需要训练
            self.lm_head.weight = self.word_embedding.weight
            
        print("### build embedding ", self.word_embedding)
        print("### build lm_head ", self.lm_head)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            print("### input_dim != net.hidden_size, build up_proj...")
            if self.self_conditions == 1:
                print("### use self_conditions up proj")
                input_dims = input_dims * 2 ### add self_conditions input up projection [emb_x || self_condition].dim to config.hidden_size     
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        ### model init
        
        if init_pretrained == 'gpt-2':
            pass
                    
        elif init_pretrained == 'bert':
            print('### initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no': # 走这里
            if "bart" in config_name:
                print("### building Net Arch from BART ..")
                self.input_transformers = BartModel(config)
                config.max_position_embeddings = 512 # bart 1024 -> 512
            else:
                self.input_transformers = BertEncoder(config) # 走这里
            
            # 位置编码 bert 512, bart 1024 -> 512
            
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))) 
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            print("### 位置编码 is ", self.position_embeddings)
            
            # LN
            if 'bert' in config_name:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
            else:
                self.LayerNorm = nn.LayerNorm(config.hidden_size)
        else:
            assert False, "invalid type of init_pretrained"
        
        ### init end
        # droput
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            print("### input_dim != net.hidden_size, build down_proj...")
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids): # token_ids -> input_dim vector 
        return self.word_embedding(input_ids)
    
    # def get_decoder_logits(self, hidden_repr): # 没用上
    #     if self.logits_mode == 1: # default
    #         return self.lm_head(hidden_repr)
    #     else: 
    #         raise NotImplementedError

    def get_logits(self, hidden_repr): # hidden_size(config.h_size) -> logits(vocab) 
        if self.logits_mode == 1: # default
            return self.lm_head(hidden_repr)
        
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, self_conditions=None, net_mask=None, decoder_inputs_id=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
                
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        ### add self_conditions before up proj
        if self_conditions is not None:
            x = th.cat((x, self_conditions), dim=-1) # x is [bzs,seqlen,h]. after cat x is [bzs,seqlen,h*2]
            
        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x) 
        else:
            emb_x = x

        seq_length = x.size(1)
        
        # if self.use_en_de == 1:

        position_ids = self.position_ids[:, : seq_length ] # [1, seqlen]
        p_emb = self.position_embeddings(position_ids)
        
        t_emb = emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        
        emb_inputs = p_emb  + emb_x + t_emb

            
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        
        if self.mask == 1: # train is ok but decode is slow
            # print("#### use mask")
            attn_mask = th.Tensor([]).to(emb_inputs.device) 
            # print(attn_mask.device)
            for sep_token_ids in net_mask:
                
                sep = 6 ;seqlen = 256
                
                mask A-1
                mmsk = th.zeros((seq_length, seq_length)).to(emb_inputs.device)  
                
                mmsk[:sep_token_ids[3]+1, :sep_token_ids[3]+1] = 1
                mmsk[sep_token_ids[3]:sep_token_ids[4]+1, :sep_token_ids[1]+1] = 1
                mmsk[sep_token_ids[4]:sep_token_ids[5]+1, sep_token_ids[0]:] = 1
                mmsk[sep_token_ids[3]:sep_token_ids[4]+1, sep_token_ids[3]:sep_token_ids[4]+1] = 1

                # mask B-1
                # mmsk = th.zeros((seq_length, seq_length))
                # mmsk[:sep_token_ids[5]+1, :sep_token_ids[3]+1] = 1
                # mmsk[sep_token_ids[3]:sep_token_ids[5]+1, sep_token_ids[3]:sep_token_ids[4]+1] = 1
                # mmsk[sep_token_ids[4]:sep_token_ids[5]+1, sep_token_ids[4]:sep_token_ids[5]+1] = 1
                
                # mask A-2
                # mmsk = th.ones((seq_length, seq_length))
                # mmsk[:sep_token_ids[3]+1:, sep_token_ids[3]+1:] = 0
                # mmsk[sep_token_ids[4]+1:sep_token_ids[5]+1:, :sep_token_ids[0]] = 0
                # mmsk[sep_token_ids[3]+1:sep_token_ids[4]:, sep_token_ids[1]+1:sep_token_ids[3]+1] = 0
                # mmsk[sep_token_ids[3]+1:, sep_token_ids[3]+1:] = th.tril(mmsk[sep_token_ids[3]+1:, sep_token_ids[3]+1:], diagonal=0)
                
                # mask B-2
                # mmsk = th.ones((seq_length, seq_length))
                # mmsk[:sep_token_ids[3]+1:, sep_token_ids[3]+1:] = 0
                # mmsk[sep_token_ids[3]+1:, sep_token_ids[3]+1:] = th.tril(mmsk[sep_token_ids[3]+1:, sep_token_ids[3]+1:], diagonal=0)
                
            # merge to batch
                # attn_mask = torch.cat((attn_mask, mmsk.unsqueeze(0)))
                
            # attn_mask = attn_mask.unsqueeze(1).to(th.float16)
            
            # print("### use net_mask as attn_mask")

            # TO:DO set it a flag
            # attn_mask = net_mask # mask = 1 when infer it must set it
            
        else:
            attn_mask = None # mask = 0 
        

        # 正常入口
        input_trans_hidden_states = self.input_transformers(emb_inputs, attention_mask=attn_mask).last_hidden_state

        # with th.no_grad():
        #     en_de_loss = mean_flat((emb_x - input_trans_hidden_states) ** 2)
        
        # print(input_trans_hidden_states.shape)
        # 加入decoder _ trg _
        if self.use_en_de == 1:
            if self.output_dims == self.hidden_size:
                decoder_hidden_states = self.decoder(
                    # encoder_outputs=input_trans_hidden_states, # bert_encoder_hidden_state [bsz, seqlne, 768]
                    # decoder_input_ids=decoder_inputs_id,

                    encoder_hidden_states = input_trans_hidden_states,
                    # input_ids=decoder_inputs_id,
                    # inputs_embeds=self.get_embeds(decoder_inputs_id)
                    inputs_embeds=self.get_embeds(decoder_inputs_id) + t_emb
                ).last_hidden_state
            else: # encoder_h 768, decoder embedding 128 need up proj
                decoder_hidden_states = self.decoder(
                    encoder_hidden_states = input_trans_hidden_states,
                    inputs_embeds=self.input_up_proj(self.get_embeds(decoder_inputs_id)) + t_emb # 128 -> 768 + 768 return 768
                ).last_hidden_state # []
                decoder_hidden_states = self.output_down_proj(decoder_hidden_states) # 768 -> 128

        else:
            decoder_hidden_states = None
        
        # input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        
        # return h , en_de_loss, decoder_hidden_states
        return h , decoder_hidden_states
