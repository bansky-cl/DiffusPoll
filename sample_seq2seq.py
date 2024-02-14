"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb, ##
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    # decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False) # 原来的
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, cf_w=0.0, cf_type='default') # 加了cf
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0
    
    # th.distributed.barrier(device_ids=int(os.environ["LOCAL_RANK"])

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print("### use ckpts'config path is ", config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    
    print("### load and update args is ...", args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)
    
    print(f"### vocab in infer is {tokenizer.vocab_size}")
    
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())

    
    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:

        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x) # x_0 [bzs, seqlen, 128]
        
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask
        
        # noise, then conditional data
        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)
   
        ### add my mask # when use mask in training, it should use mask on inferencee too.
        if args.mask == 1:
            # print("构造mask")
            net_mask = th.as_tensor(cond.pop('attn_mask')).to(th.long).to(input_ids_x.device) # 这里串de
            
            attn_mask = th.Tensor([]).to(input_ids_x.device) 
            t1 = time.time()
            for sep_token_ids in net_mask:
                # mask A-1
                mmsk = th.zeros((256, 256)).to(input_ids_x.device)  
                
                mmsk[:sep_token_ids[3]+1, :sep_token_ids[3]+1] = 1
                mmsk[sep_token_ids[3]:sep_token_ids[4]+1, :sep_token_ids[1]+1] = 1
                mmsk[sep_token_ids[4]:sep_token_ids[5]+1, sep_token_ids[0]:] = 1
                mmsk[sep_token_ids[3]:sep_token_ids[4]+1, sep_token_ids[3]:sep_token_ids[4]+1] = 1
                
                attn_mask = th.cat((attn_mask, mmsk.unsqueeze(0)))
                
            attn_mask = attn_mask.unsqueeze(1).to(th.float16)
            
            net_mask = attn_mask
            
            # t2 = time.time()
            # print("### t2 - t1",t2 - t1)
        else:
            net_mask = None
        
        # make unconditional noised data
        
        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        
        # print(model_emb.weight.device) # cuda
        
        print("### sample start ")
        # print("model", model)
        print("model", next(model.input_transformers.parameters()).device) # cuda:0
        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb), # model_emb 在cuda 里面
            model_kwargs=model_kwargs,
            progress=True, ## add pbar
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap,
            net_mask=net_mask, # add mask
            cf_w=args.cf_w, ## cf
            cf_type=args.cf_type, ## cf
        )

        # print("$$$$", len(samples))
        # sample = samples[-1]
        for x in range(100,1000+1,100):
            print(f"write {x} step samples...")
            sample = samples[x-1]
            logits = model.get_logits(sample)
            cands = th.topk(logits, k=1, dim=-1)
            word_lst_recover = []
            word_lst_ref = []
            word_lst_source = []
            
            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                len_x = args.seq_len - sum(input_mask).tolist()
                tokens = tokenizer.decode_token(seq[len_x:])
                # 这里加gpt筛选

                word_lst_recover.append(tokens)

            for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
                # tokens = tokenizer.decode_token(seq)
                len_x = args.seq_len - sum(input_mask).tolist()
                word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
                word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

            for i in range(world_size):
                if i == rank:  # Write files sequentially
                    fout = open(out_path.replace(".json","") + '_' + str(x)+ ".json", 'a') # change path
                    for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                        print(json.dumps({"recover": recov, "reference": ref, "source": src}, ensure_ascii=False), file=fout) # zh
                    fout.close()
                dist.barrier()
            
        # print('decoding for seq2seq', )
        # print(sample.shape)
        # print("### get logits")
#         logits = model.get_logits(sample)  # bsz, seqlen, vocab
#         cands = th.topk(logits, k=1, dim=-1)

#         word_lst_recover = []
#         word_lst_ref = []
#         word_lst_source = []

        # tokenizer = load_tokenizer(args)
        # print("### decode start")
#         for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
#             len_x = args.seq_len - sum(input_mask).tolist()
#             tokens = tokenizer.decode_token(seq[len_x:])
#             # 这里加gpt筛选
            
#             word_lst_recover.append(tokens)

#         for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
#             # tokens = tokenizer.decode_token(seq)
#             len_x = args.seq_len - sum(input_mask).tolist()
#             word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
#             word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

#         for i in range(world_size):
#             if i == rank:  # Write files sequentially
#                 fout = open(out_path, 'a')
#                 for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
#                     print(json.dumps({"recover": recov, "reference": ref, "source": src},ensure_ascii=False), file=fout) # zh
#                 fout.close()
#             dist.barrier()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
