Zhangyongqi123import os, sys, glob, json
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AddedToken, BertTokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
from rouge_chinese import Rouge
from tqdm import tqdm
### T5-pegasus import
import jieba
from functools import partial
from transformers import BertTokenizer

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
 
# smoothing_fn = SmoothingFunction().method7
smoothing_fn = None
# 

def get_bleu(recover, reference):
    Bleu = {}
    Bleu["bleu1"] = sentence_bleu([reference.split()], recover.split(), weights=(1, 0, 0, 0),smoothing_function=smoothing_fn)
    Bleu["bleu2"] = sentence_bleu([reference.split()], recover.split(), weights=(0.5, 0.5, 0, 0),smoothing_function=smoothing_fn)
    Bleu["bleu3"] = sentence_bleu([reference.split()], recover.split(), weights=(1./3., 1./3., 1./3., 0),smoothing_function=smoothing_fn)
    Bleu["bleu4"] = sentence_bleu([reference.split()], recover.split(), weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoothing_fn)
    return Bleu

def cal_ppl(sen_list):
    sens = sen_list
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")# 400m
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    inputs = tokenizer(sens, padding='max_length', max_length=128, truncation=True, return_tensors="pt") # 
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    # avg_ppl = np.mean(ppl)
    return ppl

def selectBest(sentences): 
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)['bleu1'] 
            selfBleu[i].append(score) # 默认是bleu-4
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def diversityOfSet(sentences): 
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])['bleu1']
            selfBleu.append(score)
    if len(selfBleu)==0:
        selfBleu.append(0)
    div4 = distinct_n_gram_inter_sent(sentences, 4)
    
    # print("$$$ self-bleu:", selfBleu)
    
    return np.mean(selfBleu), div4

def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)

import re
def map2digit(reference, summary):
    summary=[summary]
    lexicon = set()
    for line in reference:
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            assert isinstance(tokens, list)
            ref = tokens[0]
            for t in ref:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)
    for s in summary:
        for line in s:

            assert isinstance(tokens, list)
            summ = line[0]
            for t in summ:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)

    c2d = {}
    d2c = {}
    for i, value in enumerate(lexicon):
        c2d[value] = str(i)
        d2c[i] = value

    def map_string(text, c2d):

        def spliteKeyWord(str):
            regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
            matches = re.findall(regex, str, re.UNICODE)
            return matches
        str_list = spliteKeyWord(text)
        return ' '.join([c2d[t] if re.search(u'[\u4e00-\u9fff]', t) else t for t in str_list])

    # map to digit
    res_ref = []
    res_summ = []
    for line in reference:
        tmp_s = []
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            ref = tokens[0]  # string
            tmp = map_string(ref, c2d)
            tmp_s.append([tmp])
        res_ref.append(tmp_s)

    for s in summary:
        tmp_s = []
        for line in s:
            assert isinstance(line, list) and len(line) == 1
            summ = line[0]
            tmp = map_string(summ, c2d)
            tmp_s.append([tmp])
        res_summ.append(tmp_s)
    return res_ref, res_summ[0]


def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')
    parser.add_argument('--unk', type=str, default='[UNK]', help='unk token of the sentence')
    parser.add_argument('--tokenizer', type=str, default='jieba', help='tokenizer the chinese sentence')


    args = parser.parse_args()

    ## select tokenizer seems add word is no valid
    if args.tokenizer == "jieba":
        
        jieba.add_word("DIGIT")
        jieba.add_word("<c>")
        jieba.add_word("<title>")
        jieba.add_word("<choices>")
        TOKENIZE_CHINESE = lambda x: ' '.join(
                    jieba.cut(x, cut_all=False, HMM=True)
                )

    elif args.tokenizer == "T5Pegasus":
        # use pre-train model's tokenizer vocab
        tokenizer = T5PegasusTokenizer.from_pretrained("/export/data/yqzhang/diffus-glm/hf/t5-pegasus")
        # add 4 token
        for token in ["DIGIT", "<c>", "<title>", "<choices>"]:
            tokenizer.add_tokens(AddedToken(token, single_word=True), special_tokens=False)

        TOKENIZE_CHINESE = lambda x: ' '.join(
            tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
        )
        
    elif args.tokenizer == "char":
        TOKENIZE_CHINESE = lambda x: ' '.join(x)
    else:
        raise ValueError("no this tokenizer method! ")

    files = sorted(glob.glob(f"{args.folder}/*json"))

    print(f"### use {len(files)} seeds")

    
    sample_num = 0
    with open(files[0], 'r') as f:
        for row in f:
            sample_num += 1

    sentenceDict = {}
    referenceDict = {}
    sourceDict = {}

    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    div4 = []
    selfBleu = []

    for path in files:
        print("### 当前处理文件：",path)
        sources = [] # [src]
        references = [] # [truth]
        recovers = [] # [pred]

        # bleu metrics
        bleu1 = []
        bleu2 = []
        bleu3 = []
        bleu4 = []
        # rouge metrics
        rouge1 = [] ### 
        rouge2 = [] ###
        rougeL = [] ###
        rougeLsum = []

        # len & diversity
        avg_len = []
        dist1 = []
        dist2 = []
        dist3 = []
        dist4 = []
        count = []
        
        ## others
        # ppl_list = []

        with open(path, 'r') as f:
            cnt = 0
            for row in tqdm(f): # jsonl里面每行处理

                
                source = json.loads(row)['source'].strip()
                reference = json.loads(row)['reference'].strip()
                recover = json.loads(row)['recover'].strip()

                # delete special tokens
                source = source.replace(args.eos, '').replace(args.sos, '')
                reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '')
                recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '').replace(args.unk, '').replace("< c >",'<c>').replace("<c><c>",'<c>')

                ### re-tokenizer
                ### delete space
                source = source.replace(' ','')
                reference = reference.replace(' ','')
                recover = recover.replace(' ','')

                count.append(recover.count("<c>") + 1)
                # count.append(reference.count("<c>") + 1) # golden choice nums
                
                
                source = TOKENIZE_CHINESE(source)
                reference = TOKENIZE_CHINESE(reference)
                recover = TOKENIZE_CHINESE(recover)
                
                
                # print(recover)
               
                sources.append(source)
                references.append(reference)
                recovers.append(recover)
                # print("### cal ppl ......")
                ### ppl ###
                # ppl_list.append(cal_ppl(recover.replace(' ','')))
                
                avg_len.append(len(recover.split(' ')))
                # avg_len.append(len(reference.split(' '))) # golden_len
                
                ## cal each sen's bleu
                # print("### cal bleu ......")
                bleu_result = get_bleu(recover, reference)

                bleu1.append(bleu_result['bleu1'])
                bleu2.append(bleu_result['bleu2'])
                bleu3.append(bleu_result['bleu3'])
                bleu4.append(bleu_result['bleu4'])
                
                # print("### cal rouge ......")
                # cal each sen's rouge
                rouge = Rouge()
                rouge_result = rouge.get_scores(recover, reference)[0]
                # rouge_result = rougeScore(recover, reference)
                rouge1.append(rouge_result['rouge-1']['f'])
                
                # import json
                # with open("test.jsonl",'a') as f:
                #     json.dump({"score":rouge_result['rouge-1']['f'],"pred":recover,"gold":reference}, f ,ensure_ascii=False)
                #     f.write("\n")
                #     f.close()
                
                rouge2.append(rouge_result['rouge-2']['f'])
                rougeL.append(rouge_result['rouge-l']['f'])
                # rougeLsum.append(rouge_result['rouge-lsum']['f'])
                
                # rouge1.append(rouge_result['rouge1_fmeasure'])
                # rouge2.append(rouge_result['rouge2_fmeasure'])
                # rougeL.append(rouge_result['rougeL_fmeasure'])
                
                # import json
                # with open("test.jsonl",'a') as f:
                #     # print(type(rouge_result['rouge1_fmeasure']))
                #     json.dump({"score":str(np.array(rouge_result['rouge1_fmeasure'])),"pred":recover,"gold":reference}, f ,ensure_ascii=False)
                #     f.write("\n")
                #     f.close()

                # print("### cal others ......")
                dist1.append(distinct_n_gram([recover], 1)) # return a value in 0~1
                dist2.append(distinct_n_gram([recover], 2))
                dist3.append(distinct_n_gram([recover], 3))
                dist4.append(distinct_n_gram([recover], 4))
                
                sentenceDict[cnt].append(recover) # [pred]
                referenceDict[cnt].append(reference) # [truth]
                sourceDict[cnt].append(source) # [src]

                cnt += 1
                
        # cal ppl
        # print("### cal ppl ......")
        # ppl = cal_ppl([x.replace(' ','') for x in recovers])
        
        # cal bertscore
        # P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
        print("### cal bertscore ......")
        P, R, F1 = score(recovers, references, model_type='bert-base-chinese', lang='zh', verbose=True)

        print('*'*30)
        print('avg BLEU-1 score', np.mean(bleu1))
        print('avg BLEU-2 score', np.mean(bleu2))
        print('avg BLEU-3 score', np.mean(bleu3))
        print('avg BLEU-4 score', np.mean(bleu4))
        print('*'*30)
        print('avg ROUGE-1 score', np.mean(rouge1))
        print('avg ROUGE-2 score', np.mean(rouge2))
        print('avg ROUGE-L score', np.mean(rougeL))
        print('*'*30)
        print('avg berscore', torch.mean(F1))
        print('avg dist1 score', np.mean(dist1))
        print('avg dist2 score', np.mean(dist2))
        print('avg dist3 score', np.mean(dist3))
        print('avg dist4 score', np.mean(dist4))
        print('avg len', np.mean(avg_len))
        print('avg count',np.mean(count))
        # print('avg ppl',np.mean(ppl))

    
    # more than one seeds
    if len(files)>1:
        if not args.mbr:
            print('*'*30)
            print('Compute diversity...')
            print('*'*30)
            for k, v in sentenceDict.items():
                if len(v) == 0:
                    continue
                sb, d4 = diversityOfSet(v)
                selfBleu.append(sb)
                div4.append(d4)

            print('avg selfBleu score', np.mean(selfBleu))
            print('avg div4 score', np.mean(div4))
        
        else: # mbr
            print('*'*30)
            print('MBR...')
            print('*'*30)
            bleu1 = []
            bleu2 = []
            bleu3 = []
            bleu4 = []

            rouge1 = []
            rouge2 = []
            rougeL = []

            avg_len = []
            dist1 = []
            dist2 = []
            dist3 = []
            dist4 = []
            count = []
            ppl_list = []

            recovers = []
            references = []
            sources = []


            for k, v in sentenceDict.items():
                if len(v) == 0 or len(referenceDict[k]) == 0:
                    continue

                # print(v) [str, str]
                recovers.append(selectBest(v)) 
                references.append(referenceDict[k][0])
                sources.append(sourceDict[k][0])

            for (source, reference, recover) in zip(sources, references, recovers):
                bleu_result = get_bleu(recover, reference)
                bleu1.append(bleu_result['bleu1'])
                bleu2.append(bleu_result['bleu2'])
                bleu3.append(bleu_result['bleu3'])
                bleu4.append(bleu_result['bleu4'])

                count.append(recover.replace(" ",'').count("<c>") + 1)
                
                rouge = Rouge()
                rouge_result = rouge.get_scores(recover, reference)[0]
                # print(recover,"\t",reference,"\t",rouge_result['rouge-1']['f'])
                # import json
                # with open("test.jsonl",'a') as f:
                #     json.dump({"score":rouge_result['rouge-1']['f'],"pred":recover,"gold":reference}, f ,ensure_ascii=False)
                #     f.write("\n")
                #     f.close()
                
                rouge1.append(rouge_result['rouge-1']['f'])
                rouge2.append(rouge_result['rouge-2']['f'])
                rougeL.append(rouge_result['rouge-l']['f'])
                
                # rouge_result = rougeScore(recover, reference)
                # rouge1.append(rouge_result['rouge1_fmeasure'])
                # rouge2.append(rouge_result['rouge2_fmeasure'])
                # rougeL.append(rouge_result['rougeL_fmeasure'])
                
                # import json
                # with open("test.jsonl",'a') as f:
                #     json.dump({"score":np.array(rouge_result['rouge1_fmeasure']),"pred":recover,"gold":reference}, f ,ensure_ascii=False)
                #     f.write("\n")
                #     f.close()

                avg_len.append(len(recover.split(' ')))
                dist1.append(distinct_n_gram([recover], 1))
                dist2.append(distinct_n_gram([recover], 2))
                dist3.append(distinct_n_gram([recover], 3))
                dist4.append(distinct_n_gram([recover], 4))
            
            print("## cal ppl...")
            ppl_list = cal_ppl([x.replace(' ','') for x in recovers])

            # print(len(recovers), len(references), len(recovers))
            
            # P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
            P, R, F1 = score(recovers, references, model_type='bert-base-chinese', lang='zh', verbose=True)

            print('*'*30)
            print('avg BLEU-1 score', np.mean(bleu1))
            print('avg BLEU-2 score', np.mean(bleu2))
            print('avg BLEU-3 score', np.mean(bleu3))
            print('avg BLEU-4 score', np.mean(bleu4))
            print('*'*30)
            print('avg ROUGE-1 score', np.mean(rouge1))
            print('avg ROUGE-2 score', np.mean(rouge2))
            print('avg ROUGE-L score', np.mean(rougeL))
            print('*'*30)
            print('avg berscore', torch.mean(F1))
            print('avg dist1 score', np.mean(dist1))
            print('avg len', np.mean(avg_len))
            print('avg count', np.mean(count))
            print('avg ppl', np.mean(ppl_list))
            
            
