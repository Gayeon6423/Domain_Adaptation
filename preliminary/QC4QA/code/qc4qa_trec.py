from __future__ import absolute_import, division
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import io
import os
import sys
import logging
import json
import torch
import nltk
import numpy as np
import re
from infersent import InferSent
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(0)
import tensorflow_hub as hub

# Set PATHs
PATH_SENTEVAL = '../SentEval'  # specify SentEval root if not installed
PATH_TO_DATA = ''  # not necessary for inference
MODEL_VERSION = 1
PATH_TO_W2V = '../SentEval/glove/glove.840B.300d.txt' if MODEL_VERSION == 1 else '../SentEval/fasttext/crawl-300d-2M.vec'
MODEL_PATH = "../SentEval/encoder/infersent%s.pkl" % MODEL_VERSION
V = 1 # version of InferSent

sys.path.insert(0, PATH_SENTEVAL)
import senteval
from senteval.tools.classifier import MLP

def clean_sentence(text: str) -> str:
    # 소유격, 축약형 's 분리
    text = re.sub(r"(\w+)'s", r"\1 's", text)
    # 축약형 단순 변환
    text = text.replace("n't", " not")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    text = text.replace("'m", " am")
    text = text.replace("What's", "What is") 
    # 하이픈 단어 분리
    text = re.sub(r"(\w+)-(\w+)", r"\1 \2", text)
    # 불필요한 특수문자 제거
    text = re.sub(r"[^A-Za-z0-9\s\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_sentences(sent_list):
    return [clean_sentence(s) for s in sent_list]

def loadFile(fpath):
    """
    MRQA 스타일 데이터 파일을 읽어 질문과 질문 ID를 추출
    파일 한 줄씩 읽어 JSON으로 파싱->각 질문 토큰화하여 저장
    """
    qa_data = []
    qa_ids = []
    # tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
    #             'HUM': 3, 'LOC': 4, 'NUM': 5}
    with io.open(fpath, 'r', encoding='utf-8') as f:
        for example in f:
            if "header" in json.loads(example):
                continue
            paragraph = json.loads(example)
            for qa in paragraph['qas']:
                qa_data.append(qa['question'].split())
                qa_ids.append(qa['qid'])
    return qa_data, qa_ids

def loadSQuAD(fpath):
    """
    SQuAD 스타일 데이터 파일을 읽어 질문과 질문 ID를 추출
    파일 한 줄씩 읽어 JSON으로 파싱->각 질문 토큰화하여 저장
    """
    qa_data = []
    qa_ids = []
    # tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
    #             'HUM': 3, 'LOC': 4, 'NUM': 5}
    with io.open(fpath, 'r', encoding='utf-8') as f:
        input_data = json.load(f)["data"]
        input_data = input_data ################ 개수 줄이기 ################ [:30]
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                qa_data.append(qa['question'].split())
                qa_ids.append(qa['id'])
    return qa_data, qa_ids

def prepare(params, samples):
    # InferSent 모델에 사용할 단어 사전을 구축하는 함수
    # samples: 문장(질문) 리스트
    # 축약형·소유격·하이픈 결합 단어 전처리 
    sentences = [' '.join(sent) if sent != [] else '.' for sent in samples]
    sentences = preprocess_sentences(sentences)
    params['infersent'].build_vocab(sentences, tokenize=False)

def batcher(params, batch):
    # 입력된 질문 배치를 임베딩 벡터로 변환하는 함수
    # 1. 각 질문(토큰 리스트)을 문자열로 합침
    sentences = [' '.join(sent) if sent != [] else '.' for sent in batch]
    # 2. 축약형·소유격·하이픈 결합 단어 전처리
    sentences = preprocess_sentences(sentences)
    # 3. InferSent 모델로 임베딩 생성
    embeddings1 = params['infersent'].encode(sentences, bsize=params['classifier']['batch_size'], tokenize=False)
    # 4. Google Universal Sentence Encoder로 임베딩 생성
    embeddings2 = params['google_use'](sentences)
    embeddings2 = embeddings2.numpy()
    # 5. 두 임베딩을 합쳐서 반환 (문장 의미를 더 풍부하게 표현)
    return np.concatenate((embeddings1, embeddings2), axis=-1)

def make_embed_fn(module):
    embed = hub.load(module)
    f = embed.signatures["default"]
    return lambda x: f(tf.constant(x))["default"]

def getEmbeddings(qa_data, params):
    # 전체 질문 리스트를 배치 단위로 나눠 임베딩을 생성하는 함수
    out_embeds = []
    # 배치 크기만큼 반복하며 임베딩 생성
    for ii in range(0, len(qa_data), params['classifier']['batch_size']):
        batch = qa_data[ii:ii + params['classifier']['batch_size']]
        # batcher 함수로 임베딩 생성
        embeddings = batcher(params, batch)
        out_embeds.append(embeddings)
    # 모든 배치 임베딩을 하나로 합쳐 반환(행렬 세로 방향으로 이어붙임)
    # 입력: [batch_size개의 문장] -> 출력: (batch_size, emb_dim)
    return np.vstack(out_embeds) # (N, 4096+512)

def updateFile(fpath, q_type, q_ids):
    # MRQA 스타일 데이터 파일에 질문 유형(q_type)을 추가하는 함수
    paragraphs = []
    # 원본 파일을 한 줄씩 읽어서 JSON 객체로 파싱
    with io.open(fpath, 'r', encoding='utf-8') as f:
        for example in f:
            # header가 포함된 줄은 건너뜀
            if "header" in json.loads(example):
                continue
            paragraph = json.loads(example)
            paragraphs.append(paragraph)
    total_idx = 0
    # 각 paragraph의 qas 리스트를 순회하며 질문 ID(qid)와 분류 결과(q_type)를 매칭
    for paragraph in paragraphs:
        for qa in paragraph['qas']:
            # 예측된 질문 유형을 해당 질문에 추가
            if qa['qid'] == q_ids[total_idx]:
                qa['q_type'] = q_type[total_idx]
                total_idx += 1
            else:
                # 질문 ID가 맞지 않으면 경고 출력
                print('Can not match qid:', q_ids[total_idx])
    # 새로운 파일로 결과 저장 (원본 파일명에서 .jsonl을 _classified.jsonl로 변경)
    with open(fpath[:-6]+'_classified_09151457.jsonl', 'w') as f:
        for sample in paragraphs:
            f.write(json.dumps(sample)+'\n')
    f.close()

def updateSQuAD(fpath, q_type, q_ids):
    # SQuAD 스타일 데이터 파일에 질문 유형(q_type)을 추가하는 함수
    with io.open(fpath, 'r', encoding='utf-8') as f:
        input_data = json.load(f)["data"]
        input_data = input_data ################ 개수 줄이기 ################
    total_idx = 0
    # SQuAD 구조에 맞게 각 질문에 대해 분류 결과를 추가
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                # 예측된 질문 유형을 해당 질문에 추가
                if qa['id'] == q_ids[total_idx]:
                    qa['q_type'] = q_type[total_idx]
                    total_idx += 1
                else:
                    # 질문 ID가 맞지 않으면 경고 출력
                    print('Can not match qid:', q_ids[total_idx])
    
    # 새로운 파일로 결과 저장 (원본 파일명에서 .json을 _classified.json로 변경)
    with open(fpath[:-5]+'_classified_09151457.json', 'w') as f:
        f.write(json.dumps({"data": input_data})+'\n')
    f.close()

if __name__ == "__main__":
    # Set
    encoder = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 512, 'optim': 'rmsprop', 'batch_size': 16,
                                    'tenacity': 5, 'epoch_size': 4}
    params_senteval['google_use'] = encoder
    params_model = {'bsize': 16, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)
    params_senteval['infersent'] = model.cuda().eval()

    # Set Parameter
    file_path = '../data/squad/train-v1.1.json'
    all_qs, all_q_ids = loadSQuAD(file_path)
    q_type = []
    clf = MLP(params_senteval['classifier'], inputdim=4096+512, nclasses=6, batch_size=16)
    clf.model.load_state_dict(torch.load('../model/qc4qa_model.pth'))  
    clf.model.eval()
    
    with torch.no_grad(): 
        # 질문을 배치 단위로 임베딩->MLP예측->결과 누적
        for i in range(0, len(all_qs), 1000): 
            qs = all_qs[i:1000+i] # 현재 배치에 해당하는 질문 1000개 선택
            prepare(params_senteval, qs) # InferSent모델에 맞게 단어 사전 구축
            # 선택된 질문을 InferSent와 Google USE로 임베딩 벡터로 변환
            embeds = getEmbeddings(qs, params_senteval) 
            # 임베딩된 질문을 MLP 분류기에 입력하여 질문 유형 예측
            out = clf.predict(embeds)
            # 예측 결과를 리스트로 변환하여 전체 결과(q_type)에 추가
            q_type += np.array(out).squeeze().astype(int).tolist()
            
    updateSQuAD(file_path, q_type, all_q_ids)