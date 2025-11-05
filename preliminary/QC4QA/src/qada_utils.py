import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
import random
from collections import defaultdict
from abc import *


def worker_init_fn(worker_id):
    # DataLoader의 worker 초기화 함수
    # 각 worker마다 난수 시드를 고정하여 재현성 확보
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def InfiniteSampling(n):
    # 무한 반복 샘플링을 수행하는 generator
    # n개의 샘플을 무작위 순서로 계속 섞어서 반환
    i = n - 1
    order = np.random.permutation(n)  # 무작위 순열 생성
    while True:
        yield order[i]
        i += 1
        if i >= n:  # 한 바퀴 끝나면 새로 섞어서 반복
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSampler(torch.utils.data.sampler.Sampler):
    # PyTorch Sampler 확장 → 데이터셋을 무한히 샘플링 가능하게 함
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampling(self.num_samples))

    def __len__(self):
        # 사실상 무한, 큰 수 반환
        return 2 ** 31


class DirichletBatcher(metaclass=ABCMeta):
    # Dirichlet 분포를 기반으로 동의어를 샘플링하는 배처 클래스
    def __init__(self, args, lengths_1_order, lengths_2_order, 
                 joint_synonym_matrix, dirichlet_ratio,
                 dirichlet_cache, require_coeff=True):
        self.args = args
        self.lengths_1_order = lengths_1_order      # 1차 동의어 개수
        self.lengths_2_order = lengths_2_order      # 2차 동의어 개수
        self.joint_synonym_matrix = joint_synonym_matrix  # 단어-동의어 매핑 행렬
        self.dirichlet_ratio = dirichlet_ratio      # Dirichlet 샘플링 비율
        self.dirichlet_cache = dirichlet_cache      # 미리 계산된 Dirichlet 분포 캐시
        self.require_coeff = require_coeff          # 계수 사용 여부
   
    def permute(self, in_tensor, chunk_ids, permutation):
        # 주어진 permutation 순서대로 텐서를 재배열
        out_tensor = []
        for chunk in permutation:
            for idx in range(len(in_tensor)):
                if chunk_ids[idx] == chunk:
                    out_tensor.append(in_tensor[idx])
        return torch.tensor(out_tensor)
    
    def batching_fn(self, batch):
        # 하나의 배치를 받아서 동의어 ID 및 Dirichlet 계수를 생성
        ids, masks, segments, starts, ends, all_synonyms, all_coeffs = [], [], [], [], [], [], []
        for i in range(len(batch)):
            synonym_ids = torch.zeros(len(batch[i][0]), self.args.max_synonyms)
            synonym_coeffs = torch.zeros(len(batch[i][0]), self.args.max_synonyms)
            for j in range(len(batch[i][0])):
                token = batch[i][0][j]      # 토큰 ID
                mask_id = batch[i][1][j]    # 마스크 ID
                segment_id = batch[i][2][j] # 세그먼트 ID
                # 일정 확률(dirichlet_ratio) 이상이거나 특수 조건이면 원래 토큰만 사용
                if np.random.rand() > self.dirichlet_ratio or segment_id == 1 or mask_id == 0: 
                    synonym_ids[j, 0] = token
                    synonym_coeffs[j, 0] = 1
                else:
                    # Dirichlet 기반 동의어 샘플링
                    synonyms, coeffs = sample_dirichlet_synonyms(
                            self.args, self.lengths_1_order, self.lengths_2_order, self.joint_synonym_matrix,
                            self.dirichlet_cache, token, i*j, self.require_coeff)
                    synonym_ids[j] = synonyms
                    synonym_coeffs[j] = coeffs
            ids.append(batch[i][0])
            masks.append(batch[i][1])
            segments.append(batch[i][2])
            starts.append(batch[i][3])
            ends.append(batch[i][4])
            all_synonyms.append(synonym_ids)
            all_coeffs.append(synonym_coeffs)
        
        # 텐서 묶어서 반환
        return torch.stack(ids), torch.stack(masks), torch.stack(segments), torch.stack(starts), \
            torch.stack(ends), torch.stack(all_synonyms).long(), torch.stack(all_coeffs)


def sample_dirichlet_synonyms(args, lengths_1_order, lengths_2_order, joint_synonym_matrix, 
                              dirichlet_cache, token, sample_idx, require_coeff=True):
    # 특정 토큰에 대해 Dirichlet 분포 기반 동의어와 계수 샘플링
    synonyms = joint_synonym_matrix[token]          # 동의어 후보
    num_synonym_1_order = lengths_1_order[token]    # 1차 동의어 개수
    num_synonym_2_order = lengths_2_order[token]    # 2차 동의어 개수
    if require_coeff:
        # 캐시된 Dirichlet 분포에서 계수 불러오기
        coeffs = dirichlet_cache[(num_synonym_1_order.item(), \
            num_synonym_2_order.item())][sample_idx%(args.cache_size**2)]
    else:
        coeffs = torch.zeros(joint_synonym_matrix.shape[1])
    
    return synonyms, coeffs


def build_dirichlet_coeff_cache(args, lengths_1_order, lengths_2_order, joint_synonym_matrix):
    # (1차 동의어 수, 2차 동의어 수) 조합별 Dirichlet 분포 캐시 생성
    dirichlet_cache = {}
    for num_synonym_1_order in np.unique(lengths_1_order):
        if num_synonym_1_order == 0:
            continue
        for num_synonym_2_order in np.unique(lengths_2_order):
            # Dirichlet 알파 값 설정 (감쇠 적용)
            alphas = [args.alpha] * 1 + [args.alpha * args.decay] * (num_synonym_1_order - 1) \
                + [args.alpha * args.decay * args.decay] * num_synonym_2_order
            alphas = alphas[:args.max_synonyms]
            # Dirichlet 샘플 생성
            dirichlet = np.random.dirichlet(alphas, args.cache_size**2).astype(np.float32)
            zeros = np.zeros((args.cache_size**2,
                joint_synonym_matrix.shape[1]-dirichlet.shape[1])).astype(np.float32)
            # 패딩 후 텐서로 저장
            dirichlet_cache[(num_synonym_1_order, num_synonym_2_order)] = \
                torch.tensor(np.concatenate((dirichlet, zeros), axis=1))
    
    return dirichlet_cache


def build_joint_synonym_matrix(args, tokenizer=None):
    # 토크나이저 단어 사전을 기반으로 joint synonym matrix 구축
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(
                        args.bert_model, do_lower_case=args.do_lower_case)
    
    # 1차/2차 동의어 사전 생성
    synonym_dict_1_order, synonym_dict_2_order = build_2_order_synonym_dict(args, tokenizer)
    lengths_1_order = np.ones(len(tokenizer.vocab)).astype(int)  # 자기 자신 포함
    lengths_2_order = np.zeros(len(tokenizer.vocab)).astype(int)
    joint_synonym_matrix = np.zeros((len(tokenizer.vocab), args.max_synonyms)).astype(int)
    joint_synonym_matrix[:, 0] = np.arange(len(tokenizer.vocab))  # 자기 자신 포함
    
    # 1차 동의어 채우기
    for key, value in synonym_dict_1_order.items():
        lengths_1_order[key] = max(1, min(len(value), args.max_synonyms))
        joint_synonym_matrix[key, :len(value)] = np.array(value)[:args.max_synonyms].astype(int)
    # 2차 동의어 채우기 (1차 이후 위치부터)
    for key, value in synonym_dict_2_order.items():
        lengths_2_order[key] = min(len(value), args.max_synonyms)
        start_pos = lengths_1_order[key]
        joint_synonym_matrix[key, start_pos:start_pos+len(value)] = \
            np.array(value)[:args.max_synonyms-start_pos].astype(int)
    
    # Torch tensor로 변환
    lengths_1_order = torch.tensor(lengths_1_order).long()
    lengths_2_order = torch.tensor(lengths_2_order).long()
    joint_synonym_matrix = torch.tensor(joint_synonym_matrix).long()
    
    return lengths_1_order, lengths_2_order, joint_synonym_matrix


def build_2_order_synonym_dict(args, tokenizer):
    # 동의어 파일을 읽어 1차/2차 동의어 사전 구축
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(
                        args.bert_model, do_lower_case=args.do_lower_case)
    
    synonym_dict_1_order = defaultdict(list)
    synonym_dict = json.load(open(args.synonym_file, 'r'))
    existing_tokens = list(tokenizer.vocab.keys())
    # 1차 동의어 사전 구축
    for key in list(synonym_dict.keys()):
        if key in existing_tokens:
            tokenized_key = tokenizer.convert_tokens_to_ids(key)[0]
            synonym_dict_1_order[tokenized_key].append(tokenized_key)  # 자기 자신 포함
            if len(synonym_dict[key]) == 0:
                continue
            cur_synonyms = synonym_dict[key]
            for synonym in cur_synonyms:
                if synonym in existing_tokens:
                    synonym_dict_1_order[tokenized_key].append(
                        tokenizer.convert_tokens_to_ids(synonym)[0])
    
    # 2차 동의어 사전 구축
    synonym_dict_2_order = defaultdict(list)
    for token in list(synonym_dict_1_order.keys()):
        synonyms_1_order = synonym_dict_1_order[token]
        for synonym in synonyms_1_order:
            synonym_dict_2_order[token] += synonym_dict_1_order[synonym].copy()
        synonym_dict_2_order[token] = list(set(synonym_dict_2_order[token]))
        # 자기 자신과 1차 동의어는 제외
        if token in synonym_dict_2_order[token]:
            synonym_dict_2_order[token].remove(token)
        for synonym in synonyms_1_order:
            if synonym in synonym_dict_2_order[token]:
                synonym_dict_2_order[token].remove(synonym)
    
    return synonym_dict_1_order, synonym_dict_2_order