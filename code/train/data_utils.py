import collections
import gzip
import json
import math
import re
import string
import sys
from copy import deepcopy
import pickle
import json_lines
import numpy as np
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from tqdm import tqdm


def read_features_and_examples(args, file_name, tokenizer, logger, use_simple_feature=False, read_examples=False,
        limit=None):
    """
    SQuAD 또는 MRQA 형식의 데이터 파일을 읽어 InputFeatures와 SquadExample 리스트로 변환(캐시 파일 있으면 불러오고, 업으면 새로 생성 후 저장)
    """
    cached_features_file = file_name + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length))
    if use_simple_feature:
        cached_features_file = cached_features_file + '_simple'

    examples, features = None, None
    if read_examples:
        try:
            examples = read_squad_examples(input_file=file_name, is_training=True, logger=logger)
        except:
            examples = read_mrqa_examples(input_file=file_name, is_training=True, logger=logger)
    try:
        with open(cached_features_file, "rb") as reader:
            features = pickle.load(reader)
    except:
        if examples is None:
            try:
                examples = read_squad_examples(input_file=file_name, is_training=True, logger=logger)
            except:
                examples = read_mrqa_examples(input_file=file_name, is_training=True, logger=logger)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True, logger=logger, use_simple_feature=use_simple_feature)
        logger.info("  Saving eval features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as writer:
            pickle.dump(features, writer)
    if limit is not None:
        features = features[:limit]
        if examples is not None:
            examples = examples[:limit]
    return features, examples


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    하나의 SQuAD 질문-문맥-답변 데이터를 저장하는 클래스
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 answers=None,
                 q_type=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text # Ground Truth 실제 정답(학습 활용)
        self.orig_answers = answers # 정답 후보 리스트(평가 활용)
        self.start_position = start_position
        self.end_position = end_position
        self.q_type = q_type

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.q_type:
            s += ", q_type: %d" % (self.q_type)
        return s

class InputFeatures(object):
    """
    A single set of features of data.
    한 개의 example을 BERT 입력 형식으로 변환한 결과 저장
    """
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 q_type=None):
        self.unique_id = unique_id
        self.example_index = example_index # 어떤 example에서 나온 feature인지
        self.doc_span_index = doc_span_index # 긴 문맥을 자른 조각 번호
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map # WordPiece ↔ 원래 토큰 매핑
        self.token_is_max_context = token_is_max_context # 해당 토큰이 최적 context인지 여부
        self.input_ids = input_ids  # BERT vocab id
        self.input_mask = input_mask # 실제 토큰(1)과 패딩(0)
        self.segment_ids = segment_ids # 질문(0)과 문맥(1) 
        self.start_position = start_position # 문맥 토큰 기준 정답 시작 인덱스
        self.end_position = end_position # 문맥 토큰 기준 정답 끝 인덱스
        self.q_type = q_type # 질문 타입(옵션)

class InputFeaturesSimple:
    def __init__(self, unique_id, example_index, doc_span_index, input_ids, input_mask, segment_ids, start_position,
            end_position, q_type=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.q_type = q_type


def read_squad_len(input_file):
    """
    SQuAD 데이터셋의 QA 개수를 반환. (len 키가 없으면 직접 세어줌)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as reader:
            squad_len = json.load(reader)['len']
        return squad_len
    except:
        if 'train-v1.1' in input_file:  # bad coding, should fix this later
            return 87599
        elif 'dev-v1.1' in input_file:
            return 10570
        else:
            num_qa = 0
            with open(input_file, "r", encoding='utf-8') as reader:
                for example in reader:
                    if "header" in json.loads(example):
                        continue
                    paragraph = json.loads(example)
                    for qa in paragraph["qas"]:
                        num_qa += 1
            return num_qa

def read_squad_examples(input_file, is_training, logger):
    """
    SQuAD v1.1 JSON 파일을 읽어서 SquadExample 리스트 생성
    data구조: data > title,paragraphs > context,qas > question,id,q_type,answers > text,answer_start
    """
    # JSONL 형식 체크 (파일 확장자 또는 첫 줄로 판단)
    if input_file.endswith('.jsonl'):
        # JSONL 형식: 각 줄이 {"title": ..., "paragraphs": [...]}
        input_data = []
        with open(input_file, "r", encoding='utf-8') as reader:
            for line in reader:
                input_data.append(json.loads(line))
    else:
        # JSON 형식: {"data": [{"title": ..., "paragraphs": [...]}]}
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

    def is_whitespace(c):
        """
        whiltespace: 공백 문자(띄어쓰기, 탭, 줄바꿈 등 단어를 구분하는 문자) 있는지 확인
        문맥을 띄어쓰기 기준으로 토큰화하기 위한 전처리 단계
        True면 단어 구분자로 인식->새로운 단어 시작으로 간주 
        False면 현재 글자가 단어의 일부로 인식 -> 직전 단어에 이어붙임
        """
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c) # 새로운 단어 시작으로 간주 -> 새 단어로 추가
                    else:
                        doc_tokens[-1] += c # 직전 단어에 이어붙임
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                try:
                    q_type = qa["q_type"]
                except:
                    q_type = None
                orig_answer_text = None
                answers = None
                if is_training:
                    # if len(qa["answers"]) != 1:
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    answers = list(map(lambda x: x['text'], qa['answers']))
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        # logger.warning("Could not find answer: '%s' vs. '%s'",
                        #                    actual_text, cleaned_answer_text)
                        continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    answers=answers,
                    q_type=q_type)
                examples.append(example)
    return examples


def read_mrqa_examples(input_file, is_training, logger, train_qid_list=None):
    """ Read an MRQA jsonl file into a list of SQuADExample."""
    # Similar to read_squad_examples, but no negatives ($version_2_with_negative$)
    paragraphs = []
    num_paragraphs = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        for example in reader:
            if "header" in json.loads(example):
                continue
            paragraph = json.loads(example)
            if train_qid_list and len(paragraph["qas"]) == 1:
                if paragraph["qas"][0]['qid'] not in train_qid_list:
                    continue
            paragraphs.append(paragraph)
            num_paragraphs += 1
    #pdb.set_trace()
    #chosen_paragraph_indices = np.random.choice(range(num_paragraphs), 100, replace=False)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for paragraph_index, paragraph in enumerate(tqdm(paragraphs)):
        # if paragraph_index == 1000:
        #     break
        #if is_training and paragraph_index not in chosen_paragraph_indices:
        #    continue
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        #pdb.set_trace()
        for qa in paragraph["qas"]:
            qas_id = qa["qid"]
            if train_qid_list:
                if qas_id not in train_qid_list:
                    continue
            question_text = qa["question"]
            start_position = None
            end_position = None
            try:
                q_type = qa["q_type"]
            except:
                q_type = None
            orig_answer_text = None
            is_impossible = False  # Always False
            if is_training:
                #print("Note: Training on MRQA.")
                #if len(qa["detected_answers"]) != 1:
                #    import pdb
                #    pdb.set_trace()
                #    raise ValueError("For training, each question should have exactly one answer.")
                answer = qa["detected_answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["char_spans"][0][0]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                answers = [answer['text'] for answer in qa["detected_answers"]]
                if answer_offset + answer_length - 1 >= len(char_to_word_offset):
                    # logger.warning("Could not find answer, out of bounds.")
                    continue
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                
                if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
                    #pdb.set_trace()
                    # logger.warning("Could not find answer: '%s' vs. '%s'",
                    #     actual_text, cleaned_answer_text)
                    continue
            example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    answers=answers,
                    q_type=q_type)
            examples.append(example)
    #pdb.set_trace()
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training, logger, use_simple_feature=False):
    """
    SquadExample → InputFeatures 변환
    질문과 문맥 토큰화 → 문맥 길면 슬라이딩 윈도우 → [CLS] 질문 [SEP] 문맥 [SEP] 구조로 변환 → 정답 위치를 인덱스로 매핑
    - `examples`: SquadExample 리스트 / `tokenizer`: BERT 토크나이저 / `max_seq_length`: 최대 입력 길이
    - `doc_stride`: 긴 문맥 잘라내는 stride 크기 / `max_query_length`: 질문 최대 길이 /`is_training`: 정답 포함 여부
    """
    unique_id = 1000000000   # 각 feature에 고유 ID 부여 시작값

    features = []   # 변환된 InputFeatures를 담을 리스트
    for example_index in tqdm(range(len(examples))):  # 모든 example 순회
        example = examples[example_index]  # 하나의 SquadExample
        try:
            q_type = example.q_type       # 질문 타입 (있을 수도 있고 없을 수도 있음)
        except:
            q_type = None
        query_tokens = tokenizer.tokenize(example.question_text)  # 질문 토큰화

        # 질문이 너무 길면 잘라내기
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 문맥 토큰을 WordPiece 단위로 분해하면서 매핑 정보 저장
        tok_to_orig_index = []   # WordPiece → 원래 단어 인덱스
        orig_to_tok_index = []   # 원래 단어 인덱스 → WordPiece 시작 위치
        all_doc_tokens = []      # 문맥 전체 WordPiece 토큰
        for (i, token) in enumerate(example.doc_tokens):  # 원문 단어 단위 토큰 순회
            orig_to_tok_index.append(len(all_doc_tokens))  # 현재 WordPiece 시작 인덱스 기록
            sub_tokens = tokenizer.tokenize(token)         # WordPiece 분리
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)                # 각 sub_token이 원래 몇 번째 단어였는지
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:  # 학습 모드일 때만 정답 위치 매핑
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # WordPiece 토큰화 후 정답 span을 더 잘 맞도록 보정
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # 입력 최대 길이에서 [CLS], [SEP], [SEP] 3개 토큰 뺀 만큼 문맥에 할당 가능
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # 문맥이 너무 길 경우 슬라이딩 윈도우로 나눔
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))  # 문맥 조각 저장
            if start_offset + length == len(all_doc_tokens):  # 마지막 조각이면 종료
                break
            start_offset += min(length, doc_stride)  # stride 만큼 이동하며 다음 조각 생성

        # 각 문맥 조각(doc_span)을 feature로 변환
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []            # 최종 토큰 시퀀스 ([CLS] 질문 [SEP] 문맥 [SEP])
            token_to_orig_map = {} # WordPiece → 원래 단어 인덱스 매핑
            token_is_max_context = {} # 해당 토큰이 가장 좋은 context인지 여부
            segment_ids = []       # 질문/문맥 구분 (0=질문, 1=문맥)

            # [CLS] 추가
            tokens.append("[CLS]")
            segment_ids.append(0)

            # 질문 토큰 추가
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            # [SEP] 추가
            tokens.append("[SEP]")
            segment_ids.append(0)

            # 문맥 토큰 추가
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # WordPiece 인덱스 → 원문 단어 인덱스 매핑
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                # 이 토큰이 가장 좋은 context인지 확인
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context

                tokens.append(all_doc_tokens[split_token_index])  # WordPiece 토큰 추가
                segment_ids.append(1)  # 문맥=1

            # 문맥 끝나면 [SEP] 추가
            tokens.append("[SEP]")
            segment_ids.append(1)

            # 토큰들을 ID로 변환
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # attention mask: 실제 토큰=1, 패딩=0
            input_mask = [1] * len(input_ids)

            # 길이를 max_seq_length로 맞추기 (패딩 추가)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            # 길이 확인 / 길이 틀리면 AssertionError 발생
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:  # 학습 모드라면 정답 위치 계산
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                # 정답이 현재 span에 포함되지 않으면 스킵
                if (example.start_position < doc_start or
                        example.end_position < doc_start or
                        example.start_position > doc_end or example.end_position > doc_end):
                    continue

                # 질문 + [CLS], [SEP] 만큼 오프셋을 고려하여 실제 위치 보정
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            if use_simple_feature:
                features.append(InputFeaturesSimple(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        q_type=q_type))
            else:
                # 원본 feature 객체 생성 및 추가
                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        q_type=q_type))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """
    WordPiece 토크나이저 적용 후 실제 정답 span과 annotation이 더 잘 맞도록 보정.
    SQuAD의 정답은 문자 단위로 제공됨
    BERT는 WordPiece 단위로 토큰화해서 문자열이 잘게 쪼개지거나 다르게 표현될 수 있음.
    -> annotation에서 준 정답 범위(input_start, input_end)가 토큰 단위랑 안 맞을 수 있음.
    * 원래 span (1895-1943).이 ( 1895 - 1943 ) . 로 토큰화되는 경우 span을 1895 하나로 보정
    * 원래 span이 Japanese로 토큰화되는 경우 span을 Japanese로 보정 못함, 그대로 유지(정답이 Japan이어도)
    """

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However ## 원래 span ##
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match ## 토큰화 후 ##
    # the exact answer, 1895. ## 정확히 추출하도록 span 보정 ##
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    
    # 원래 정답(orig_answer_text)을 WordPiece 기준으로 다시 토큰화한 문자열
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    # 기존 정답 span 근처에서 토큰 단위로 정확히 매칭되는 span을 찾음
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                # 정확히 매칭되는 span 찾으면 반환
                return (new_start, new_end)
    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    긴 문맥을 슬라이딩 윈도우로 나눌 때, 
    어떤 토큰이 여러 span에 걸쳐 등장했을 경우 “어느 span이 이 토큰을 대표해야 하는가?”를 결정
    하나의 span만 대표(span of max context)로 지정해야 함
    * 좌우 컨텍스트가 균형잡힌 span이 더 높은 점수를 받음(좌우 토큰 개수 중 작은 값이 상대 보다 클수록)
    """

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # "bought" 토큰은 Span B와 Span C에 모두 속함. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start # 해당 토큰의 왼쪽에 있는 토큰 개수
        num_right_context = end - position # 해당 토큰의 오른쪽에 있는 토큰 개수
        # 해당 토큰의 좌우 컨텍스트 중 작은 값 + 0.01 * doc_span 길이
        # => 좌우 컨텍스트가 균형잡힌 span이 더 높은 점수를 받음
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        # 가장 큰 점수를 받은 span이 이 토큰의 최대 컨텍스트로 지정됨
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(args, all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging, logger, write_json):
    """
    모델 예측 logits → 실제 텍스트 답변으로 변환 후 JSON 파일로 저장
    """
    logger.info("Writing predictions to: %s" % (output_prediction_file))  # 결과 저장 경로 로그
    logger.info("Writing nbest to: %s" % (output_nbest_file))            # 상위 n개 후보 저장 경로 로그

    example_index_to_features = collections.defaultdict(list)  # example → feature 매핑
    for feature in all_features:  # 각 feature를 example_index 기준으로 묶음
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}  # unique_id → 모델 결과(logit) 매핑
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # prelim(임시) prediction 구조체 정의
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()  # 최종 답변 저장
    all_nbest_json = collections.OrderedDict()   # 상위 n개 후보 저장
    all_probs, all_indices = [], []              # softmax 확률 / 예측된 span 인덱스
    for (example_index, example) in enumerate(all_examples):  # 각 example에 대해
        features = example_index_to_features[example_index]   # 해당 example의 features 가져옴
        all_probs.append(0)  # 확률 placeholder
        if len(features) == 0:
            continue

        prelim_predictions = []  # 후보 span 저장
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]  # 해당 feature의 모델 결과 가져오기

            # start, end 상위 n개 인덱스 추출
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 잘못된 후보(span이 문맥 밖에 있거나, max length 초과 등) 필터링
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    # 유효한 후보 span만 추가
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        # logit 합(start+end) 기준으로 내림차순 정렬
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        # n-best 후보 구조체 정의
        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_position", "end_position", "doc_span_index"])

        seen_predictions = {}  # 중복 텍스트 방지
        nbest = []             # n-best 후보 저장
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:  # 상위 n개까지만
                break
            feature = features[pred.feature_index]

            # 토큰 단위 예측 → 원문 매핑
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # WordPiece 후처리 ("##" 제거 등)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # 공백 정리
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            # 최종 텍스트 보정 (WordPiece vs 원문 align)
            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:  # 이미 본 텍스트면 스킵
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_position=pred.start_index,
                    end_position=pred.end_index,
                    doc_span_index=feature.doc_span_index))

        # 후보가 아예 없는 경우 대비 → dummy prediction 추가
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_position=-1, end_position=-1, doc_span_index=-1))

        assert len(nbest) >= 1  # 최소 하나는 있어야 함

        # softmax 확률 계산
        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
        probs = _compute_softmax(total_scores)

        # nbest 후보들을 JSON 구조로 변환
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1  # 최소 하나는 있어야 함

        # 최종 예측: 가장 확률 높은 후보 선택
        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_probs[example_index] = probs[0]
        all_nbest_json[example.qas_id] = nbest_json

        # 예측된 start/end 인덱스도 기록
        pred_start_pos, pred_end_pos = nbest[0].start_position, nbest[0].end_position
        pred_doc_span_index = nbest[0].doc_span_index
        for feature in features:
            cur_start_pos = pred_start_pos + (pred_doc_span_index - feature.doc_span_index) * args.doc_stride
            cur_end_pos = pred_end_pos + (pred_doc_span_index - feature.doc_span_index) * args.doc_stride
            if cur_end_pos in range(args.max_seq_length) and cur_start_pos in range(args.max_seq_length):
                all_indices.append((cur_start_pos, cur_end_pos))
            else:
                all_indices.append((-1, -1))

    # JSON 파일로 저장
    if write_json:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions, all_probs, all_indices  # 최종 결과 반환



def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging=False):
    """토크나이즈된 예측(pred_text)을 원래 텍스트(orig_text)로 다시 투영하는 함수"""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    # 공백 제거용 내부 함수 정의
    def _strip_spaces(text):
        ns_chars = []  # 공백 제거된 문자 리스트
        ns_to_s_map = collections.OrderedDict()  # 공백 제거된 문자 인덱스 → 원래 인덱스 매핑
        for (i, c) in enumerate(text):
            if c == " ":
                continue  # 공백은 무시
            ns_to_s_map[len(ns_chars)] = i  # 공백 제거 후 인덱스 → 원래 인덱스
            ns_chars.append(c)
        ns_text = "".join(ns_chars)  # 공백 제거된 문자열
        return (ns_text, ns_to_s_map)

    # 원본 텍스트를 토크나이즈
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))  # WordPiece 기반 토큰화

    # pred_text가 tok_text 안에서 어디서 시작하는지 찾음
    start_position = tok_text.find(pred_text)
    if start_position == -1:  # 못 찾으면
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text  # 원래 텍스트 반환
    end_position = start_position + len(pred_text) - 1  # 끝 위치 계산

    # 공백 제거한 원본 텍스트와 토큰화된 텍스트 비교
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    # 공백 제거 후 길이가 다르면 매핑 실패 → 원본 반환
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # 토큰화된 텍스트 인덱스를 공백 제거 인덱스로 매핑
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    # pred_text의 시작 위치를 원래 텍스트 인덱스로 변환
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:  # 시작 매핑 실패 시 원본 반환
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    # pred_text의 끝 위치를 원래 텍스트 인덱스로 변환
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:  # 끝 매핑 실패 시 원본 반환
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    # 최종적으로 원래 텍스트에서 해당 범위를 잘라 반환
    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _get_best_indexes(logits, n_best_size):
    """ logits 값이 큰 상위 n개의 인덱스를 반환"""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """
    주어진 점수 리스트 logits에 softmax 적용 → 확률 분포 반환
    """
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def warmup_linear(x, warmup=0.002):
    # BERT 논문에서 사용한 학습률 스케줄. 초기에는 선형 증가, 이후 일정 유지.
    if x < warmup:
        return x/warmup
    return 1.0
