""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """
    텍스트를 소문자로 변환하고, 구두점, 관사(영어의 a, an, the), 불필요한 공백을 제거하는 함수
    """
    def remove_articles(text):
        # 관사(a, an, the)를 공백으로 치환
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # 연속된 공백을 하나로 줄이고, 앞뒤 공백 제거
        return ' '.join(text.split())

    def remove_punc(text):
        # 구두점(.,!? 등) 모두 제거
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 모두 소문자로 변환
        return text.lower()

    # 소문자 변환 → 구두점 제거 → 관사 제거 → 공백 정리 순으로 적용
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """
    예측값과 정답을 normalize하여 완전히 일치하는지 여부 반환 (True/False)
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """
    예측(prediction)과 정답(ground_truth)을 정규화 후 토큰 단위로 나눔
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # 예측과 정답의 토큰별로 공통으로 등장하는 토큰의 개수를 셈
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    # 공통 토큰이 하나도 없으면 F1 점수는 0
    if num_same == 0:
        return 0

    # 정밀도(precision) 계산: 예측에서 맞춘 토큰 비율
    precision = 1.0 * num_same / len(prediction_tokens)
    # 재현율(recall) 계산: 정답에서 맞춘 토큰 비율
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # F1 점수 계산: 정밀도와 재현율의 조화평균
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    예측과 정답을 정규화한 후 완전히 일치하는지 확인
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    여러 개의 정답(ground_truths) 중에서, 예측(prediction)과 가장 잘 맞는(최고 점수) 정답에 대한 점수를 반환
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        # 각 정답에 대해 metric_fn(예: F1, exact match) 점수 계산
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    # 가장 높은 점수 반환
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    """
    F1, Exact Match, 전체 문제 수 초기화
    """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                # 전체 문제(질문) 수 증가
                total += 1
                # 해당 질문에 대한 예측이 없는 경우, 점수 0 처리 및 경고 출력
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                # 정답 리스트 추출 (여러 개일 수 있음)
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                # 예측값 추출
                prediction = predictions[qa['id']]
                # 정확 일치 점수(여러 정답 중 최고 점수) 누적
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                # F1 점수(여러 정답 중 최고 점수) 누적
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    # 전체 대비 정확 일치, F1 점수를 백분율로 변환
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    # 결과를 딕셔너리 형태로 반환
    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    # F1, Exact Match 점수 계산 및 출력
    print(json.dumps(evaluate(dataset, predictions)))
