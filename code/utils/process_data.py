import sys
from common_imports import *

def clean_sentence(text: str) -> str:
    """
    문장 전처리 기능 함수
    축약형·소유격·하이픈 결합 단어 전처리
    """
    # 소유격, 축약형 's 분리
    text = re.sub(r"(\w+)'s", r"\1 's", text)
    # 축약형 단순 변환s
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
    """
    문장 리스트 전처리 함수
    """
    return [clean_sentence(s) for s in sent_list]

def open_jsonl(path):
    """
    JSONL 파일 로드 함수
    """
    data = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for example in f:
            data.append(json.loads(example))
    # data = data[:5] ################ debug size ################
    return data

def open_json(path):
    """
    JSON 파일 로드 함수
    """
    with io.open(path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)["data"]
        # input_data = input_data[:5]  ################ debug size ################
    return input_data

def update_qtype_prob(fpath, q_type, q_type_prob, q_ids):
    # JSON 파일 로드
    with io.open(fpath, 'r', encoding='utf-8') as f:
        input_data = json.load(f)["data"]
        input_data = input_data[:5] ################ debug size ################
    # 각 질문에 대해 분류 결과를 추가
    total_idx = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                # 예측된 질문 유형과 유형 확률값을 해당 질문에 추가
                if qa['id'] == q_ids[total_idx]:
                    qa['q_type'] = q_type[total_idx]
                    qa['q_type_prob'] = q_type_prob[total_idx]
                    total_idx += 1
                else:
                    # 질문 ID가 맞지 않으면 경고 출력
                    print('Can not match qid:', q_ids[total_idx])
                    
def update_file(fpath, q_type, q_type_prob, q_ids):
    """
    기존 파일에 질문 유형(q_type)과 질문 유형 확률(q_type_prob)을 추가하는 함수
    * format은 squad 형식으로 통합 *
    """

    # JSON 파일 처리(Ex.SQuAD)
    with io.open(fpath, 'r', encoding='utf-8') as f:
        input_data = json.load(f)["data"]
        input_data = input_data[:5] ################ debug size ################
    # 각 질문에 대해 분류 결과를 추가
    total_idx = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                # 예측된 질문 유형과 유형 확률값을 해당 질문에 추가
                if qa['id'] == q_ids[total_idx]:
                    qa['q_type'] = q_type[total_idx]
                    qa['q_type_prob'] = q_type_prob[total_idx]
                    total_idx += 1
                else:
                    # 질문 ID가 맞지 않으면 경고 출력
                    print('Can not match qid:', q_ids[total_idx])
        
    # 새로운 파일로 저장 (원본 파일명.json->_qtype_prob.jsonl)
    with open(fpath[:-5]+'_qtype_prob.jsonl', 'w') as f:
        for sample in input_data:
            f.write(json.dumps(sample)+'\n')
    f.close()
        

def parse_custom_data(raw_data):
    formatted_data = []
    for article in raw_data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                formatted_data.append({
                    'context': context.strip(),
                    'q_type': qa['q_type'],
                    'question': qa['question'].strip()
                })
    return formatted_data


def convert_jsonl_to_json(data):
    """
    jsonl 파일을 json 파일 형식으로 변환하는 함수
    """
    converted_data = []
    for item in data:
        # title: data의 id
        title = item.get('id', '')
        # context: data의 context
        context = item.get('context', '')
        # qas: data의 qas에서 필요한 필드만 추출
        qas = []
        for qa in item.get('qas', []):
            new_qa = {
                'answers': qa.get('answers', []),
                'question': qa.get('question', ''),
                'id': qa.get('qid', ''),  # qid를 id로
                'q_type': qa.get('q_type', None)
            }
            qas.append(new_qa)
        
        # jsonl 형식으로 구성
        new_entry = {
            'title': title,
            'paragraphs': [{
                'context': context,
                'qas': qas
            }]
        }
        converted_data.append(new_entry)
    return converted_data

def save_jsonl_to_json(input_path: str, output_path: str):
    """
    JSONL을 JSON 형식으로 변환하는 함수
    """
    data = load_jsonl(input_path)
    converted_data = convert_jsonl_to_json(data)
    json_data = {"data": converted_data}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)