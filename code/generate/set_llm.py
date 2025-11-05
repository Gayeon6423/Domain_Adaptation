"""
기본적인 작업들을 처리하는 데 있어 간편하게 사용할 수 있도록 제작한 Wrapper LLM 클래스
google Gemini와 OpenAI GPT 지원
"""
from typing import Dict, List, Optional, Type, Union, Any
from pydantic import BaseModel, Field
import json
from google import genai
from openai import OpenAI
import pathlib
import os



#########################
######## Utility ########

class TokenUsage(BaseModel):
    """토큰 사용량을 나타내는 Pydantic 모델
    NOTE: 각 모델에서 마지막 출력의 토큰 사용량을 저장하기 위한 용도로 사용
    """
    input_tokens: int = Field(default=0, description="입력 토큰 수 (cache 포함")
    thinking_tokens: int = Field(default=0, description="생각 토큰 수")
    output_tokens: int = Field(default=0, description="출력 토큰 수 (thinking 제외)")
    total_tokens: int = Field(default=0, description="총 토큰 수")
    cache_tokens: int = Field(default=0, description="캐시된 토큰 수")

    def set_usage(self, input_tokens: int = 0, thinking_tokens: int = 0, output_tokens: int = 0, total_tokens: Optional[int] = 0, cache_tokens: int = 0):
        """토큰 사용량을 설정하는 메서드
        
        Args:
            input_tokens(int): 입력 토큰 수
            output_tokens(int): 출력 토큰 수
            thinking_tokens(int): thinking/reasoning 토큰 수
            cache_tokens(int): 캐시된 토큰 수
            total_tokens(int): 총 토큰 수
        """
        self.input_tokens = input_tokens
        self.thinking_tokens = thinking_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.cache_tokens = cache_tokens
    
    def clear_usage(self):
        """토큰 사용량을 초기화하는 메서드
        """
        self.set_usage()
    
    def get_usage(self) -> Dict[str, Any]:
        """현재 토큰 사용량의 필드와 값을 딕셔너리 형태로 반환
        
        Returns:
            dict: 현재 토큰 사용량의 필드와 값들
        """
        return self.model_dump()



#########################
######## OpenAI #########
#########################


class OpenAIResponseConfig(BaseModel):
    """OpenAI API의 설정(top-p, temperature 등)을 정의하는 Pydantic 모델
    """
    temperature: float = Field(default=None)
    top_p: float = Field(default=None)
    reasoning: dict = Field(default={}, description="{'effort': 'low/medium/high'}")


class GPT:
    """OpenAI GPT 모델을 래핑하는 클래스
    
    이 클래스는 OpenAI API를 사용하여 텍스트 생성 및 대화 기능을 제공한다.
    
    Attributes:
        api_key (str): OpenAI API 인증을 위한 API 키
        model (str): 사용할 모델의 이름
        llm (OpenAI): OpenAI API와의 상호작용을 위한 객체
    
    Methods:
        completion(input_text: str) -> str:
            주어진 입력에 대한 모델의 응답을 생성한다.
        chat(user_input: str, history: list[dict] = None) -> str:
            모델과의 대화 응답을 처리한다.
        structured_output(input_text: str, output_schema: Type[BaseModel]) -> dict:
            Pydantic 모델을 사용하여 구조화된 출력을 생성한다.

    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """지정된 모델과 API 키로 openai의 객체를 초기화한다
        
        Args:
            api_key(str): 인증을 위한 API 키
            model(str): 사용할 특정 모델 이름 | Default: "gpt-4o-mini"
        """
        self.model = model
        self.api_key = api_key
        
        # 모델 초기화
        self.llm = OpenAI(api_key=api_key)

        # 설정 초기화 (기본값으로)
        self.set_config()
        self.last_token_usage = TokenUsage()  # 토큰 사용량 초기화
        self.last_token_usage.set_usage()  # 초기화 시 기본값으로 설정
    

    ## Config 관련 메서드 ##
    def set_config(self, **kwargs):
        """OpenAI 모델의 설정을 초기화한다
        
        Args:
            args(dict): OpenAIResponseConfig에 필요한 설정 값들
                - temperature, top_p, reasoning_effort, instruction 등
        """
        # OpenAIResponseConfig를 사용하여 설정 초기화
        self.config = OpenAIResponseConfig(**kwargs)
    
    def reset_config(self):
        """현재 설정을 초기화하고 기본값으로 되돌린다
        """
        self.set_config()
    
    def get_config_fields(self) -> Dict[str, Any]:
        """현재 설정의 필드와 값을 딕셔너리 형태로 반환한다
        
        Returns:
            dict: 현재 설정의 필드와 값들
        """
        return self.config.model_dump()
    

    def set_usage(self, usage):
        """토큰 사용량을 설정하는 메서드
        """
        self.last_token_usage.set_usage(
            input_tokens=usage.input_tokens,
            thinking_tokens=usage.output_tokens_details.reasoning_tokens,
            output_tokens=usage.output_tokens - usage.output_tokens_details.reasoning_tokens, # reasoning 제외
            total_tokens=usage.total_tokens,
            cache_tokens=usage.input_tokens_details.cached_tokens
        )
    

    def completion(self, input_text: str) -> str:
        """주어진 입력에 대한 모델의 응답을 생성한다
        
        Args:
            input_text(str): 모델에 제공할 입력 텍스트
            
        Returns:
            str: 모델의 응답 문자열
        """
        response = self.llm.responses.create(
            model=self.model,
            input=input_text,
            **self.config.model_dump()  # 설정 값들을 전달
        )
        output_text = response.output_text
        self.set_usage(response.usage)
        return output_text

    
    def chat(self, user_input: str, history: list[dict] = None) -> str:
        """모델과의 대화 응답을 처리한다
        
        Args:
            user_input(str): 사용자가 입력한 텍스트
            history(list[dict]): 대화의 이전 기록
                    각 항목은 {"role": "user", "content": "대화 내용"} 형식.
                    eg. [{"role": "user", "content": "안녕하세요!"}, {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]
            
        Returns:
            str: 모델의 응답 문자열
        
        NOTE: GPT의 response API의 지원상 문제로 파일 첨부 기능은 지원하지 않음 (현재 pdf만 지원)
        """
        # 대화 이력이 None인 경우 빈 리스트로 초기화
        if history is None:
            history = []
        
        # 대화 이력에 사용자의 입력 추가
        history.append({"role": "user", "content": user_input})

        # 대화 이력을 포함하여 모델에 요청
        response = self.llm.responses.create(
            model=self.model,
            input=history,
            **self.config.model_dump()  # 설정 값들을 전달
        )
        output_text = response.output_text
        self.set_usage(response.usage)
        self.response = response  # 디버깅용 resposne 객체 저장
        return output_text
    

    def structured_output(self, input_text: str, output_schema: BaseModel, history: list[dict] = None) -> BaseModel:
        """Pydantic 모델을 사용하여 구조화된 출력을 생성한다

        Args:
            input_text(str): 모델에 제공할 입력 텍스트
            output_schema(BaseModel): 출력 스키마를 정의하는 Pydantic 모델
                - Field에서는 description 사용 가능
        history(list[dict]): 대화의 이전 기록
                    각 항목은 {"role": "user", "content": "대화 내용"} 형식.
                    eg. [{"role": "user", "content": "안녕하세요!"}, {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

        Returns:
            지정된 pydantic 모델의 인스턴스

        NOTE: GPT의 response API의 지원상 문제로 파일 첨부 기능은 지원하지 않음 (현재 pdf만 지원)
        """
        # 대화 이력이 None인 경우 빈 리스트로 초기화
        if history is None:
            history = []
        
        # 대화 이력에 사용자의 입력 추가
        history.append({"role": "user", "content": input_text})

        # NOTE: 최신 버전으로 업데이트하여 beta에서 response.parse로 변경함
        # response = self.llm.responses.parse(
        #     model=self.model,
        #     input=history,
        #     response_format=output_schema,
        #     **self.config.model_dump()  # 설정 값들을 전달
        # )
        # parsed_output = response.parsed
        
        
        # 허용된 config 값만 추려서 전달
        allowed_keys = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"}
        config_kwargs = {k: v for k, v in self.config.model_dump().items() if k in allowed_keys}
        
        response = self.llm.beta.chat.completions.parse(
            model=self.model,
            messages=history,
            response_format=output_schema,
            **config_kwargs
        )

        parsed_output = response.choices[0].message.parsed
        
        # self.set_usage(response.usage)
        self.usage = response.usage
        self.response = response  # 디버깅용 resposne 객체 저장
        return parsed_output
    

#########################
######## Gemini #########
#########################


class Gemini:
    """
    Google Gemini 모델을 래핑하는 클래스
    이 클래스는 Google GenAI API를 사용하여 텍스트 생성 및 대화 기능을 제공한다.

    Attributes:
        api_key (str): Google GenAI API 인증을 위한 API 키
        model (str): 사용할 모델의 이름
        llm (genai.Client): Google GenAI API와의 상호작용을 위한 객체

    Methods:
        completion(input_text: str) -> str:
            주어진 입력에 대한 모델의 응답을 생성한다.
        chat(user_input: str, history: list[dict] = None) -> str:
            모델과의 대화 응답을 처리한다.
        structured_output(input_text: str, output_schema: Type[BaseModel]) -> dict:
            Pydantic 모델을 사용하여 구조화된 출력을 생성한다.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """지정된 모델과 API 키로 google genai의 객체를 초기화한다
        
        Args:
            api_key(str): 인증을 위한 API 키
            model(str): 사용할 특정 모델 이름 | Default: "gemini-2.0-flash"
        """
        self.model = model
        self.api_key = api_key
        
        # 모델 초기화
        self.llm = genai.Client(api_key=api_key)

        # 설정 초기화 (기본값으로)
        self.set_config()
        self.last_token_usage = TokenUsage()  # 토큰 사용량 초기화


    def set_config(self, **kwargs):
        """Gemini 모델의 설정을 초기화한다
        
        Args:
            args(dict): GeminiConfig에 필요한 설정 값들
                - temperature, top_p, top_k, max_output_tokens, system_instruction 등
        

        참고: https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig
        Genai API의 GenerateContentConfig 설정값들 목록
        """
        # thinking config 설정 (기본값은 False로)
        thinking_config = genai.types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
        # kwargs에서 필요한 설정 값들을 추출하여 thinking_config에 설정 (존재하는 경우에만)
        if "include_thoughts" in kwargs:
            thinking_config.include_thoughts = kwargs.pop("include_thoughts", False)
        if "thinking_budget" in kwargs:
            thinking_config.thinking_budget = kwargs.pop("thinking_budget", 0)
        
        # config 설정
        self.config = genai.types.GenerateContentConfig(**kwargs, thinking_config=thinking_config)
    
    def reset_config(self):
        """현재 설정을 초기화하고 기본값으로 되돌린다
        """
        self.set_config()
    
    def get_config_fields(self) -> Dict[str, Any]:
        """현재 설정의 필드와 값을 딕셔너리 형태로 반환한다
        
        Returns:
            dict: 현재 설정의 필드와 값들
        """
        return self.config.model_dump()
    
    def set_usage(self, usage_metadata: genai.types.GenerateContentResponseUsageMetadata):
        """토큰 사용량을 설정하는 메서드
        
        Args:
            usage_metadata(genai.types.GenerateContentResponseUsageMetadata): 토큰 사용량 메타데이터
        """
        self.last_token_usage.set_usage(
            input_tokens=usage_metadata.prompt_token_count,
            thinking_tokens=usage_metadata.thoughts_token_count,
            output_tokens=usage_metadata.candidates_token_count,
            total_tokens=usage_metadata.total_token_count,
            cache_tokens=usage_metadata.cached_content_token_count
        )
    

    def completion(self, input_text: str) -> str:
        """주어진 입력에 대한 모델의 응답을 생성한다
        
        Args:
            input_text(str): 모델에 제공할 입력 텍스트
            
        Returns:
            str: 모델의 응답 문자열
        """
        response = self.llm.models.generate_content(
            model=self.model,
            contents=input_text,
            config=self.config,
        )
        output_text = response.text
        self.set_usage(response.usage_metadata)
        return output_text
    

    def chat(self, user_input: str, history: list[dict] = None, files: list[str] = []) -> str:
        """모델과의 대화 응답을 처리한다
        
        Args:
            user_input(str): 사용자가 입력한 텍스트
            history(list[dict]): 대화의 이전 기록
                    각 항목은 {"role": "user", "content": "대화 내용"} 형식.
                    eg. [{"role": "user", "content": "안녕하세요!"}, {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]
            files(list[str]): 첨부할 파일들의 path 목록 (선택)
            
        Returns:
            str: 모델의 응답 문자열
        """
        # 대화 이력이 None인 경우 빈 리스트로 초기화
        if history is None:
            history = []
        
        # 대화 이력에 사용자의 입력 추가
        history.append({"role": "user", "content": user_input})

        # 임시로 이런 방식으로 채팅 기록을 문자열로 주는 방식으로 변환
        history_list = [f"{c['role']}: {c['content']}" for c in history]

        # 첨부 파일이 있는 경우 파일을 추가
        for file in files:
            file_path = pathlib.Path(file)
            file_uploaded = self.llm.files.upload(file=file_path, config={"display_name": os.path.basename(file_path)})
            history_list.append(file_uploaded)

        history_list.append("assistant: ")

        response = self.llm.models.generate_content(
            model=self.model,
            contents=history_list,
            config=self.config,
        )
        output_text = response.text
        self.set_usage(response.usage_metadata)
        self.response = response  # 디버깅용 resposne 객체 저장
        return output_text
    

    def structured_output(self, input_text: str, output_schema: BaseModel, history: list[dict] = None, files: list[str] = []) -> dict:
        """Pydantic 모델을 사용하여 구조화된 출력을 생성한다

        Args:
            input_text(str): 모델에 제공할 입력 텍스트
            output_schema(BaseModel): 출력 스키마를 정의하는 Pydantic 모델
                - Field에서는 description 사용 가능
            history(list[dict]): 대화의 이전 기록
                각 항목은 {"role": "user", "content": "대화 내용"} 형식.
                eg. [{"role": "user", "content": "안녕하세요!"}, {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

        Returns:
            지정된 pydantic 모델의 인스턴스

        """
        # 대화 이력이 None인 경우 빈 리스트로 초기화
        if history is None:
            history = []
        # 대화 이력에 사용자의 입력 추가
        history.append({"role": "user", "content": input_text})

        # 임시로 이런 방식으로 채팅 기록을 문자열로 주는 방식으로 변임
        history_list = [f"{c['role']}: {c['content']}" for c in history]

        # 첨부 파일이 있는 경우 파일을 추가
        for file in files:
            file_path = pathlib.Path(file)
            file_uploaded = self.llm.files.upload(file=file_path, config={"display_name": os.path.basename(file_path)})
            history_list.append(file_uploaded)

        history_list.append("assistant: ")

        # Structured output을 위해 GeminiConfig를 복사하고 response_mime_type과 response_schema 설정
        config_copy = self.config.model_copy(deep=True)
        config_copy.response_mime_type = 'application/json'
        config_copy.response_schema = output_schema

        response = self.llm.models.generate_content(
            model=self.model,
            contents=history_list,
            config=config_copy,
        )
        parsed_output = response.parsed
        self.set_usage(response.usage_metadata)
        self.response = response  # 디버깅용 resposne 객체 저장
        return parsed_output



#########################
########## LLM ##########
#########################

class LLM:
    """
    LLM 래퍼 클래스
    - Google Gemini 또는 OpenAI GPT 모델을 래핑한 클래스를 사용하여 LLM 기능을 제공

    Attributes:
        provider (str): 사용할 LLM 제공자('google', 'openai')
        model (str): 사용할 모델의 이름
        api_key (str): 인증을 위한 API 키
        llm (object): 선택된 LLM 제공자의 객체
    
    Methods:
        completion(input_text: str) -> str:
            주어진 입력에 대한 모델의 응답을 생성한다.
        chat(user_input: str, history: list[dict] = None) -> str:
            모델과의 대화 응답을 처리한다.
        structured_output(input_text: str, output_schema: Type[BaseModel]) -> dict:
            Pydantic 모델을 사용하여 구조화된 출력을 생성한다.
        choose_number(input_text: str, options: dict[int, str]) -> int:
            번호가 매겨진 선택지 목록에서 하나의 옵션을 선택한다.
    """


    def __init__(self, provider: str, model: str, api_key: str):
        """지정된 제공자와 모델로 LLM 래퍼를 초기화한다.
        
        Args:
            provider: 사용할 LLM 제공자('google', 'openai')
            model: 사용할 특정 모델 이름
            api_key: 인증을 위한 API 키
            
        Raises:
            ValueError: 지원되지 않는 제공자가 지정된 경우
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        
        if self.provider == "google":
            self.llm = Gemini(model=model, api_key=api_key)
        elif self.provider == "openai":
            self.llm = GPT(model=model, api_key=api_key)
        else:
            raise ValueError(f"지원되지 않는 제공자: {self.provider}")
        

        # llm 객체의 기본적인 메서드들을 래핑
        self.completion = self.llm.completion
        self.chat = self.llm.chat
        self.structured_output = self.llm.structured_output
        self.set_config = self.llm.set_config
        self.reset_config = self.llm.reset_config
        self.get_config_fields = self.llm.get_config_fields
        self.last_token_usage = self.llm.last_token_usage  # 토큰 사용량 객체
    

    def choose_number(self, 
                        input_text: str, 
                        options: dict[int, str],
                        str_number: bool = False) -> int:
        """번호가 매겨진 선택지 목록에서 하나의 옵션을 선택합니다.
        TODO: choose_numbers 도 추가하기
        
        Args:
            input_text: 선택할 내용을 설명하는 입력 텍스트/질문
            options: 옵션 번호와 설명을 매핑하는 딕셔너리
                예시: {1: "첫 번째 옵션", 2: "두 번째 옵션"}
                
        Returns:
            선택된 옵션 번호(int)
            
        Raises:
            ValueError: 모델이 유효한 옵션 번호를 반환하지 못한 경우.
        """
        # 번호가 매겨진 옵션으로 형식화된 프롬프트 생성
        formatted_options = "\n".join([f"{num}. {text}" for num, text in options.items()])
        full_prompt = f"{input_text}\n\n선택지:\n{formatted_options}\n\n번호만 선택하여 가장 적합한 옵션을 고르세요."
        
        # 출력 스키마 정의
        if str_number:
            class OptionChoice(BaseModel):
                choice: str = Field(description="선택한 옵션 번호 (문자열 형식)")
        else:
            class OptionChoice(BaseModel):
                choice: int = Field(description="선택한 옵션 번호")
        
        # 구조화된 응답 받기
        response = self.structured_output(input_text=full_prompt, output_schema=OptionChoice)
        
        # 선택 검증
        if response.choice not in options:
            raise ValueError(f"모델이 유효하지 않은 옵션 번호를 반환했습니다: {response.choice}")
        
        return response.choice