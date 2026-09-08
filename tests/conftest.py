import os


# API 테스트도 production 인증 경계를 실제로 통과하도록 공통 test key를 설정한다.
os.environ["APP_ENV"] = "test"
os.environ["FACTORY_AGENT_API_KEY"] = "test-api-key"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
