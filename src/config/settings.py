import os
from dotenv import load_dotenv

class Settings:
    def __init__(self, env_file="yf.env"):
        load_dotenv(env_file)
        self.YF_API_URL = os.getenv("YF_API_URL")
        self.DB_URL = os.getenv("DB_URL")

settings = Settings()