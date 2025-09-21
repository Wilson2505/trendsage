import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BEARER = os.getenv("TWITTER_BEARER_TOKEN", "").strip()
    API_KEY = os.getenv("TWITTER_API_KEY", "").strip()
    API_SECRET = os.getenv("TWITTER_API_SECRET", "").strip()
    ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "").strip()
    ACCESS_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "").strip()

    def validate(self):
        if not self.BEARER:
            raise ValueError("TWITTER_BEARER_TOKEN is missing in .env")

settings = Settings()
