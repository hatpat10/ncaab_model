# config.py
from dotenv import load_dotenv
import os

load_dotenv()

ODDS_API_KEY = os.getenv('ODDS_API_KEY')
KENPOM_USER  = os.getenv('KENPOM_USER')
KENPOM_PASS  = os.getenv('KENPOM_PASS')
DB_PATH      = os.getenv('DB_PATH', 'data/ncaab.db')
SEASON       = os.getenv('SEASON', '2025-26')