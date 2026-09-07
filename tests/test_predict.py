# tests/test_predict.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import predict

def test_relaxed_extract_finds_json():
    assert predict.relaxed_extract('input: {"state": "def"} trailing') == '{"state": "def"}'

def test_relaxed_extract_none_if_missing():
    assert predict.relaxed_extract('no json here') is None
