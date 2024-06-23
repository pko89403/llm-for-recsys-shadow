import os
import streamlit as st
from loguru import logger

from macrec.systems import *
from macrec.utils import task2name, read_json

def scan_list(config: list) -> bool:
    pass