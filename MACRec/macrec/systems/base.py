import pandas as pd
from streamlit as st
from abc import ABC, abstractmethod
from typing import Any, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents import Agent
from macrec.utils import is_correct, init_answer, read_json, read_prompts, get_avatar