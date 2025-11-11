# Standard library imports
from __future__ import absolute_import, division
import os
import sys
import io
import logging
import json
import copy
import ast
import warnings
import re
import time
import datetime
import collections
from collections import defaultdict, Counter
import math
import random
import pickle
import argparse
import uuid

# Other utilities
import nltk
# nltk.download('punkt_tab')
from dotenv import load_dotenv
from pathlib import Path
import importlib
import types
from abc import ABCMeta

# Data science and ML libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


# Hugging Face libraries
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
    pipeline, logging
)
from peft import LoraConfig, peft_model, PeftModel
from trl import SFTTrainer

# # TensorFlow and related
from tensorflow.python.framework.ops import enable_eager_execution
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


# # Custom modules (adjust paths as needed)
from ConfigLogger import config_logger
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths
from BERTRandomSampler import BERTRandomSampler

# # Suppress warnings
warnings.filterwarnings('ignore')