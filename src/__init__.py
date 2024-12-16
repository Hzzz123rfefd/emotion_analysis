from src.model import ModelPretrainForEmotionAnalysis
from src.dataset import DatasetForEmotionAnalysis


datasets = {
   "emotion_analysis": DatasetForEmotionAnalysis
}

models = {
    "emotion_analysis_base_bert": ModelPretrainForEmotionAnalysis,
}