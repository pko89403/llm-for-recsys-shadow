from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

from datasets import Dataset, DatasetDict


class BaseProcessor(ABC):
    def __init__(
            self, 
            ds_articles: Dataset, 
            ds_embeddings:Optional[Union[None, DatasetDict]] = None
        ) -> None:
        """
        Args:
            ds_articles (Dataset): The articles.parquet file contains the detailed information of news articles.
            ds_embeddings (Optional[Union[None, DatasetDict]], optional): 
                To initiate the quick use of EB-NeRD, the dataset features embedding artifacts. 
                This includes the textual representation of the articles and the encoded thumbnail images. 
                The textual representations are based on the title, subtitle, and body. 
                We provide three representations, namely, the multilingual BERT, RoBERTa, and a proprietary contrastive-based model. 
                We also provide scripts to generate your own document embeddings using Hugging Face models (link coming soon!).
        """
        self.ds_articles = ds_articles
        self.ds_embeddings = ds_embeddings

    @abstractmethod
    def __call__(self, examples:Dict[str, Any]) -> Dict[str, Any]:
        """This call must be used with ds.map with batched=True."""
        pass