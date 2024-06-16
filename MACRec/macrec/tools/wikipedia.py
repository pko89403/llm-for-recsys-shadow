from langchain_core.documents import Document
from langchain_community.retrievers.wikipedia import WikipediaRetriever

from macrec.tools.base import RetrieverTool

class Wikipedia(RetrieverTool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.top_k: int = self.config.get("top_k", 3)
        max_doc_length: int = self.config.get("max_doc_length", 4000)
        language: str = self.config.get("language", "en")
        self.retriever = WikipediaRetriever(top_k_results=self.top_k, doc_content_chars_max=max_doc_length, lang=language)
        self.cache = {}
        
    def reset(self) -> None:
        self.cache = {}
        
    def _format_documents(self, documents: list[Document]) -> str:
        titles = []
        summary = []
        for document in documents:
            assert "title" in document.metadata
            title = document.metadata["title"]
            if title not in self.cache:
                self.cache[title] = {
                    "document": document,
                    "lookup_index": {},
                }
            titles.append(title)
            summary_content = document.metadata["summary"] if "summary" in document.metadata else document.page_content.split("\n\n")[0]
            if len(summary_content.split()) > 20:
                summary_content = " ".join(summary_content.split()[:20]) + "..."
            summary.append(summary_content)
        return ", ".join([f"{title} ({summary})" for title, summary in zip(titles, summary)])
    
    def search(self, query: str) -> str:
        try:
            results = self.retriever.get_relevant_documents(query=query)
            if len(results) == 0:
                return f"No documents found for query {query}."
            else:
                return f"Found {len(results)} documents. Their titles and summaries are (with the format title (summary)): {self._format_documents(results)}"
        except Exception as e:
            return f"Error occurred during search: {str(e)}"

    def lookup(self, title: str, term: str) -> str:
        """
        문서에서 용어를 찾아 해당 결과를 반환합니다.

        매개변수:
            title (str): 검색할 문서의 제목입니다.
            term (str): 문서에서 찾을 용어입니다.

        반환값:
            str: 문서에서 찾은 용어를 포함한 결과입니다.

        예외:
            없음

        """
        if title not in self.cache:
            return "No title found in search results."
        document: Document = self.cache[title]["document"]
        if term not in self.cache[title]["lookup_index"]:
            self.cache[title]["lookup_index"][term] = 0
        else:
            self.cache[title]["lookup_index"][term] += 1
        lookups = [p for p in document.page_content.split("\n\n") if term.lower() in p.lower()]
        if len(lookups) == 0:
            return f"No results for term {term} in document {title}."
        elif self.cache[title]["lookup_index"][term] >= len(lookups):
            return f"No more results for term {term} in document {title}."
        else:
            result_prefix = f"(Result {self.cache[title]['lookup_index'][term] + 1} / {len(lookups)})"
            return f"{result_prefix} {lookups[self.cache[title]['lookup_index'][term]]}"