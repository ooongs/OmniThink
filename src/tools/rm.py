import logging
import os
from typing import List, Dict, Any, Optional, Union, Callable
import dspy
import requests
import re
import uuid
import json
import random
import time
import threading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.WebPageHelper import WebPageHelper


def clean_text(res):
    pattern = r'\[.*?\]\(.*?\)'
    result = re.sub(pattern, '', res)
    url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    result = re.sub(url_pattern, '', result)
    result = re.sub(r"\n\n+", "\n", result)
    return result


LEAKAGE_DOMAIN_PATTERNS = {
    "cnx.org",
    "openstax.org",
    "pressbooks.pub",
    "pb.unizin.org",
    "libretexts.org",
    "courses.lumenlearning.com",
    "lumenlearning.com",
    "open.lib.umn.edu",
    "open.oregonstate.education",
    "open.maricopa.edu",
    "openwa.pressbooks.pub",
    "openfl.pressbooks.pub",
    "openoregon.pressbooks.pub",
    "louis.pressbooks.pub",
    "lmu.pressbooks.pub",
    "minnstate.pressbooks.pub",
    "ecampusontario.pressbooks.pub",
    "pressbooks.atlanticoer-relatlantique.ca",
    "erau.edu",
    "eaglepubs.erau.edu",
}

LEAKAGE_HOST_KEYWORDS = {
    "pressbooks",
}


def _source_url(source):
    if isinstance(source, dict):
        return (
            source.get("url")
            or source.get("link")
            or source.get("host")
            or ""
        )
    return source or ""


def _source_text(source):
    if isinstance(source, dict):
        parts = [
            source.get("url", ""),
            source.get("link", ""),
            source.get("title", ""),
            source.get("description", ""),
            source.get("snippet", ""),
        ]
        snippets = source.get("snippets") or []
        if isinstance(snippets, str):
            parts.append(snippets)
        else:
            parts.extend(snippets)
        return " ".join(str(part or "") for part in parts).lower()
    return str(source or "").lower()


def _url_path(source):
    value = (_source_url(source) or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    try:
        from urllib.parse import urlparse

        return urlparse(value).path.lower()
    except Exception:
        return value.lower()


def _normalize_host(url_or_host):
    value = (_source_url(url_or_host) or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"

    try:
        from urllib.parse import urlparse

        host = urlparse(value).netloc or urlparse(value).path
    except Exception:
        host = value

    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_edu_host(host):
    return host == "edu" or host.endswith(".edu")


def _is_edu_pdf_source(source, host):
    if not _is_edu_host(host):
        return False
    path = _url_path(source)
    text = _source_text(source)
    return path.endswith(".pdf") or ".pdf/" in path or "[pdf]" in text


def is_leakage_url(source):
    host = _normalize_host(source)
    if not host:
        return False

    if _is_edu_pdf_source(source, host):
        return True

    for domain in LEAKAGE_DOMAIN_PATTERNS:
        normalized_domain = _normalize_host(domain)
        if host == normalized_domain or host.endswith(f".{normalized_domain}"):
            return True

    return any(keyword in host for keyword in LEAKAGE_HOST_KEYWORDS)


def is_allowed_source(url):
    return not is_leakage_url(url)


class SerperSearch(dspy.Retrieve):
    def __init__(self, serper_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 query_budget: Optional[int] = None, source_budget: Optional[int] = None,
                 gl: str = "us", hl: str = "en", endpoint: str = "https://google.serper.dev/search",
                 is_valid_query: Callable = None, blocked_sample_limit: int = 20,
                 **kwargs):
        super().__init__(k=k)
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY") or os.environ.get("SEARCHKEY")
        if not self.serper_api_key:
            raise RuntimeError("Set SERPER_API_KEY or SEARCHKEY before using SerperSearch.")
        self.k = k
        self.endpoint = endpoint
        self.gl = gl
        self.hl = hl
        self.params = kwargs
        self.query_budget = query_budget
        self.source_budget = source_budget
        self.usage = 0
        self.skipped_queries = 0
        self.accepted_urls = set()
        self.blocked_sample_limit = blocked_sample_limit
        self.leakage_query_blocked_count = 0
        self.leakage_source_blocked_count = 0
        self.blocked_query_samples = []
        self.blocked_source_samples = []
        self._budget_lock = threading.Lock()
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.custom_is_valid_source = is_valid_source or (lambda _u: True)
        self.custom_is_valid_query = is_valid_query or (lambda _q: True)

    def _custom_source_allowed(self, source: Dict[str, Any]) -> bool:
        try:
            return bool(self.custom_is_valid_source(source))
        except Exception:
            try:
                return bool(self.custom_is_valid_source(_source_url(source)))
            except Exception:
                return False

    def _record_blocked_query(self, query: str, reason: str = "leakage"):
        with self._budget_lock:
            self.leakage_query_blocked_count += 1
            if len(self.blocked_query_samples) < self.blocked_sample_limit:
                self.blocked_query_samples.append({"query": query, "reason": reason})

    def _record_blocked_source(self, source: Dict[str, Any], reason: str = "leakage"):
        with self._budget_lock:
            self.leakage_source_blocked_count += 1
            if len(self.blocked_source_samples) < self.blocked_sample_limit:
                self.blocked_source_samples.append({
                    "url": _source_url(source),
                    "title": source.get("title", "") if isinstance(source, dict) else "",
                    "description": source.get("description", "") if isinstance(source, dict) else "",
                    "reason": reason,
                })

    def _query_allowed(self, query: str) -> bool:
        try:
            allowed = bool(self.custom_is_valid_query(query))
        except Exception:
            allowed = False
        if not allowed:
            self._record_blocked_query(query)
        return allowed

    def _source_allowed(self, source: Dict[str, Any]) -> bool:
        if not is_allowed_source(source):
            self._record_blocked_source(source, reason="static_leakage_filter")
            return False
        if not self._custom_source_allowed(source):
            self._record_blocked_source(source, reason="chapter_leakage_filter")
            return False
        return True

    def set_budget(self, query_budget: Optional[int] = None, source_budget: Optional[int] = None):
        with self._budget_lock:
            self.query_budget = query_budget
            self.source_budget = source_budget
            self.usage = 0
            self.skipped_queries = 0
            self.accepted_urls = set()
            self.leakage_query_blocked_count = 0
            self.leakage_source_blocked_count = 0
            self.blocked_query_samples = []
            self.blocked_source_samples = []

    def _reserve_queries(self, queries: List[str]) -> List[str]:
        with self._budget_lock:
            if self.query_budget is None:
                allowed = len(queries)
            else:
                allowed = max(self.query_budget - self.usage, 0)
            selected = queries[:allowed]
            self.usage += len(selected)
            self.skipped_queries += max(len(queries) - len(selected), 0)
        return selected

    def _can_accept_url(self, url: str) -> bool:
        if url in self.accepted_urls:
            return True
        if self.source_budget is not None and len(self.accepted_urls) >= self.source_budget:
            return False
        return True

    def _mark_accepted_url(self, url: str):
        if url not in self.accepted_urls:
            self.accepted_urls.add(url)

    def get_budget_report(self) -> Dict[str, Optional[int]]:
        return {
            "query_budget": self.query_budget,
            "queries_used": self.usage,
            "queries_skipped": self.skipped_queries,
            "source_budget": self.source_budget,
            "sources_accepted": len(self.accepted_urls),
            "leakage_query_blocked_count": self.leakage_query_blocked_count,
            "leakage_source_blocked_count": self.leakage_source_blocked_count,
        }

    def get_usage_and_reset(self):
        usage = self.usage
        accepted_sources = len(self.accepted_urls)
        blocked_queries = self.leakage_query_blocked_count
        blocked_sources = self.leakage_source_blocked_count
        self.usage = 0
        self.skipped_queries = 0
        self.accepted_urls = set()
        self.leakage_query_blocked_count = 0
        self.leakage_source_blocked_count = 0
        self.blocked_query_samples = []
        self.blocked_source_samples = []
        return {
            'SerperSearch': usage,
            'SerperAcceptedSources': accepted_sources,
            'SerperLeakageQueriesBlocked': blocked_queries,
            'SerperLeakageSourcesBlocked': blocked_sources,
        }

    def get_leakage_report(self) -> Dict[str, Any]:
        return {
            "leakage_query_blocked_count": self.leakage_query_blocked_count,
            "leakage_source_blocked_count": self.leakage_source_blocked_count,
            "blocked_query_samples": list(self.blocked_query_samples),
            "blocked_source_samples": list(self.blocked_source_samples),
            "accepted_urls": sorted(self.accepted_urls),
        }

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        queries = [query for query in queries if query]
        queries = [query for query in queries if self._query_allowed(query)]
        queries = self._reserve_queries(queries)
        if not queries:
            return []

        url_to_results: Dict[str, Dict[str, Any]] = {}
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        for query in queries:
            try:
                payload = {
                    "q": query,
                    "num": self.k,
                    "gl": self.gl,
                    "hl": self.hl,
                    **self.params,
                }
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                results = response.json()
                for result in results.get("organic", []):
                    url = result.get("link")
                    source = {
                        "url": url,
                        "title": result.get("title", ""),
                        "description": result.get("snippet", ""),
                    }
                    if not url or url in exclude_urls or not self._source_allowed(source):
                        continue
                    if not self._can_accept_url(url):
                        continue
                    url_to_results[url] = source
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            if not self._source_allowed(r):
                continue
            with self._budget_lock:
                if not self._can_accept_url(url):
                    continue
                self._mark_accepted_url(url)
            collected_results.append(r)
        return collected_results

class GoogleSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):

        super().__init__(k=k)
        # key = os.environ.get('SEARCHKEY', 'default_value')
        key = "19WaNVGhRjcjYcKuOV96w"
        self.header = {
            "Content-Type": "application/json",
            "Accept-Encoding": "utf-8",
            "Authorization": f"Bearer lm-/{key}== ",
        }

        self.template = {
            "rid": str(uuid.uuid4()),
            "scene": "dolphin_search_bing_nlp",
            "uq": "",
            "debug": True,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQuery": False,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "qpTermsWeight": False,
                "pluginServiceConfig": {"qp": "mvp_search_qp_qwen"},  # v3 rewrite
            },
            "headers": {"__d_head_qto": 5000},
        }
        
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        MAX_RETRIES = 30
        for query in queries:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    self.template["uq"] = query

                    response = requests.post(
                        "https://nlp-cn-beijing.aliyuncs.com/gw/v1/api/msearch-sp/qwen-search",
                        data=json.dumps(self.template),
                        headers=self.header,
                    )
                    response = json.loads(response.text)
                    search_results = response['data']['docs']
                    for result in search_results:
                        url_to_results[result['url']] = {
                            'url': result['url'],
                            'title': result['title'],
                            'description': result.get('snippet', '')
                        }
                except Exception as e:
                    retries += 1
                    RETRY_DELAY = random.uniform(0, 10)
                    logging.error(f"Error occurred when searching query {query}: {e}")
                    if retries < MAX_RETRIES:
                        logging.info(f"Retrying ({retries}/{MAX_RETRIES}) after {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logging.error(f"Max retries reached for query {query}. Skipping.")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)

        print(f'lengt of collected_results :{len(collected_results)}')
        return collected_results
    

class BingSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("SEARCH_ALI_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_api_key or set environment variable SEARCH_ALI_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["SEARCH_ALI_API_KEY"]
        self.endpoint = "https://idealab.alibaba-inc.com/api/v1/search/search"
        self.count = k
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        payload_template = {
            "query": "pleaceholder",
            "num": self.count,
            "extendParams": {
                "country": "US",
                "locale": "en-US",
                "location": "United States",
                "page": 2
            },
            "platformInput": {
                "model": "google-search",
                "instanceVersion": "S1"
            }
        }
        header = {"X-AK": self.bing_api_key, "Content-Type": "application/json"}

        for query in queries:
            try:
                payload_template["query"] = query
                response = requests.post(
                    self.endpoint,
                    headers=header,
                    json=payload_template,
                ).json()
                search_results = response['data']['originalOutput']['webPages']['value']

                for result in search_results:
                    url_to_results[result['url']] = {
                        'url': result['url'],
                        'title': result['name'],
                        'description': result.get('snippet', '')
                    }
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key , "Content-Type": "application/json" }

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint,
                    headers=headers,
                    params={**self.params, 'q': query}
                ).json()

                for d in results['webPages']['value']:
                    if self.is_valid_source(d['url']) and d['url'] not in exclude_urls:
                        url_to_results[d['url']] = {'url': d['url'], 'title': d['name'], 'description': d['snippet']}
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


class OfflineRAGFlow:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 800,
                 overlap: int = 120,
                 k: int = 5):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k

        self.docs: Dict[str, Dict[str, Any]] = {}
        self.keys: List[tuple] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    # -----------------------------
    # Helper functions
    # -----------------------------
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove markdown links
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text)  # remove urls
        text = re.sub(r"\n\n+", "\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        text = self._clean_text(text)
        n = len(text)
        if n == 0:
            return []
        chunks = []
        i = 0
        while i < n:
            j = min(n, i + self.chunk_size)
            slice_ = text[i:j]
            k = slice_.rfind("。")
            if k == -1:
                k = slice_.rfind(".")
            if k != -1 and (i + k + 1 - i) > self.chunk_size * 0.6:
                j = i + k + 1
            chunks.append(text[i:j].strip())
            if j >= n:
                break
            i = max(j - self.overlap, i + 1)
        return [c for c in chunks if c]

    def _encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32
        )

    # -----------------------------
    # Document operations
    # -----------------------------
    def ingest(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        doc_id = meta.get("doc_id") if meta else str(uuid.uuid4())
        chunks = self._chunk_text(text)
        self.docs[doc_id] = {"text": text, "meta": meta or {}, "chunks": chunks}
        self._update_index(doc_id, chunks)
        return doc_id

    def _update_index(self, doc_id: str, chunks: List[str]):
        if not chunks:
            return
        embs = self._encode(chunks)
        if self.index is None:
            d = embs.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(embs)
            self.embeddings = embs
            self.keys = [(doc_id, i) for i in range(len(chunks))]
        else:
            self.index.add(embs)
            self.embeddings = np.vstack([self.embeddings, embs]) if self.embeddings is not None else embs
            base = len([k for k in self.keys if k[0] == doc_id])
            self.keys.extend([(doc_id, base + i) for i in range(len(chunks))])

    # -----------------------------
    # Retrieval
    # -----------------------------
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.keys:
            return []
        k = k or self.k
        q = self._encode([query])
        D, I = self.index.search(q, min(k, len(self.keys)))
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            doc_id, chunk_idx = self.keys[idx]
            doc = self.docs.get(doc_id)
            if not doc:
                continue
            results.append({
                "doc_id": doc_id,
                "title": doc["meta"].get("title", doc_id),
                "snippet": doc["chunks"][chunk_idx],
                "score": float(score)
            })
        return results

    def qa(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        hits = self.search(query, k=k)
        context = "\n\n".join([f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits)])
        answer = (
            f"Based on the retrieved context, here is a possible answer.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        return {"answer": answer, "citations": hits}


class LocalSearch(dspy.Retrieve):
    def __init__(self,
                 search: OfflineRAGFlow,
                 k: int = 3,
                 is_valid_source: Optional[Callable[[str], bool]] = None,
                 **kwargs):
        super().__init__(k=k)
        self.search = search
        self.k = k
        self.is_valid_source = is_valid_source or (lambda _u: True)
        self.usage = 0

    def get_usage_and_reset(self) -> Dict[str, int]:
        u = self.usage
        self.usage = 0
        return {"LocalSearch": u}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Dict[str, Any]]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        self.usage += len(queries)
        url_to_results: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            hits = self.search.search(q, k=self.k)
            for h in hits:
                doc_id = h["doc_id"]
                url = f"doc://{doc_id}"
                if url in exclude_urls or not self.is_valid_source(url):
                    continue
                doc = self.search.docs.get(doc_id, {})
                description = doc.get("text", "")
                if len(description) > 220:
                    description = description[:220] + "..."
                url_to_results[url] = {
                    "url": url,
                    "title": h.get("title", doc_id),
                    "description": description,
                    "snippets": [h.get("snippet", "")],
                }
        return list(url_to_results.values())
