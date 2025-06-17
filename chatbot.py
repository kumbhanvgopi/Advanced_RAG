# chatbot.py - Improved with better guardrails and security

import re
import os
import logging
import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from functools import wraps

# LangChain imports with fallbacks
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_groq import ChatGroq
except ImportError:
    raise ImportError("langchain-groq required: pip install langchain-groq")

try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    try:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
    except ImportError:
        from langchain.vectorstores import Qdrant as QdrantVectorStore

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Security and Cost Controls - Adjusted for document analysis
MAX_QUERY_LENGTH = 1000  # Increased for complex document questions
MAX_RESPONSE_LENGTH = 8000  # Increased for detailed document analysis
MAX_TOKENS_PER_SESSION = 50000
RATE_LIMIT_SECONDS = 1  # Minimum time between requests
BLOCKED_PATTERNS = [
    r'<script.*?>.*?</script>',
    r'javascript:',
    r'data:text/html',
    r'<iframe.*?>.*?</iframe>'
]

# Enhanced PII patterns
PII_PATTERNS = {
    'ssn': r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
    'credit_card': r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
    'phone': r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
}

# Content filtering
PROFANITY_LIST = [
    'badword1', 'badword2', 'inappropriate1', 'inappropriate2'
    # Add actual profanity list for production
]

SENSITIVE_TOPICS = [
    'password', 'api_key', 'secret', 'token', 'private_key',
    'social_security', 'bank_account', 'credit_card'
]

def rate_limit(min_interval=RATE_LIMIT_SECONDS):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class ContentFilter:
    """Enhanced content filtering with multiple security layers"""
    
    def __init__(self):
        self.pii_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}
        self.blocked_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BLOCKED_PATTERNS]
        self.profanity_pattern = re.compile('|'.join(PROFANITY_LIST), re.IGNORECASE)
        self.sensitive_pattern = re.compile('|'.join(SENSITIVE_TOPICS), re.IGNORECASE)
    
    def scan_for_pii(self, text: str) -> Tuple[bool, List[str], str]:
        """Scan text for PII and return cleaned version"""
        found_pii = []
        cleaned_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii.append(pii_type)
                cleaned_text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', cleaned_text)
        
        return bool(found_pii), found_pii, cleaned_text
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check for various content safety issues"""
        issues = []
        
        # Check for blocked patterns (XSS, injection, etc.)
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                issues.append("malicious_content")
                break
        
        # Check for profanity
        if self.profanity_pattern.search(text):
            issues.append("profanity")
        
        # Check for sensitive topics
        if self.sensitive_pattern.search(text):
            issues.append("sensitive_content")
        
        # Check for excessive length
        if len(text) > MAX_QUERY_LENGTH:
            issues.append("excessive_length")
        
        return bool(issues), issues
    
    def filter_response(self, response: str) -> Tuple[str, bool, List[str]]:
        """Filter and clean response content"""
        issues = []
        
        # Check for PII in response
        has_pii, pii_types, cleaned_response = self.scan_for_pii(response)
        if has_pii:
            issues.extend(pii_types)
        
        # Check response safety
        has_safety_issues, safety_issues = self.check_content_safety(cleaned_response)
        if has_safety_issues:
            issues.extend(safety_issues)
        
        # Truncate if too long
        if len(cleaned_response) > MAX_RESPONSE_LENGTH:
            cleaned_response = cleaned_response[:MAX_RESPONSE_LENGTH] + "... [Response truncated for safety]"
            issues.append("response_truncated")
        
        return cleaned_response, bool(issues), issues

class ChatbotManager:
    """Enhanced RAG chatbot with comprehensive security and guardrails"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = None,
        llm_model: str = "llama3-70b-8192",
        llm_temperature: float = 0.7,
        max_tokens: int = 4000,
        qdrant_url: str = None,
        collection_name: str = None,  # Make it required or session-based
        retrieval_k: int = 1,
        score_threshold: float = 0.3
    ):
        """Initialize with improved validation and security"""
        
        # Validate and sanitize inputs
        self.model_name = self._sanitize_string(model_name, "BAAI/bge-small-en")
        self.device = device if device in ["cpu", "cuda"] else "cpu"
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        self.llm_model = self._sanitize_string(llm_model, "llama3-70b-8192")
        self.llm_temperature = max(0.0, min(llm_temperature, 2.0))
        self.max_tokens = max(100, min(max_tokens, 8000))
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        
        # IMPORTANT: Use session-specific collection or default
        if collection_name:
            self.collection_name = self._sanitize_collection_name(collection_name)
        else:
            # Generate unique collection name if none provided
            unique_id = str(uuid.uuid4())[:8]
            self.collection_name = f"chat_collection_{unique_id}"
            
        self.api_key = os.getenv('QDRANT_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.retrieval_k = max(1, min(retrieval_k, 20))
        self.score_threshold = max(0.0, min(score_threshold, 1.0))
        
        # Initialize security components
        self.content_filter = ContentFilter()
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'start_time': time.time()
        }
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("ChatbotManager initialized with enhanced security")

    def _sanitize_string(self, value: str, default: str) -> str:
        """Sanitize string inputs"""
        if not value or not isinstance(value, str):
            return default
        return re.sub(r'[^\w\-/.]', '', value) or default

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name"""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:64]

    def _validate_config(self) -> None:
        """Enhanced configuration validation"""
        required_vars = {
            'QDRANT_URL': self.qdrant_url,
            'GROQ_API_KEY': self.groq_api_key
        }
        
        missing_vars = [k for k, v in required_vars.items() if not v]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Validate URL format
        if not (self.qdrant_url.startswith('http://') or self.qdrant_url.startswith('https://')):
            raise ValueError("QDRANT_URL must be a valid HTTP/HTTPS URL")

    def _initialize_components(self) -> None:
        """Initialize all components with error handling"""
        try:
            self._initialize_embeddings()
            self._initialize_llm()
            self._initialize_vector_store()
            self._initialize_retrieval_chain()
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    def _initialize_embeddings(self) -> None:
        """Initialize embeddings with fallback"""
        embedding_models = [
            self.model_name,
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        for model in embedding_models:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={"device": self.device},
                    encode_kwargs=self.encode_kwargs,
                )
                logger.info(f"Initialized embedding model: {model}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                continue
        
        raise RuntimeError("Failed to initialize any embedding model")

    def _initialize_llm(self) -> None:
        """Initialize LLM with fallback models"""
        llm_models = [
            self.llm_model,
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ]
        
        for model in llm_models:
            try:
                self.llm = ChatGroq(
                    model_name=model,
                    temperature=self.llm_temperature,
                    max_tokens=self.max_tokens,
                    groq_api_key=self.groq_api_key,
                    timeout=60,
                    max_retries=2
                )
                logger.info(f"Initialized LLM: {model}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                continue
        
        raise RuntimeError("Failed to initialize any LLM model")

    def _initialize_vector_store(self) -> None:
        """Initialize vector store with connection validation"""
        try:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                prefer_grpc=False,
                timeout=30
            )
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant: {len(collections.collections)} collections")
            
            # Initialize vector store
            init_methods = [
                self._init_vector_store_method1,
                self._init_vector_store_method2,
                self._init_vector_store_method3
            ]
            
            for method in init_methods:
                try:
                    method()
                    logger.info("Vector store initialized successfully")
                    return
                except Exception as e:
                    logger.warning(f"Vector store init method failed: {e}")
                    continue
            
            raise ConnectionError("All vector store initialization methods failed")
            
        except Exception as e:
            raise ConnectionError(f"Vector store initialization failed: {e}")

    def _init_vector_store_method1(self):
        """Vector store initialization method 1"""
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )

    def _init_vector_store_method2(self):
        """Vector store initialization method 2"""
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=self.qdrant_url,
            api_key=self.api_key
        )

    def _init_vector_store_method3(self):
        """Vector store initialization method 3"""
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings.embed_query
        )

    def _create_secure_prompt(self) -> PromptTemplate:
        """Create document-focused prompt template with security guidelines"""
        template = """You are a professional document analysis assistant.

CORE MISSION: Answer questions based ONLY on the provided document context.

SECURITY RULES:
- Use ONLY information from the CONTEXT below
- Never reveal sensitive data (passwords, API keys, personal information)
- Refuse questions unrelated to the document content
- Be factual and cite specific sections when possible

DOCUMENT ANALYSIS GUIDELINES:
- Clearly state if information is not in the document
- Reference page numbers and sections when available
- Maintain professional, helpful tone
- Provide concise, accurate answers

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

DOCUMENT-BASED RESPONSE:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _initialize_retrieval_chain(self) -> None:
        """Initialize retrieval chain with secure prompt"""
        try:
            self.prompt = self._create_secure_prompt()
            
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            self.chain_type_kwargs = {
                "prompt": self.prompt,
                "document_variable_name": "context"
            }
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs=self.chain_type_kwargs,
                verbose=False
            )
            
            logger.info("Secure retrieval chain initialized")
            
        except Exception as e:
            raise RuntimeError(f"Retrieval chain initialization failed: {e}")

    @rate_limit(min_interval=RATE_LIMIT_SECONDS)
    def get_response(
        self, 
        query: str, 
        enable_content_filter: bool = True,
        enable_pii_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Generate secure response with comprehensive filtering
        """
        self.session_stats['total_queries'] += 1
        
        # Basic validation
        if not query or not query.strip():
            return self._create_error_response("Empty query provided", "empty_query")
        
        # Check session limits
        if self.session_stats['total_tokens_used'] > MAX_TOKENS_PER_SESSION:
            return self._create_error_response("Session token limit exceeded", "token_limit")
        
        try:
            # Pre-process query security
            if enable_content_filter:
                has_issues, issues = self.content_filter.check_content_safety(query)
                if has_issues:
                    self.session_stats['flagged_queries'] += 1
                    return self._create_error_response(
                        "Query contains inappropriate content", 
                        f"content_filter: {', '.join(issues)}"
                    )
            
            # PII detection in query
            if enable_pii_detection:
                has_pii, pii_types, cleaned_query = self.content_filter.scan_for_pii(query)
                if has_pii:
                    logger.warning(f"PII detected in query: {pii_types}")
                    query = cleaned_query
            
            # Generate response
            logger.info(f"Processing query: {query[:100]}...")
            
            result = self.qa_chain.invoke({"query": query.strip()})
            
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Post-process response security
            if enable_content_filter or enable_pii_detection:
                filtered_answer, has_filter_issues, filter_issues = self.content_filter.filter_response(answer)
                if has_filter_issues:
                    self.session_stats['flagged_queries'] += 1
                    return {
                        "answer": filtered_answer,
                        "sources": self._process_sources(source_documents),
                        "is_flagged": True,
                        "flag_reason": f"Response filtered: {', '.join(filter_issues)}"
                    }
                answer = filtered_answer
            
            # Estimate token usage (rough approximation)
            estimated_tokens = len(answer.split()) + len(query.split())
            self.session_stats['total_tokens_used'] += estimated_tokens
            
            # Process sources
            sources = self._process_sources(source_documents)
            
            return {
                "answer": answer.strip() or "No response generated from the available context.",
                "sources": sources,
                "is_flagged": False,
                "flag_reason": None,
                "tokens_used": estimated_tokens,
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._create_error_response(f"System error: {str(e)}", "system_error")

    def _create_error_response(self, message: str, reason: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "answer": f"⚠️ {message}",
            "sources": [],
            "is_flagged": True,
            "flag_reason": reason,
            "tokens_used": 0,
            "processing_time": time.time()
        }

    def _process_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Process and secure source documents"""
        if not source_documents:
            return []
        
        sources = []
        seen_sources = set()
        
        for i, doc in enumerate(source_documents[:5]):  # Limit to 5 sources
            try:
                metadata = doc.metadata
                
                # Extract metadata safely
                file_name = str(metadata.get('file_name', 'Unknown Document'))[:100]
                page = metadata.get('page', 'N/A')
                chunk_id = metadata.get('chunk_id', 'N/A')
                word_count = metadata.get('word_count', 'N/A')
                
                # Create secure content preview
                content = doc.page_content[:300].replace('\n', ' ').strip()
                
                # Filter content for PII
                _, _, clean_content = self.content_filter.scan_for_pii(content)
                
                if len(doc.page_content) > 300:
                    clean_content += "..."
                
                # Deduplication
                source_key = f"{file_name}_{page}_{chunk_id}"
                if source_key in seen_sources:
                    continue
                
                source_info = {
                    "file_name": file_name,
                    "page": page,
                    "section": chunk_id + 1 if isinstance(chunk_id, int) else 'N/A',
                    "word_count": word_count,
                    "content_preview": clean_content,
                    "relevance_score": getattr(doc, 'score', 'N/A')
                }
                
                sources.append(source_info)
                seen_sources.add(source_key)
                
            except Exception as e:
                logger.warning(f"Error processing source {i}: {e}")
                continue
        
        return sources

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        current_time = time.time()
        session_duration = current_time - self.session_stats['start_time']
        
        return {
            **self.session_stats,
            "session_duration_seconds": round(session_duration, 2),
            "average_tokens_per_query": (
                self.session_stats['total_tokens_used'] / max(1, self.session_stats['total_queries'])
            ),
            "flagged_percentage": (
                (self.session_stats['flagged_queries'] / max(1, self.session_stats['total_queries'])) * 100
            ),
            "tokens_remaining": MAX_TOKENS_PER_SESSION - self.session_stats['total_tokens_used']
        }

    def update_retrieval_settings(self, k: int = None, threshold: float = None) -> None:
        """Update retrieval parameters with validation"""
        try:
            if k is not None:
                self.retrieval_k = max(1, min(k, 20))
            
            if threshold is not None:
                self.score_threshold = max(0.0, min(threshold, 1.0))
            
            # Reinitialize retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            # Update chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs=self.chain_type_kwargs,
                verbose=False
            )
            
            logger.info(f"Updated retrieval: k={self.retrieval_k}, threshold={self.score_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to update retrieval settings: {e}")

    def reset_session(self) -> None:
        """Reset session statistics"""
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'start_time': time.time()
        }
        logger.info("Session statistics reset")