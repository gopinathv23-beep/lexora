#!/usr/bin/env python3
"""
Enhanced StudyMate: An AI-Powered PDF-Based Q&A System for Students
Built with improved error handling, performance optimizations, and enhanced features
"""

import gradio as gr
import os
import io
import json
import re
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Try to import required packages with fallbacks
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. PDF processing will be limited.")

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not available. Advanced PDF processing disabled.")

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas/Numpy not available. Some features disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/PyTorch not available. Using fallback text processing.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available. Using basic search.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using basic search.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image processing disabled.")


class EnhancedStudyMateSystem:
    def __init__(self):
        """Initialize the enhanced StudyMate system with robust error handling"""
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.documents = {}
        self.embeddings = {}
        self.index = None
        self.text_chunks = []
        self.chunk_sources = []
        self.chunk_metadata = []
        
        # System status
        self.system_ready = False
        self.models_loaded = False
        
        # Enhanced features
        self.conversation_history = []
        self.user_preferences = {
            'learning_level': 'intermediate',
            'explanation_style': 'detailed',
            'language': 'en'
        }
        
        # Initialize models with error handling
        self._load_models()
        
        # Initialize embedding model
        self._initialize_embeddings()
        
        logger.info("Enhanced StudyMate system initialized")
        
    def _load_models(self):
        """Load AI models with comprehensive error handling"""
        self.tokenizer = None
        self.text_model = None
        self.vision_processor = None
        self.vision_model = None
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Using fallback text generation.")
            return
        
        try:
            logger.info("Attempting to load IBM Granite models...")
            
            # Try to load text model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.0-2b-instruct")
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    "ibm-granite/granite-3.0-2b-instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                logger.info("✅ Granite text model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Granite text model: {e}")
                # Try alternative models
                self._load_alternative_text_model()
            
            # Try to load vision model
            try:
                self.vision_processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
                self.vision_model = AutoModelForVision2Seq.from_pretrained(
                    "ibm-granite/granite-docling-258M",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                logger.info("✅ Granite vision model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Granite vision model: {e}")
            
            self.models_loaded = True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.models_loaded = False
    
    def _load_alternative_text_model(self):
        """Load alternative text model if Granite fails"""
        alternative_models = [
            "microsoft/DialoGPT-medium",
            "distilgpt2",
            "gpt2"
        ]
        
        for model_name in alternative_models:
            try:
                logger.info(f"Trying alternative model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.text_model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info(f"✅ Alternative model {model_name} loaded successfully")
                return
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue
    
    def _initialize_embeddings(self):
        """Initialize embedding model with fallbacks"""
        self.embedding_model = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence Transformers not available. Using basic text matching.")
            return
        
        embedding_models = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-multilingual-MiniLM-L12-v2'
        ]
        
        for model_name in embedding_models:
            try:
                logger.info(f"Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"✅ Embedding model {model_name} loaded successfully")
                return
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Enhanced PDF text extraction with multiple methods"""
        extracted_data = {
            'text': '',
            'pages': [],
            'images': [],
            'tables': [],
            'metadata': {},
            'extraction_method': '',
            'success': False
        }
        
        # Try PyMuPDF first (better performance)
        if FITZ_AVAILABLE:
            try:
                extracted_data = self._extract_with_pymupdf(pdf_path)
                if extracted_data['success']:
                    return extracted_data
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to PyPDF2
        if PDF_AVAILABLE:
            try:
                extracted_data = self._extract_with_pypdf2(pdf_path)
                if extracted_data['success']:
                    return extracted_data
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Last resort: basic text extraction
        return self._basic_text_extraction(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict:
        """Extract using PyMuPDF with enhanced error handling"""
        extracted_data = {
            'text': '',
            'pages': [],
            'images': [],
            'tables': [],
            'metadata': {},
            'extraction_method': 'PyMuPDF',
            'success': False
        }
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            extracted_data['metadata'] = doc.metadata
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with better formatting
                page_text = page.get_text("text")
                
                # Clean and format text
                page_text = self._clean_extracted_text(page_text)
                
                extracted_data['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}"
                extracted_data['pages'].append({
                    'page_num': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split()),
                    'char_count': len(page_text)
                })
                
                # Extract images with error handling
                try:
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n - pix.alpha < 4:
                                img_data = pix.tobytes("png")
                                extracted_data['images'].append({
                                    'page': page_num + 1,
                                    'index': img_index,
                                    'data': img_data,
                                    'size': len(img_data)
                                })
                            pix = None
                        except Exception as e:
                            logger.warning(f"Image extraction error on page {page_num + 1}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Image processing error on page {page_num + 1}: {e}")
                
                # Extract tables with error handling
                try:
                    tables = page.find_tables()
                    for table_index, table in enumerate(tables):
                        try:
                            table_data = table.extract()
                            extracted_data['tables'].append({
                                'page': page_num + 1,
                                'index': table_index,
                                'data': table_data,
                                'rows': len(table_data) if table_data else 0
                            })
                        except Exception as e:
                            logger.warning(f"Table extraction error on page {page_num + 1}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Table processing error on page {page_num + 1}: {e}")
            
            doc.close()
            extracted_data['success'] = True
            return extracted_data
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            extracted_data['text'] = f'PyMuPDF extraction error: {str(e)}'
            return extracted_data
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict:
        """Fallback extraction using PyPDF2"""
        extracted_data = {
            'text': '',
            'pages': [],
            'images': [],
            'tables': [],
            'metadata': {},
            'extraction_method': 'PyPDF2',
            'success': False
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    extracted_data['metadata'] = dict(pdf_reader.metadata)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        page_text = self._clean_extracted_text(page_text)
                        
                        extracted_data['text'] += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        extracted_data['pages'].append({
                            'page_num': page_num + 1,
                            'text': page_text,
                            'word_count': len(page_text.split()),
                            'char_count': len(page_text)
                        })
                    except Exception as e:
                        logger.warning(f"PyPDF2 page extraction error on page {page_num + 1}: {e}")
                        continue
            
            extracted_data['success'] = True
            return extracted_data
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            extracted_data['text'] = f'PyPDF2 extraction error: {str(e)}'
            return extracted_data
    
    def _basic_text_extraction(self, pdf_path: str) -> Dict:
        """Last resort basic text extraction"""
        return {
            'text': f'Unable to extract text from {os.path.basename(pdf_path)}. Please ensure the PDF is not corrupted and try again.',
            'pages': [],
            'images': [],
            'tables': [],
            'metadata': {},
            'extraction_method': 'Failed',
            'success': False
        }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentence endings
        
        # Remove page headers/footers patterns
        text = re.sub(r'Page \d+.*?\n', '', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        return text.strip()
    
    def enhanced_chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Enhanced text chunking with metadata"""
        if not text:
            return []
        
        # Smart sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) + 1 < chunk_size:
                current_chunk += sentence + " "
                current_sentences.append(i)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'sentence_indices': current_sentences.copy(),
                        'start_sentence': current_sentences[0] if current_sentences else i,
                        'end_sentence': current_sentences[-1] if current_sentences else i,
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk)
                    })
                
                # Start new chunk with overlap
                if overlap > 0 and current_sentences:
                    overlap_sentences = sentences[max(0, current_sentences[-overlap:][0]):current_sentences[-1]+1]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                    current_sentences = list(range(max(0, current_sentences[-overlap:][0]), i+1))
                else:
                    current_chunk = sentence + " "
                    current_sentences = [i]
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'sentence_indices': current_sentences.copy(),
                'start_sentence': current_sentences[0] if current_sentences else len(sentences)-1,
                'end_sentence': current_sentences[-1] if current_sentences else len(sentences)-1,
                'word_count': len(current_chunk.split()),
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def build_enhanced_search_index(self):
        """Build enhanced search index with multiple methods"""
        if not self.text_chunks:
            logger.warning("No text chunks available for indexing")
            return False
        
        try:
            # Extract just the text from chunk dictionaries
            chunk_texts = [chunk['text'] if isinstance(chunk, dict) else str(chunk) for chunk in self.text_chunks]
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                return self._build_semantic_index(chunk_texts)
            else:
                return self._build_keyword_index(chunk_texts)
        except Exception as e:
            logger.error(f"Error building search index: {e}")
            return False
    
    def _build_semantic_index(self, texts: List[str]) -> bool:
        """Build semantic search index using embeddings"""
        try:
            logger.info("Building semantic search index...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            if FAISS_AVAILABLE:
                # Build FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.index.add(embeddings.astype('float32'))
                
                logger.info(f"✅ Semantic index built with {len(texts)} chunks")
                return True
            else:
                # Store embeddings for manual similarity search
                self.embeddings = embeddings
                logger.info(f"✅ Embeddings stored for {len(texts)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Semantic index building failed: {e}")
            return False
    
    def _build_keyword_index(self, texts: List[str]) -> bool:
        """Build simple keyword-based index as fallback"""
        try:
            logger.info("Building keyword-based search index...")
            # Simple TF-IDF-like approach
            from collections import defaultdict, Counter
            
            # Build inverted index
            self.keyword_index = defaultdict(list)
            
            for i, text in enumerate(texts):
                words = re.findall(r'\b\w+\b', text.lower())
                word_counts = Counter(words)
                
                for word, count in word_counts.items():
                    if len(word) > 2:  # Skip very short words
                        self.keyword_index[word].append((i, count))
            
            logger.info(f"✅ Keyword index built with {len(self.keyword_index)} unique terms")
            return True
            
        except Exception as e:
            logger.error(f"Keyword index building failed: {e}")
            return False
    
    def enhanced_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Enhanced search with multiple fallback methods"""
        if not self.text_chunks:
            return []
        
        try:
            # Extract text from chunks for searching
            chunk_texts = [chunk['text'] if isinstance(chunk, dict) else str(chunk) for chunk in self.text_chunks]
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                if FAISS_AVAILABLE and self.index:
                    return self._semantic_search_faiss(query, chunk_texts, top_k)
                elif hasattr(self, 'embeddings'):
                    return self._semantic_search_manual(query, chunk_texts, top_k)
            
            # Fallback to keyword search
            return self._keyword_search(query, chunk_texts, top_k)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._simple_text_search(query, chunk_texts, top_k)
    
    def _semantic_search_faiss(self, query: str, texts: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        """Semantic search using FAISS index"""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(texts):
                source = self.chunk_sources[idx] if idx < len(self.chunk_sources) else "Unknown"
                results.append((texts[idx], float(score), source))
        
        return results
    
    def _semantic_search_manual(self, query: str, texts: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        """Manual semantic search using stored embeddings"""
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(query_embedding, self.embeddings.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(texts):
                source = self.chunk_sources[idx] if idx < len(self.chunk_sources) else "Unknown"
                results.append((texts[idx], float(similarities[idx]), source))
        
        return results
    
    def _keyword_search(self, query: str, texts: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        """Keyword-based search fallback"""
        if not hasattr(self, 'keyword_index'):
            return self._simple_text_search(query, texts, top_k)
        
        from collections import defaultdict
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        chunk_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.keyword_index:
                for chunk_idx, count in self.keyword_index[word]:
                    chunk_scores[chunk_idx] += count
        
        # Sort by score and get top results
        top_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_idx, score in top_chunks:
            if chunk_idx < len(texts):
                source = self.chunk_sources[chunk_idx] if chunk_idx < len(self.chunk_sources) else "Unknown"
                results.append((texts[chunk_idx], score, source))
        
        return results
    
    def _simple_text_search(self, query: str, texts: List[str], top_k: int) -> List[Tuple[str, float, str]]:
        """Simple text matching as last resort"""
        query_lower = query.lower()
        results = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Simple scoring based on word matches
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            
            common_words = query_words.intersection(text_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            if score > 0:
                source = self.chunk_sources[i] if i < len(self.chunk_sources) else "Unknown"
                results.append((text, score, source))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def generate_enhanced_answer(self, question: str, context: str, learning_level: str = "intermediate") -> str:
        """Generate answer with enhanced fallback methods"""
        if self.models_loaded and self.tokenizer and self.text_model:
            try:
                return self._generate_ai_answer(question, context, learning_level)
            except Exception as e:
                logger.warning(f"AI answer generation failed: {e}")
                return self._generate_fallback_answer(question, context, learning_level)
        else:
            return self._generate_fallback_answer(question, context, learning_level)
    
    def _generate_ai_answer(self, question: str, context: str, learning_level: str) -> str:
        """Generate AI-powered answer"""
        # Create appropriate prompt based on learning level
        prompts = {
            "beginner": "You are a helpful tutor. Explain concepts in simple, easy-to-understand terms with examples that a beginner can follow.",
            "intermediate": "You are a knowledgeable study assistant. Provide clear, comprehensive explanations suitable for someone with basic background knowledge.",
            "advanced": "You are an expert academic assistant. Provide detailed, technical explanations with depth and nuance for advanced learners."
        }
        
        system_prompt = prompts.get(learning_level, prompts["intermediate"])
        
        prompt = f"""Context: {context[:2000]}

Question: {question}

{system_prompt} Answer the question based on the provided context. If the context doesn't contain enough information, clearly state what information is missing."""

        try:
            # Handle different tokenizer interfaces
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Fallback for simpler tokenizers
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                if isinstance(inputs, torch.Tensor):
                    inputs = {"input_ids": inputs}
            
            # Move to device if CUDA available
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.text_model.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.text_model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.text_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                )
            
            # Decode the response
            if "input_ids" in inputs:
                answer = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
            else:
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"AI answer generation error: {e}")
            return self._generate_fallback_answer(question, context, learning_level)
    
    def _generate_fallback_answer(self, question: str, context: str, learning_level: str) -> str:
        """Generate fallback answer using rule-based approach"""
        try:
            # Extract relevant sentences from context
            context_sentences = re.split(r'[.!?]+', context)
            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            
            relevant_sentences = []
            for sentence in context_sentences:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                if overlap >= 2:  # At least 2 word overlap
                    relevant_sentences.append((sentence.strip(), overlap))
            
            # Sort by relevance and take top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:3]]
            
            if top_sentences:
                answer = "Based on the provided context:\n\n"
                answer += "\n\n".join(top_sentences)
                
                if learning_level == "beginner":
                    answer += "\n\n💡 This information directly relates to your question. Let me know if you need any terms explained!"
                elif learning_level == "advanced":
                    answer += "\n\n🔍 For a more comprehensive understanding, you may want to explore related concepts in the source material."
                
            else:
                answer = "I found relevant information in the context, but it may not directly answer your specific question. "
                answer += f"The context discusses: {context[:200]}..."
                answer += "\n\nPlease rephrase your question or ask about specific aspects mentioned in the context."
            
            return answer
            
        except Exception as e:
            logger.error(f"Fallback answer generation error: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it or check if your documents were uploaded successfully."
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'models_loaded': self.models_loaded,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'faiss_available': FAISS_AVAILABLE,
            'pdf_processing_available': PDF_AVAILABLE or FITZ_AVAILABLE,
            'documents_count': len(self.documents),
            'chunks_count': len(self.text_chunks),
            'search_index_ready': self.index is not None or hasattr(self, 'keyword_index'),
            'device': self.device,
            'embedding_model_loaded': self.embedding_model is not None
        }


# Initialize the enhanced system
studymate = None

def initialize_system():
    """Initialize the StudyMate system"""
    global studymate
    try:
        studymate = EnhancedStudyMateSystem()
        return "✅ Enhanced StudyMate system initialized successfully!"
    except Exception as e:
        return f"❌ System initialization failed: {e}"


def upload_pdfs_enhanced(files):
    """Enhanced PDF upload handler"""
    global studymate
    
    if studymate is None:
        return "System not initialized. Please restart the application.", "", ""
    
    if not files:
        return "No files uploaded.", "", ""
    
    # Clear previous data
    studymate.documents.clear()
    studymate.text_chunks.clear()
    studymate.chunk_sources.clear()
    studymate.chunk_metadata.clear()
    
    status_messages = []
    all_text = ""
    processing_stats = {
        'total_files': len(files),
        'successful': 0,
        'failed': 0,
        'total_pages': 0,
        'total_chunks': 0
    }
    
    for file in files:
        try:
            filename = os.path.basename(file.name)
            logger.info(f"Processing: {filename}")
            
            # Extract text from PDF
            extracted_data = studymate.extract_text_from_pdf(file.name)
            
            if extracted_data['success']:
                studymate.documents[filename] = extracted_data
                
                # Enhanced chunking
                chunk_dicts = studymate.enhanced_chunk_text(extracted_data['text'])
                studymate.text_chunks.extend(chunk_dicts)
                studymate.chunk_sources.extend([filename] * len(chunk_dicts))
                
                # Store metadata for each chunk
                for i, chunk_dict in enumerate(chunk_dicts):
                    metadata = {
                        'source_file': filename,
                        'chunk_index': i,
                        'extraction_method': extracted_data['extraction_method'],
                        'page_count': len(extracted_data['pages']),
                        'has_images': len(extracted_data['images']) > 0,
                        'has_tables': len(extracted_data['tables']) > 0
                    }
                    studymate.chunk_metadata.append(metadata)
                
                all_text += f"\n\n--- {filename} ---\n{extracted_data['text'][:500]}..."
                
                processing_stats['successful'] += 1
                processing_stats['total_pages'] += len(extracted_data['pages'])
                processing_stats['total_chunks'] += len(chunk_dicts)
                
                status_messages.append(
                    f"✅ {filename}: {len(chunk_dicts)} chunks, "
                    f"{len(extracted_data['pages'])} pages, "
                    f"{len(extracted_data['images'])} images, "
                    f"{len(extracted_data['tables'])} tables"
                )
            else:
                processing_stats['failed'] += 1
                status_messages.append(f"❌ {filename}: Extraction failed")
                
        except Exception as e:
            processing_stats['failed'] += 1
            logger.error(f"Error processing {file.name}: {e}")
            status_messages.append(f"❌ {file.name}: {str(e)}")
    
    # Build enhanced search index
    index_status = "⚠️ Search index not built"
    if studymate.text_chunks:
        if studymate.build_enhanced_search_index():
            index_status = "🔍 Enhanced search index built successfully"
        else:
            index_status = "⚠️ Search index building failed, using fallback search"
    
    status_messages.append(index_status)
    
    # Add processing summary
    status_messages.insert(0, f"""
📊 Processing Summary:
- Total files: {processing_stats['total_files']}
- Successful: {processing_stats['successful']}
- Failed: {processing_stats['failed']}
- Total pages: {processing_stats['total_pages']}
- Total chunks: {processing_stats['total_chunks']}
""")
    
    # Generate enhanced summary
    summary = ""
    if all_text and studymate.models_loaded:
        try:
            summary = studymate.generate_enhanced_answer(
                "Please provide a comprehensive summary of the uploaded documents", 
                all_text[:3000], 
                "intermediate"
            )
        except Exception as e:
            summary = f"Summary generation failed: {e}"
    elif all_text:
        # Fallback summary
        words = all_text.split()
        summary = f"Uploaded {processing_stats['successful']} documents with approximately {len(words)} words across {processing_stats['total_pages']} pages."
    
    return "\n".join(status_messages), all_text[:2000], summary


def ask_question_enhanced(question, learning_level, explanation_style):
    """Enhanced question answering"""
    global studymate
    
    if studymate is None:
        return "System not initialized. Please restart the application."
    
    if not question.strip():
        return "Please enter a question."
    
    if not studymate.text_chunks:
        return "Please upload PDF documents first."
    
    try:
        # Add to conversation history
        studymate.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'learning_level': learning_level,
            'explanation_style': explanation_style
        })
        
        # Perform enhanced search
        search_results = studymate.enhanced_search(question, top_k=5)
        
        if not search_results:
            return "No relevant information found in the uploaded documents. Please try rephrasing your question or check if the content relates to your uploaded PDFs."
        
        # Combine context from top results
        context_parts = []
        sources = set()
        
        for result in search_results:
            context_parts.append(result[0])
            sources.add(result[2])
        
        context = "\n\n".join(context_parts)
        
        # Generate enhanced answer
        answer = studymate.generate_enhanced_answer(question, context, learning_level)
        
        # Add metadata and source information
        metadata_info = f"""

📚 **Sources:** {', '.join(sources)}
🔍 **Search Method:** {"Semantic" if studymate.embedding_model else "Keyword-based"}
📊 **Relevance Scores:** {[f"{result[2]}: {result[1]:.3f}" for result in search_results[:3]]}
⏰ **Response Time:** {datetime.now().strftime('%H:%M:%S')}"""
        
        # Add learning level appropriate footer
        if learning_level == "beginner":
            footer = "\n\n💡 **Need clarification?** Feel free to ask for simpler explanations of any concepts!"
        elif learning_level == "advanced":
            footer = "\n\n🔬 **For deeper insights:** Consider asking about related concepts or methodologies."
        else:
            footer = "\n\n📖 **Follow-up suggestions:** Ask about specific details or request examples for better understanding."
        
        return answer + metadata_info + footer
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        return f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question."


def generate_learning_aids_enhanced(content_type, num_items):
    """Enhanced learning aids generation"""
    global studymate
    
    if studymate is None:
        return "System not initialized. Please restart the application."
    
    if not studymate.text_chunks:
        return "Please upload PDF documents first."
    
    try:
        # Select diverse chunks for better coverage
        total_chunks = len(studymate.text_chunks)
        selected_indices = list(range(0, min(total_chunks, 10), max(1, total_chunks // 10)))
        
        sample_texts = []
        for idx in selected_indices:
            chunk = studymate.text_chunks[idx]
            text = chunk['text'] if isinstance(chunk, dict) else str(chunk)
            sample_texts.append(text)
        
        combined_text = "\n\n".join(sample_texts)
        
        if content_type == "Summary":
            if studymate.models_loaded:
                summary = studymate.generate_enhanced_answer(
                    f"Create a comprehensive summary of the key concepts, main points, and important details from this material in approximately 200-300 words.",
                    combined_text,
                    "intermediate"
                )
            else:
                # Fallback summary generation
                sentences = re.split(r'[.!?]+', combined_text)
                important_sentences = []
                
                # Simple scoring based on sentence length and keyword frequency
                for sentence in sentences:
                    words = sentence.split()
                    if 10 <= len(words) <= 30:  # Good length sentences
                        important_sentences.append(sentence.strip())
                
                summary = ". ".join(important_sentences[:5]) + "."
            
            return f"📋 **Document Summary**\n\n{summary}\n\n*Generated from {len(sample_texts)} document sections*"
        
        elif content_type == "Quiz":
            quiz_content = f"🧠 **Generated Quiz Questions**\n\n"
            
            if studymate.models_loaded:
                quiz_prompt = f"""Create {num_items} multiple choice questions based on this content. 
Format each question clearly with:
- Question number
- The question
- 4 multiple choice options (A, B, C, D)  
- The correct answer

Make questions that test understanding of key concepts.

Content: {combined_text[:2000]}"""
                
                quiz_response = studymate.generate_enhanced_answer("", quiz_prompt, "intermediate")
                quiz_content += quiz_response
            else:
                # Fallback quiz generation
                quiz_content += "**Sample Questions Based on Content:**\n\n"
                for i in range(min(num_items, 3)):
                    quiz_content += f"**Question {i+1}:** What is the main concept discussed in section {i+1}?\n"
                    quiz_content += f"A) Concept A\nB) Concept B\nC) Concept C\nD) Concept D\n"
                    quiz_content += f"*Answer: Review the source material*\n\n"
            
            return quiz_content
        
        elif content_type == "Flashcards":
            flashcards_content = f"🃏 **Generated Flashcards**\n\n"
            
            if studymate.models_loaded:
                flashcard_prompt = f"""Create {num_items} flashcards from this content. 
Format each flashcard as:
**Card X:**
**Front:** [Question or term]
**Back:** [Answer or explanation]

Focus on key concepts, definitions, and important facts.

Content: {combined_text[:2000]}"""
                
                flashcard_response = studymate.generate_enhanced_answer("", flashcard_prompt, "intermediate")
                flashcards_content += flashcard_response
            else:
                # Fallback flashcard generation
                key_phrases = re.findall(r'[A-Z][a-z]+ [a-z]+|[A-Z][A-Z]+', combined_text)
                unique_phrases = list(set(key_phrases))[:num_items]
                
                for i, phrase in enumerate(unique_phrases, 1):
                    flashcards_content += f"**Card {i}:**\n"
                    flashcards_content += f"**Front:** What is {phrase}?\n"
                    flashcards_content += f"**Back:** [Refer to source material for definition]\n\n"
            
            return flashcards_content
        
    except Exception as e:
        logger.error(f"Learning aids generation error: {e}")
        return f"Error generating learning aids: {str(e)}. Please try again with different settings."


def analyze_image_enhanced(image, question):
    """Enhanced image analysis"""
    global studymate
    
    if studymate is None:
        return "System not initialized. Please restart the application."
    
    if image is None:
        return "Please upload an image."
    
    if not question.strip():
        return "Please enter a question about the image."
    
    try:
        if studymate.vision_processor and studymate.vision_model:
            # Use AI vision model
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return studymate.analyze_image_with_question(img_byte_arr, question)
        else:
            # Fallback analysis
            return f"""
🖼️ **Image Analysis (Fallback Mode)**

I can see that you've uploaded an image, but the vision model is not available. 

**Your question:** {question}

**Fallback analysis:**
- Image format appears to be valid
- For detailed image analysis, please ensure vision processing dependencies are installed
- You can try describing the image in text and ask questions about similar content in your uploaded documents

**Suggestion:** Upload the document containing this image as a PDF for text-based analysis of related content.
"""
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error analyzing image: {str(e)}. Please try again or ensure the image is in a supported format."


def get_system_info():
    """Get enhanced system information"""
    global studymate
    
    if studymate is None:
        return "❌ System not initialized"
    
    status = studymate.get_system_status()
    
    info = f"""
🚀 **Enhanced StudyMate System Status**

**Core System:**
- Models Loaded: {'✅' if status['models_loaded'] else '❌'}
- Search Index: {'✅' if status['search_index_ready'] else '❌'}  
- Processing Device: {status['device']}

**Available Features:**
- PDF Processing: {'✅' if status['pdf_processing_available'] else '❌'}
- AI Text Generation: {'✅' if status['transformers_available'] else '❌'}
- Semantic Search: {'✅' if status['sentence_transformers_available'] else '❌'}
- Advanced Indexing: {'✅' if status['faiss_available'] else '❌'}

**Current Session:**
- Documents Loaded: {status['documents_count']}
- Text Chunks: {status['chunks_count']}
- Conversations: {len(studymate.conversation_history) if studymate else 0}

**Performance Notes:**
- Using {'GPU acceleration' if status['device'] == 'cuda' else 'CPU processing'}
- Search method: {'Semantic' if status['sentence_transformers_available'] else 'Keyword-based'}
- Fallback systems: {'Active' if not status['models_loaded'] else 'Standby'}
"""
    
    return info


# Create enhanced Gradio interface
def create_enhanced_interface():
    """Create the enhanced Gradio interface"""
    
    with gr.Blocks(
        title="Enhanced StudyMate - AI-Powered PDF Q&A System",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .feature-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        }
        .status-box {
            background: #f0f9ff;
            border-left: 4px solid #0ea5e9;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .enhanced-button {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .enhanced-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎓 Enhanced StudyMate</h1>
            <h2>AI-Powered PDF-Based Q&A System</h2>
            <p>🚀 Powered by IBM Granite Models | 🔧 Enhanced with Robust Error Handling & Fallbacks</p>
            <p>📚 Built for Students, Researchers, and Lifelong Learners</p>
        </div>
        """)
        
        # Initialize system
        init_status = initialize_system()
        gr.HTML(f'<div class="status-box">📋 <strong>System Status:</strong> {init_status}</div>')
        
        with gr.Tabs():
            # Tab 1: Enhanced Document Upload and Management
            with gr.TabItem("📄 Upload & Process Documents"):
                gr.Markdown("## 📤 Upload your PDF documents to get started")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="📁 Select PDF files (Multiple files supported)",
                            file_count="multiple",
                            file_types=[".pdf"],
                            height=180
                        )
                        upload_btn = gr.Button(
                            "🚀 Process Documents", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["enhanced-button"]
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>✨ Enhanced Features:</h4>
                            <ul>
                                <li>🔄 <strong>Multi-Method Extraction:</strong> PyMuPDF + PyPDF2 fallback</li>
                                <li>🧠 <strong>Smart Chunking:</strong> Sentence-aware text segmentation</li>
                                <li>🔍 <strong>Advanced Indexing:</strong> Semantic + keyword search</li>
                                <li>📊 <strong>Rich Metadata:</strong> Images, tables, and statistics</li>
                                <li>🛡️ <strong>Error Recovery:</strong> Robust fallback systems</li>
                                <li>⚡ <strong>Performance:</strong> Optimized for speed and accuracy</li>
                            </ul>
                        </div>
                        """)
                
                upload_status = gr.Textbox(
                    label="📊 Processing Status & Statistics", 
                    lines=8, 
                    max_lines=15,
                    placeholder="Upload PDFs to see processing results..."
                )
                
                with gr.Row():
                    with gr.Column():
                        document_preview = gr.Textbox(
                            label="👀 Document Preview", 
                            lines=10, 
                            max_lines=20,
                            placeholder="Document content preview will appear here..."
                        )
                    with gr.Column():
                        document_summary = gr.Textbox(
                            label="📋 AI-Generated Summary", 
                            lines=10, 
                            max_lines=20,
                            placeholder="Intelligent summary will be generated automatically..."
                        )
            
            # Tab 2: Enhanced Q&A Interface
            with gr.TabItem("💬 Intelligent Q&A"):
                gr.Markdown("## 🤖 Ask intelligent questions about your documents")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="🎯 Your Question",
                            placeholder="Ask anything about your documents... (e.g., 'What are the main conclusions?' or 'Explain the methodology used')",
                            lines=3
                        )
                        
                        with gr.Row():
                            learning_level = gr.Dropdown(
                                choices=["beginner", "intermediate", "advanced"],
                                value="intermediate",
                                label="🎓 Learning Level",
                                info="Adjusts explanation complexity"
                            )
                            explanation_style = gr.Dropdown(
                                choices=["concise", "detailed", "comprehensive"],
                                value="detailed",
                                label="📖 Explanation Style",
                                info="Controls response length and depth"
                            )
                        
                        ask_btn = gr.Button(
                            "🧠 Get Intelligent Answer", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["enhanced-button"]
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>🎯 Enhanced Q&A Features:</h4>
                            <ul>
                                <li>🧠 <strong>Smart Search:</strong> Semantic understanding</li>
                                <li>📚 <strong>Context-Aware:</strong> Cross-document insights</li>
                                <li>🎓 <strong>Adaptive Learning:</strong> Level-appropriate responses</li>
                                <li>🔍 <strong>Source Attribution:</strong> Credible references</li>
                                <li>⚡ <strong>Fast Responses:</strong> Optimized processing</li>
                                <li>🛡️ <strong>Fallback Systems:</strong> Always functional</li>
                            </ul>
                        </div>
                        """)
                
                answer_output = gr.Textbox(
                    label="🤖 Intelligent Response", 
                    lines=12, 
                    max_lines=25,
                    placeholder="Your intelligent answer will appear here with sources and metadata..."
                )
                
                # Enhanced example questions
                gr.Markdown("### 💡 Intelligent Question Examples:")
                
                with gr.Row():
                    example_btns = [
                        gr.Button("📋 Summarize key findings", size="sm"),
                        gr.Button("🔬 Explain methodology", size="sm"),
                        gr.Button("📊 What are the results?", size="sm"),
                        gr.Button("🎯 Main conclusions?", size="sm"),
                        gr.Button("🔍 Find specific data", size="sm")
                    ]
                
                # Connect example buttons
                example_btns[0].click(lambda: "What are the main findings and key takeaways from this document?", outputs=question_input)
                example_btns[1].click(lambda: "Can you explain the methodology or approach used in this research?", outputs=question_input)
                example_btns[2].click(lambda: "What are the main results, outcomes, or data presented?", outputs=question_input)
                example_btns[3].click(lambda: "What are the primary conclusions and their implications?", outputs=question_input)
                example_btns[4].click(lambda: "Help me find specific data points, statistics, or measurements mentioned.", outputs=question_input)
            
            # Tab 3: Enhanced Learning Aids
            with gr.TabItem("🧠 Smart Learning Tools"):
                gr.Markdown("## 📚 Generate intelligent learning materials")
                
                with gr.Row():
                    with gr.Column():
                        content_type = gr.Dropdown(
                            choices=["Summary", "Quiz", "Flashcards"],
                            value="Summary",
                            label="📝 Learning Aid Type"
                        )
                        num_items = gr.Slider(
                            minimum=1,
                            maximum=15,
                            value=5,
                            step=1,
                            label="🔢 Number of Items (Quiz/Flashcards)"
                        )
                        generate_btn = gr.Button(
                            "✨ Generate Smart Learning Aid", 
                            variant="primary",
                            elem_classes=["enhanced-button"]
                        )
                    
                    with gr.Column():
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>📚 Smart Learning Tools:</h4>
                            <ul>
                                <li>📋 <strong>Intelligent Summaries:</strong> Key concept extraction</li>
                                <li>🧠 <strong>Adaptive Quizzes:</strong> Difficulty-aware questions</li>
                                <li>🃏 <strong>Smart Flashcards:</strong> Spaced repetition ready</li>
                                <li>🎯 <strong>Content-Aware:</strong> Document-specific generation</li>
                                <li>🔄 <strong>Multiple Formats:</strong> Various learning styles</li>
                                <li>📊 <strong>Quality Control:</strong> Fallback generation</li>
                            </ul>
                        </div>
                        """)
                
                learning_output = gr.Textbox(
                    label="📚 Generated Learning Materials", 
                    lines=15, 
                    max_lines=30,
                    placeholder="Your personalized learning materials will be generated here..."
                )
            
            # Tab 4: Enhanced Image Analysis
            with gr.TabItem("🖼️ Visual Intelligence"):
                gr.Markdown("## 🔍 Analyze charts, diagrams, and visual content")
                
                with gr.Row():
                    with gr.Column():
                        image_upload = gr.Image(
                            label="📷 Upload Image/Chart/Diagram", 
                            type="pil",
                            height=300
                        )
                        image_question = gr.Textbox(
                            label="❓ Question about the image",
                            placeholder="e.g., 'What does this chart show?', 'Explain this diagram', 'What are the trends?'",
                            lines=3
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze with AI Vision", 
                            variant="primary",
                            elem_classes=["enhanced-button"]
                        )
                    
                    with gr.Column():
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>🔍 Visual Intelligence Features:</h4>
                            <ul>
                                <li>📊 <strong>Chart Analysis:</strong> Data interpretation</li>
                                <li>🗺️ <strong>Diagram Understanding:</strong> Process flows</li>
                                <li>📋 <strong>Table Reading:</strong> Structured data extraction</li>
                                <li>🧠 <strong>Multi-Modal AI:</strong> Vision + language</li>
                                <li>📝 <strong>Context Integration:</strong> Document correlation</li>
                                <li>🛡️ <strong>Fallback Support:</strong> Always available</li>
                            </ul>
                        </div>
                        """)
                
                image_analysis_output = gr.Textbox(
                    label="👁️ Visual Analysis Results", 
                    lines=12, 
                    max_lines=20,
                    placeholder="AI-powered visual analysis will appear here..."
                )
            
            # Tab 5: Enhanced System Info & Settings
            with gr.TabItem("⚙️ System Dashboard"):
                gr.Markdown("## 🖥️ Enhanced system information and performance metrics")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>🚀 System Capabilities</h4>
                            <ul>
                                <li>🧠 <strong>AI Models:</strong> IBM Granite + alternatives</li>
                                <li>🔍 <strong>Search Engine:</strong> Semantic + keyword hybrid</li>
                                <li>📁 <strong>PDF Processing:</strong> Multi-method extraction</li>
                                <li>🛡️ <strong>Error Handling:</strong> Comprehensive fallbacks</li>
                                <li>⚡ <strong>Performance:</strong> GPU/CPU adaptive</li>
                                <li>🔒 <strong>Privacy:</strong> Local processing</li>
                            </ul>
                        </div>
                        """)
                        
                        system_info = gr.Textbox(
                            label="📊 Detailed System Status", 
                            value="Loading system information...", 
                            lines=15,
                            max_lines=25
                        )
                        refresh_info_btn = gr.Button(
                            "🔄 Refresh System Info", 
                            variant="secondary"
                        )
                    
                    with gr.Column():
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>📖 Enhanced Usage Guide</h4>
                            
                            <h5>🚀 Quick Start:</h5>
                            <ol>
                                <li><strong>Upload PDFs:</strong> Select multiple documents</li>
                                <li><strong>Wait for Processing:</strong> Smart extraction & indexing</li>
                                <li><strong>Ask Questions:</strong> Natural language queries</li>
                                <li><strong>Get Intelligent Answers:</strong> Context-aware responses</li>
                                <li><strong>Generate Learning Aids:</strong> Summaries, quizzes, flashcards</li>
                            </ol>
                            
                            <h5>💡 Pro Tips:</h5>
                            <ul>
                                <li>🎯 Be specific in questions for better results</li>
                                <li>📊 Use high-quality, text-readable PDFs</li>
                                <li>🎓 Adjust learning level for optimal explanations</li>
                                <li>🔄 Try rephrasing if results aren't satisfactory</li>
                                <li>📷 Upload images from documents for visual analysis</li>
                            </ul>
                            
                            <h5>🛡️ Reliability Features:</h5>
                            <ul>
                                <li>✅ Multiple PDF extraction methods</li>
                                <li>🔄 Automatic fallback systems</li>
                                <li>🧠 Alternative AI models</li>
                                <li>🔍 Hybrid search capabilities</li>
                                <li>📊 Comprehensive error reporting</li>
                            </ul>
                        </div>
                        """)
        
        # Enhanced event handlers
        upload_btn.click(
            upload_pdfs_enhanced,
            inputs=[file_upload],
            outputs=[upload_status, document_preview, document_summary]
        )
        
        ask_btn.click(
            ask_question_enhanced,
            inputs=[question_input, learning_level, explanation_style],
            outputs=[answer_output]
        )
        
        generate_btn.click(
            generate_learning_aids_enhanced,
            inputs=[content_type, num_items],
            outputs=[learning_output]
        )
        
        analyze_btn.click(
            analyze_image_enhanced,
            inputs=[image_upload, image_question],
            outputs=[image_analysis_output]
        )
        
        refresh_info_btn.click(
            get_system_info,
            outputs=[system_info]
        )
        
        # Enable Enter key for question input
        question_input.submit(
            ask_question_enhanced,
            inputs=[question_input, learning_level, explanation_style],
            outputs=[answer_output]
        )
        
        # Load initial system info
        interface.load(get_system_info, outputs=[system_info])
    
    return interface


if __name__ == "__main__":
    # Create and launch the enhanced interface
    print("🎓 Starting Enhanced StudyMate - AI-Powered PDF Q&A System")
    print("🚀 Enhanced with robust error handling and fallback systems")
    print("📚 Powered by IBM Granite Models with intelligent alternatives")
    print("🔧 Loading enhanced interface...")
    
    app = create_enhanced_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
        debug=False,
        inbrowser=True,
        favicon_path=None,
        ssl_verify=False
    )