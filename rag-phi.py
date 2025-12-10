
import re
import numpy as np
import faiss
import streamlit as st
import pdfplumber
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# ========================
# CONFIGURATION
# ========================
class Config:
    CHUNK_SIZE = 500  # Smaller chunks for better precision
    CHUNK_OVERLAP = 100
    TOP_K = 6  # Retrieve more, then re-rank
    FINAL_K = 3  # Use top 3 after re-ranking
    EMBED_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/phi-2"  # Phi-2 for better reasoning
    MAX_ANSWER_LENGTH = 256

# ========================
# IMPROVED PDF EXTRACTION
# ========================
class PDFProcessor:
    """Enhanced PDF processing with better structure preservation"""
    
    @staticmethod
    def extract_text_structured(file) -> Dict:
        """Extract text while preserving document structure"""
        pages = []
        full_text = []
        
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout
                text = page.extract_text(layout=True) or ""
                
                if text.strip():
                    pages.append({
                        'page_num': page_num + 1,
                        'text': text,
                        'char_count': len(text)
                    })
                    full_text.append(text)
        
        combined = "\n\n".join(full_text)
        cleaned = PDFProcessor.deep_clean(combined)
        
        return {
            'text': cleaned,
            'pages': pages,
            'total_pages': len(pages)
        }
    
    @staticmethod
    def deep_clean(text: str) -> str:
        """Deep cleaning of PDF artifacts"""
        # Fix broken words across lines
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Remove excessive whitespace but preserve paragraphs
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page numbers and headers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs and email addresses (usually not useful for QA)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Fix common PDF encoding issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace("'", "'").replace('"', '"').replace('"', '"')

        
        return text.strip()

# ========================
# SEMANTIC CHUNKING
# ========================
class SemanticChunker:
    """Creates semantically coherent chunks"""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences more accurately"""
        # Better sentence splitting that handles abbreviations
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Ignore very short fragments
                cleaned.append(sent)
        
        return cleaned
    
    @staticmethod
    def create_semantic_chunks(text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create chunks that preserve semantic meaning"""
        sentences = SemanticChunker.split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'id': len(chunks),
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i,
                    'length': len(chunk_text)
                })
                
                # Calculate overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'id': len(chunks),
                'start_sentence': len(sentences) - len(current_chunk),
                'end_sentence': len(sentences),
                'length': len(chunk_text)
            })
        
        return chunks

# ========================
# RETRIEVAL WITH RE-RANKING
# ========================
class SmartRetriever:
    """Retrieval with semantic re-ranking"""
    
    def __init__(self, model_name: str = Config.EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_embeddings = None
    
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index with stored embeddings"""
        self.chunks = chunks
        texts = [c['text'] for c in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = vectors.astype('float32')
        self.chunk_embeddings = vectors
        
        faiss.normalize_L2(vectors)
        
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
    
    def retrieve_and_rerank(self, query: str, top_k: int, final_k: int) -> List[Tuple[int, float, str]]:
        """Retrieve chunks and re-rank by cross-encoder similarity"""
        # Initial retrieval
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        # Re-rank using cosine similarity with original embeddings
        query_embedding = torch.tensor(query_vec[0])
        
        reranked = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk_text = self.chunks[idx]['text']
            
            # Calculate more accurate similarity
            chunk_emb = torch.tensor(self.chunk_embeddings[idx])
            similarity = util.cos_sim(query_embedding, chunk_emb).item()
            
            reranked.append((int(idx), float(similarity), chunk_text))
        
        # Sort by similarity and return top final_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:final_k]

# ========================
# PHI ANSWER GENERATION
# ========================
class PhiAnswerGenerator:
    """Answer generation using Microsoft Phi-2"""
    
    def __init__(self, model_name: str = Config.LLM_MODEL):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token to eos_token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model_name = model_name
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer with Phi-2"""
        # Combine contexts intelligently
        combined_context = "\n\n".join([f"Context {i+1}:\n{ctx[:400]}" for i, ctx in enumerate(contexts)])
        
        # Optimized prompt for Phi models
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

{combined_context}

Question: {question}

Answer the question concisely using only the information from the context above. If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        # Generate with optimized parameters for Phi
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_ANSWER_LENGTH,
                min_new_tokens=10,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decode and extract answer
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (after "Answer:")
        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        else:
            answer = full_output[len(prompt):].strip()
        
        # Clean up answer
        answer = answer.split("\n")[0].strip()  # Take first line if multiple
        
        # Post-process
        if answer and len(answer) > 0:
            if not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
        
        return answer if answer else "I couldn't generate an answer based on the provided context."

# ========================
# STREAMLIT UI
# ========================
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = None

def display_chunk_quality(chunks: List[Dict]):
    """Display chunk quality metrics"""
    lengths = [c['length'] for c in chunks]
    avg_length = np.mean(lengths)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Avg Chunk Size", f"{int(avg_length)} chars")
    with col3:
        st.metric("Size Range", f"{min(lengths)}-{max(lengths)}")

def main():
    st.set_page_config(
        page_title="Smart PDF RAG with Phi",
        page_icon="🧠",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.subheader("Retrieval")
        top_k = st.slider("Initial retrieval", 3, 10, Config.TOP_K, 
                         help="Number of chunks to retrieve initially")
        final_k = st.slider("Final chunks", 1, 5, Config.FINAL_K,
                           help="Number of chunks to use for answer")
        
        Config.TOP_K = top_k
        Config.FINAL_K = final_k
        
        st.subheader("Chunking")
        chunk_size = st.slider("Chunk size", 300, 800, Config.CHUNK_SIZE, step=50)
        overlap = st.slider("Overlap", 50, 200, Config.CHUNK_OVERLAP, step=25)
        
        Config.CHUNK_SIZE = chunk_size
        Config.CHUNK_OVERLAP = overlap
        
        if st.session_state.pdf_processed:
            st.divider()
            st.subheader("📊 Document Info")
            display_chunk_quality(st.session_state.chunks)
        
        st.divider()
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    st.title("🧠 Smart PDF RAG with Phi-2")
    st.markdown("*Semantic chunking + Re-ranking + Microsoft Phi-2*")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        help="Upload a PDF to analyze"
    )
    
    if not uploaded_file:
        st.info("👆 Upload a PDF document to begin")
        st.markdown("""
        **Features:**
        - ✅ Semantic chunking preserves context
        - ✅ Re-ranking improves relevance
        - ✅ Microsoft Phi-2 for better reasoning
        - ✅ Chunk quality metrics
        """)
        return
    
    # Process PDF
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            progress = st.progress(0)
            
            # Extract
            progress.progress(20, "Extracting text...")
            pdf_data = PDFProcessor.extract_text_structured(uploaded_file)
            st.session_state.pdf_metadata = pdf_data
            
            if len(pdf_data['text']) < 100:
                st.error("❌ Insufficient text extracted")
                return
            
            # Chunk
            progress.progress(40, "Creating semantic chunks...")
            chunker = SemanticChunker()
            chunks = chunker.create_semantic_chunks(
                pdf_data['text'],
                Config.CHUNK_SIZE,
                Config.CHUNK_OVERLAP
            )
            st.session_state.chunks = chunks
            
            # Build index
            progress.progress(60, "Building search index...")
            retriever = SmartRetriever()
            retriever.build_index(chunks)
            st.session_state.retriever = retriever
            
            # Load model
            progress.progress(80, "Loading Phi-2 model...")
            generator = PhiAnswerGenerator()
            st.session_state.generator = generator
            
            progress.progress(100, "Complete!")
            st.session_state.pdf_processed = True
            
        st.success(f"✅ Processed {pdf_data['total_pages']} pages into {len(chunks)} chunks")
    
    # Document preview
    with st.expander("📄 Document Preview"):
        st.text(st.session_state.chunks[0]['text'][:600] + "...")
    
    # Query interface
    st.divider()
    st.subheader("💬 Ask Questions")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="What is the main topic of this document?",
            label_visibility="collapsed"
        )
    with col2:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_button and query:
        with st.spinner("Searching and generating answer..."):
            # Retrieve and re-rank
            results = st.session_state.retriever.retrieve_and_rerank(
                query, Config.TOP_K, Config.FINAL_K
            )
            
            # Generate answer
            contexts = [r[2] for r in results]
            answer = st.session_state.generator.generate_answer(query, contexts)
            
            # Save to history
            st.session_state.history.append({
                'question': query,
                'answer': answer,
                'contexts': results
            })
        
        st.rerun()
    
    # Display history
    if st.session_state.history:
        st.divider()
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"### Q{len(st.session_state.history) - i}: {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                
                with st.expander("📚 Source Chunks"):
                    for j, (idx, score, text) in enumerate(item['contexts']):
                        st.markdown(f"**Chunk {idx}** • Relevance: `{score:.3f}`")
                        st.info(text[:400] + ("..." if len(text) > 400 else ""))
                
                st.markdown("---")

if __name__ == "__main__":
    main()