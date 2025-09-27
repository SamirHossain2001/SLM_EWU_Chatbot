import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import math
from dataclasses import dataclass
import gdown
import os
import time
from datetime import datetime
import json

# Set page config
st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (Dark Theme Compatible)
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #4fc3f7;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(79, 195, 247, 0.3);
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.8rem 0;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* User message styling - works in both light and dark */
    .user-message {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(33, 150, 243, 0.05));
        border-left: 4px solid #2196f3;
        border: 1px solid rgba(33, 150, 243, 0.2);
    }
    
    /* Bot message styling - works in both light and dark */
    .bot-message {
        background: linear-gradient(135deg, rgba(156, 39, 176, 0.15), rgba(156, 39, 176, 0.05));
        border-left: 4px solid #9c27b0;
        border: 1px solid rgba(156, 39, 176, 0.2);
    }
    
    /* Message headers */
    .message-header {
        font-weight: bold;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    /* Sidebar info box - adapts to theme */
    .sidebar-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric container - adapts to theme */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Ensure text is visible in both themes */
    .sidebar-info, .metric-container {
        color: inherit;
    }
    
    .sidebar-info strong, .metric-container strong {
        color: #4fc3f7;
        font-weight: 600;
    }
    
    /* Loading spinner custom style */
    .stSpinner > div > div {
        border-color: #4fc3f7 transparent transparent transparent !important;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2196f3, #21cbf3);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    
    /* Slider customization */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #2196f3, #21cbf3);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        .chat-message {
            padding: 0.8rem;
            margin: 0.5rem 0;
        }
        .sidebar-info, .metric-container {
            padding: 1rem;
        }
    }
    
    /* Dark mode specific adjustments */
    @media (prefers-color-scheme: dark) {
        .user-message {
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(33, 150, 243, 0.1));
        }
        .bot-message {
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.2), rgba(156, 39, 176, 0.1));
        }
        .sidebar-info, .metric-container {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
    }
</style>
""", unsafe_allow_html=True)

# Model architecture classes (same as your training code)
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Caching functions
@st.cache_resource
def load_tokenizer():
    """Load and cache the tokenizer"""
    return tiktoken.get_encoding("gpt2")

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and load it"""
    # Your model's Google Drive URL
    drive_url = "https://drive.google.com/file/d/1IOJWO5YGAqMJaziKsMilGfBt8BHHsMDB/view?usp=sharing"
    
    try:
        with st.spinner("Loading model..."):
            # Extract file ID from Google Drive URL
            file_id = drive_url.split("/d/")[1].split("/")[0]
            
            # Download file
            model_path = "university_chatbot_complete.pt"
            if not os.path.exists(model_path):
                st.info("Downloading model from Google Drive... This may take a few minutes.")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
            
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'config' in checkpoint:
                config = checkpoint['config']
            elif hasattr(checkpoint, 'config'):
                config = checkpoint.config
            else:
                # Create default config if not found
                st.warning("Config not found in checkpoint. Using default configuration.")
                config = GPTConfig(
                    block_size=128,
                    vocab_size=50257,
                    n_layer=6,
                    n_head=8,
                    n_embd=768,
                    dropout=0.1,
                    bias=False
                )
            
            model = GPT(config)
            
            # Handle different checkpoint formats for model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            
            return model, checkpoint, device, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check if the model file is accessible and properly formatted.")
        return None, None, None, None

def generate_response(model, tokenizer, prompt, max_tokens, temperature, top_k, device):
    """Generate response from the model with improved formatting"""
    try:
        # Clean and prepare the prompt
        prompt = prompt.strip()
        
        # Encode prompt
        tokens = tokenizer.encode_ordinary(prompt)
        if len(tokens) > model.config.block_size - max_tokens:
            tokens = tokens[-(model.config.block_size - max_tokens):]
        
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate with improved parameters
        with torch.no_grad():
            y = model.generate(
                x, 
                max_new_tokens=max_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
        
        # Decode the full generated text
        full_generated = tokenizer.decode(y[0].tolist())
        
        # Extract only the new part (response)
        if len(full_generated) > len(prompt):
            response = full_generated[len(prompt):].strip()
        else:
            response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # Clean up the response
        response = clean_response(response, prompt)
        
        return response
        
    except Exception as e:
        return f"I'm sorry, I encountered an error while generating a response. Please try again with a different question."

def clean_response(response, original_prompt):
    """Clean and format the generated response"""
    # Remove any repetition of the original prompt
    response = response.strip()
    
    # If response starts with prompt words, remove them
    prompt_words = original_prompt.lower().split()
    response_words = response.split()
    
    # Find where the actual response starts
    clean_words = []
    skip_mode = True
    
    for word in response_words:
        word_clean = word.lower().strip('.,!?;:')
        
        # If we find a word that's not in the prompt, start including
        if skip_mode and word_clean not in [w.lower() for w in prompt_words]:
            skip_mode = False
        
        if not skip_mode:
            clean_words.append(word)
    
    # Join the clean words
    clean_response = ' '.join(clean_words).strip()
    
    # If response is too short or empty, return original
    if len(clean_response) < 10:
        clean_response = response
    
    # Ensure response doesn't start with lowercase (unless intentional)
    if clean_response and clean_response[0].islower() and clean_response[0].isalpha():
        clean_response = clean_response[0].upper() + clean_response[1:]
    
    # Remove incomplete sentences at the end
    sentences = clean_response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        clean_response = '.'.join(sentences[:-1]) + '.'
    
    return clean_response

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'total_tokens_generated' not in st.session_state:
    st.session_state.total_tokens_generated = 0

# Auto-load model on startup
if not st.session_state.model_loaded:
    try:
        model, checkpoint, device, config = download_and_load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.checkpoint = checkpoint
            st.session_state.device = device
            st.session_state.config = config
            st.session_state.tokenizer = load_tokenizer()
            st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"Failed to auto-load model: {str(e)}")

# Main UI
st.markdown('<h1 class="main-header">🎓 University Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model status section
    st.subheader("🤖 Model Status")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully!")
        
        # Reload button
        if st.button("🔄 Reload Model"):
            st.session_state.model_loaded = False
            st.cache_resource.clear()
            st.rerun()
    else:
        st.error("❌ Model not loaded")
        if st.button("🚀 Retry Loading Model"):
            st.cache_resource.clear()
            st.rerun()
    
    # Generation parameters (Fixed optimal values)
    if st.session_state.model_loaded:
        st.subheader("🎛️ Generation Settings")
        
        # Fixed optimal parameters with explanations
        st.markdown("""
        **Fixed Optimal Settings:**
        - **Max Tokens: 80**
        - **Temperature: 0.7**
        - **Top-K: 25**
        """)
        
        # Fixed values - no sliders
        max_tokens = 80
        temperature = 0.7
        top_k = 25
        
        # Show current settings
        st.info(f"🎯 Current: {max_tokens} tokens | Temp: {temperature} | Top-K: {top_k}")
        
        # Optional: Advanced settings (collapsed by default)
        with st.expander("🔧 Advanced Settings (Optional)"):
            custom_max_tokens = st.slider("Max Tokens", min_value=20, max_value=150, value=80, step=10)
            custom_temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            custom_top_k = st.slider("Top-K", min_value=5, max_value=50, value=25, step=5)
            
            if st.button("Apply Custom Settings"):
                max_tokens = custom_max_tokens
                temperature = custom_temperature
                top_k = custom_top_k
                st.success("✅ Custom settings applied!")
        
        # Model info
        st.subheader("📊 Model Info")
        if 'checkpoint' in st.session_state:
            checkpoint = st.session_state.checkpoint
            config = st.session_state.config
            
            # Calculate model parameters
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            
            st.markdown(f"""
            <div class="sidebar-info">
            <strong>Parameters:</strong> {total_params/1e6:.2f}M<br>
            <strong>Layers:</strong> {config.n_layer}<br>
            <strong>Heads:</strong> {config.n_head}<br>
            <strong>Embedding:</strong> {config.n_embd}<br>
            <strong>Block Size:</strong> {config.block_size}<br>
            <strong>Best Val Loss:</strong> {checkpoint.get('best_val_loss', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}<br>
            <strong>Device:</strong> {st.session_state.device}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat controls
    if st.session_state.model_loaded:
        st.subheader("🗨️ Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.total_tokens_generated = 0
            st.success("Chat history cleared!")
        
        # Export chat
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="📥 Export Chat",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Main chat interface
if not st.session_state.model_loaded:
    st.info("🤖 Loading your university chatbot model...")
    st.markdown("""
    ### 📚 About Your University Chatbot:
    - **Specialized Knowledge**: Trained on university-specific content
    - **Instant Responses**: Get quick answers about admissions, courses, fees, and more
    - **24/7 Availability**: Always ready to help with your university questions
    
    ### 💡 What you can ask:
    - **"What are the admission requirements?"**
    - **"Tell me about tuition fees"**
    - **"What courses are available?"**
    - **"Where is the campus located?"**
    - **"How do I apply for admission?"**
    - **"What are the payment methods?"**
    """)
else:
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
        <strong>💬 Messages:</strong> {len(st.session_state.chat_history)}<br>
        <strong>🔤 Tokens Generated:</strong> {st.session_state.total_tokens_generated}
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 You</div>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-header">🎓 University Bot</div>
                    <div>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about university admissions, courses, fees, etc...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response = generate_response(
                st.session_state.model,
                st.session_state.tokenizer,
                user_input,
                max_tokens,
                temperature,
                top_k,
                st.session_state.device
            )
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "bot", "content": response})
        st.session_state.total_tokens_generated += len(st.session_state.tokenizer.encode_ordinary(response))
        
        # Rerun to update the display
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🎓 University Chatbot - Powered by Custom GPT Model</div>",
    unsafe_allow_html=True
)