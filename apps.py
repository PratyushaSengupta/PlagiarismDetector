import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from langdetect import detect

import re
from collections import defaultdict
from math import ceil

from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

from PIL import Image
import easyocr
import time

# === F4: Heatmap Imports ===
import matplotlib.pyplot as plt  # [F4 Added]
from sklearn.metrics.pairwise import cosine_similarity  # [F4 Added]

st.set_page_config(page_title="Document Plagiarism Analyzer")
# Basic Streamlit Styling with minimal CSS (for just the button)
st.markdown(
    """
    <style>
        /* Change main background */
        .stApp {
            background-color: #E6F7FF;  /* Light Blue */
        }

        .stButton>button {
            background-color: #FF6347;  /* Red Button */
            color: white;  /* Button text color */
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #FF4500;  /* Darker Red on Hover */
            color: white !important;  /* Force text to stay white on hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# === Simple Tokenizers (regex-based) ===
def sent_tokenize(text: str) -> list[str]:
    # Split on sentence-ending punctuation followed by whitespace
    return re.split(r'(?<=[.!?])\s+', text.strip())

def word_tokenize(sent: str) -> list[str]:
    # Simple whitespace split
    return sent.split()


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# === F4: Similarity Heatmap Function Added ===
def generate_similarity_heatmap(input_text, abstract_text, sbert_model):  # [Feature4]
    # Split texts into sentences
    input_sents = sent_tokenize(input_text)
    abstract_sents = sent_tokenize(abstract_text)
    # Embed sentences
    inp_embs = sbert_model.encode(input_sents, convert_to_numpy=True)
    abs_embs = sbert_model.encode(abstract_sents, convert_to_numpy=True)
    # Compute similarity matrix
    sim_mat = cosine_similarity(inp_embs, abs_embs)
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns = None
    try:
        import seaborn as sns  # local import
        sns.heatmap(sim_mat, annot=True, cmap="Blues",
                    xticklabels=[f"A{i+1}" for i in range(len(abstract_sents))],
                    yticklabels=[f"I{i+1}" for i in range(len(input_sents))],
                    ax=ax)
    except ImportError:
        ax.imshow(sim_mat, aspect='auto')
    ax.set_xlabel("Abstract Sentences")
    ax.set_ylabel("Input Sentences")
    st.pyplot(fig)

# === Load Data ===
@st.cache_data
def load_metadata():
    df = pd.read_csv("arxiv_cs_subset.csv")
    return df

@st.cache_resource
def load_embeddings():
    return np.load("specter_embeddings.npy")

@st.cache_resource
def load_faiss_index(embeddings):
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# === Load Models ===
@st.cache_resource
def load_models():
    sbert_model = SentenceTransformer('allenai-specter')
    mt_model_name = 'Helsinki-NLP/opus-mt-mul-en'
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name)
    return sbert_model, mt_model, tokenizer

# === Subdomain Mapping ===
subdomain_mapping = {
    "cs.AI": "Artificial Intelligence", "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision", "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases", "cs.LG": "Machine Learning", "cs.CR": "Cryptography and Security",
    "cs.NI": "Networking and Internet Architecture", "cs.SE": "Software Engineering",
    "cs.IR": "Information Retrieval", "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.HC": "Human-Computer Interaction", "cs.CY": "Computers and Society",
    "cs.RO": "Robotics", "cs.SI": "Social and Information Networks",
    "cs.PL": "Programming Languages", "cs.NE": "Neural and Evolutionary Computing",
    "cs.MM": "Multimedia", "cs.OS": "Operating Systems", "cs.CC": "Computational Complexity",
    "cs.CG": "Computational Geometry", "cs.MA": "Multiagent Systems",
    "cs.FL": "Formal Languages and Automata Theory", "cs.LO": "Logic in Computer Science",
    "cs.GR": "Graphics", "cs.SC": "Symbolic Computation", "cs.AR": "Hardware Architecture",
    "cs.ET": "Emerging Technologies", "cs.SY": "Systems and Control", "cs.MS": "Mathematical Software",
    "cs.CE": "Computational Engineering, Finance, and Science", "cs.PF": "Performance"
}

# === EasyOCR Reader ===
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_image(image):
    result = reader.readtext(np.array(image), detail=0)
    return ' '.join(result)
    
# === Preprocess Input ===
def preprocess_input_paragraph(text, tokenizer, mt_model):
    language = detect(text)
    if language != 'en':
        batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        with torch.no_grad():
            generated = mt_model.generate(**batch)
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translated
    return text

# === Hybrid Chunking Function ===
def chunk_text(text, max_tokens=60):
    """
    Splits text into semantically coherent chunks by:
    1. Sentence tokenization
    2. Merging sentences until reaching max_tokens
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_tokens = []
    token_count = 0

    for sent in sentences:
        tokens = word_tokenize(sent)
        if token_count + len(tokens) > max_tokens and current_chunk_tokens:
            chunks.append(' '.join(current_chunk_tokens))
            current_chunk_tokens = tokens
            token_count = len(tokens)
        else:
            current_chunk_tokens.extend(tokens)
            token_count += len(tokens)

    if current_chunk_tokens:
        chunks.append(' '.join(current_chunk_tokens))
    return chunks

# === Search Function with Chunk-Level Logic ===
def search_similar_paragraph(input_text, selected_category, df, embeddings, index, sbert_model,
                              sim_threshold=0.75, freq_ratio=0.3, top_k=3,
                              knn_k=5, full_para_threshold=0.90, high_para_threshold=0.94):
    # Step 1: Chunk input text
    chunks = chunk_text(input_text)
    if not chunks:
        return []

    # Step 2: Embed and normalize chunks
    chunk_embs = sbert_model.encode(chunks, convert_to_numpy=True).astype(np.float32)
    chunk_embs = normalize_embeddings(chunk_embs)

    # Step 3: FAISS search with top-k per chunk
    D, I = index.search(chunk_embs, k=knn_k)

    # Step 4: Count frequencies of abstracts above sim_threshold
    abstract_hits = defaultdict(int)
    abstract_scores = defaultdict(list)

    # **[Step 2]** Initialize storage for which chunks matched which abstract
    sentence_matches = defaultdict(list)      # **[Step 2]**

    for chunk_idx, (dist_list, idx_list) in enumerate(zip(D, I)):
        for score, idx in zip(dist_list, idx_list):
            if score >= sim_threshold:
                abstract_hits[idx] += 1
                abstract_scores[idx].append(score)
                # **[Step 2]** Record the actual chunk text that matched
                sentence_matches[idx].append(chunks[chunk_idx])   # **[Step 2]**

    # Step 5: Apply frequency filter
    total_chunks = len(chunks)
    min_hits = ceil(freq_ratio * total_chunks)
    final_ids = [aid for aid, count in abstract_hits.items() if count >= min_hits]

    # Step 6: Embed full paragraph and run FAISS search on it
    full_para_emb = sbert_model.encode([input_text], convert_to_numpy=True).astype(np.float32)
    full_para_emb = normalize_embeddings(full_para_emb)

    # Step 7: FAISS search for full paragraph
    D_full, I_full = index.search(full_para_emb, k=knn_k)

    # Step 8: Apply full-paragraph similarity thresholds
    full_para_matches = []
    for dist_list, idx_list in zip(D_full, I_full):
        for score, idx in zip(dist_list, idx_list):
            if score >= full_para_threshold:
                full_para_matches.append((idx, score))

    # Step 9: Combine chunk-level and full-paragraph results with adjusted weightings
    combined_results = {}
    for aid in abstract_hits:
        combined_results[aid] = {
            "score": np.mean(abstract_scores[aid]) * 0.7,
            "full_para_score": 0,
            "is_full_match": False
        }

    for aid, score in full_para_matches:
        if score >= high_para_threshold:
            combined_results.setdefault(aid, {"score":0,"full_para_score":0,"is_full_match":False})
            combined_results[aid]["full_para_score"] = score
            combined_results[aid]["is_full_match"] = True
        else:
            combined_results.setdefault(aid, {"score":0,"full_para_score":0,"is_full_match":False})
            combined_results[aid]["full_para_score"] += score * 0.3

    # Step 10: Prepare the final results with weighted scores
    results = []
    for aid, values in combined_results.items():
        row = df.iloc[aid]
        if values["is_full_match"]:
            final_score = values["full_para_score"]
        else:
            final_score = values["score"] + values["full_para_score"]

        if selected_category == "All" or row["categories"] == selected_category:
            results.append({
                "Title": row["title"],
                "Abstract": row["abstract"],
                "Subdomain": subdomain_mapping.get(row["categories"], row["categories"]),
                "Similarity Score": final_score,
                # **[Step 2]** include the actual matched chunks for explainability
                "Matched Sentences": sentence_matches.get(aid, [])   # **[Step 2]**
            })

    # Step 11: Sort by similarity
    results = sorted(results, key=lambda x: x["Similarity Score"], reverse=True)[:top_k]
    return results




# === Streamlit UI Implementation ===
st.title("AI-Based Plagiarism Detection (arXiv CS Papers)")
st.markdown("Detect similarity between your input (typed or image-based) and arXiv CS abstracts using SBERT + FAISS + MarianMT.")

# Load resources
df = load_metadata()
embeddings = load_embeddings()
index = load_faiss_index(embeddings)
sbert_model, mt_model, tokenizer = load_models()

# Subdomain filter
subdomain_options = ["All"] + [v for k, v in subdomain_mapping.items()]
selected_subdomain_full = st.selectbox("Filter by Subdomain", subdomain_options)
reverse_subdomain_map = {v: k for k, v in subdomain_mapping.items()}
selected_category = reverse_subdomain_map.get(selected_subdomain_full, "All")

# Accept both image and text inputs
uploaded_image = st.file_uploader("Upload an image containing a paragraph (optional)", type=["png", "jpg", "jpeg"])
input_text = st.text_area("Or type your paragraph (any language):", height=300)

# Similarity check with progress
def similarity_check_with_progress(uploaded_image=None, input_text=None):
    progress_bar = st.progress(0)
    message_placeholder = st.empty()
    
    # Stage 1: Extract text
    if uploaded_image:
        message_placeholder.markdown("**Extracting text from image...**")
        progress_bar.progress(20)
        image = Image.open(uploaded_image)
        extracted_text = extract_text_from_image(image)
        time.sleep(1)
        message_placeholder.markdown("**Text extracted.**")
        st.info(extracted_text)
    elif input_text.strip():
        extracted_text = input_text
    else:
        st.warning("Please provide either an image or some text.")
        return None

    if not extracted_text.strip():
        st.warning("No text detected in the uploaded image.")
        return None

    # Stage 2: Preprocessing
    message_placeholder.markdown("**Preprocessing the text...**")
    progress_bar.progress(50)
    translated = preprocess_input_paragraph(extracted_text, tokenizer, mt_model)
    time.sleep(1)

    # Stage 3: Similarity Search
    message_placeholder.markdown("**Performing similarity check...**")
    progress_bar.progress(80)
    results = search_similar_paragraph(translated, selected_category, df, embeddings, index, sbert_model)
    time.sleep(1)

    progress_bar.progress(100)
    message_placeholder.markdown("**Processing complete.**")
    return results

# Trigger on button press
if st.button("Check Similarity"):
    results = similarity_check_with_progress(uploaded_image=uploaded_image, input_text=input_text)

    if results:
        st.subheader("Top Matches:")
        for i, r in enumerate(results, 1):
            st.markdown(f"{i}- **Title**: {r['Title']}")
            st.markdown(f"*Subdomain: {r['Subdomain']}*")
            st.markdown(f"**Abstract:** {r['Abstract']}")
            st.markdown(f"**Similarity Score:** {r['Similarity Score']:.4f}")

            st.markdown(f"**Matched Sentences ({len(r['Matched Sentences'])}/{len(chunk_text(input_text))})**:")
            for sent in r['Matched Sentences']:
                st.write(f"- {sent}")

            # === F4: Generate Heatmap Visualization ===
            st.markdown("**Similarity Heatmap:**")  # [F4 Added]
            generate_similarity_heatmap(input_text, r['Abstract'], sbert_model)  # [F4 Added]
                
            if r['Similarity Score'] > 0.8:
                st.markdown("**Plagiarized Document!!!!** ❌")
            else:
                st.markdown("**Not Plagiarized** ✔")
            st.markdown("---")
    elif uploaded_image or input_text.strip():
        st.info("No similar results found.")
