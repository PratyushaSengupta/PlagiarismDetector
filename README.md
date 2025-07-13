# AI-Based Plagiarism Detection System
This project detects plagiarism in the scientific/ research community and compares input paragraphs to the abstracts of research papers from arxiv cs datastore.

## Features
-- **Language Detection** using `langdetect` to identify the language of input text  
- **Cross-Language Translation**: Translates non-English text to English using MarianMT (helps catch obfuscated cross-language plagiarism)
- **Paraphrased Content Detection**: Using SBERT allenai-specter`, optimized for scientific texts (BERT Models capture semantic meanings)
- **Hybrid Semantic Matching**: Sentence level and paragraph level match/ comparison for better average
- **FAISS Indexing** for fast similarity search  
- **Visual Heatmaps** and top-match scoring for interpretability  
- **OCR Support** for text in screenshots or scanned images using EasyOCR
