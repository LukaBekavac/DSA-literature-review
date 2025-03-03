import os
import fitz  # PyMuPDF
from keybert import KeyBERT
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class PDFExtractor:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def extract_texts(self):
        """Extracts text from all PDF files in the specified folder."""
        documents = []
        filenames = []
        for file in os.listdir(self.folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, file)
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text("text") + " "
                documents.append(text)
                filenames.append(file)
        return documents, filenames

class KeywordExtractor:
    def __init__(self):
        self.model = KeyBERT()
        # Initialize the sentence transformer separately for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Custom stop words specific to your domain
        self.custom_stop_words = {
            'said', 'would', 'one', 'also', 'may', 'article', 'regulation',
            'european', 'union', 'eu', 'commission', 'dsa', 'platform',
            'page', 'paragraph', 'section'
        }
        
        # Relevant terms to guide keyword extraction
        self.relevant_terms = [
            "data access",
            "data sharing",
            "data collection",
            "research data",
            "data protection",
            "privacy",
            "transparency",
            "compliance",
            "researcher access",
            "data quality",
            "data format",
            "technical barriers",
            "legal barriers",
            "data security",
            "confidentiality",
            "data governance",
            "data standards",
            "data infrastructure",
            "data requirements",
            "data limitations"
        ]

    def extract_keywords(self, text: str, top_n: int = 15, diversity: float = 0.7):
        """
        Extracts keywords with improved focus on data access topics.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            diversity: Diversity of keywords (0-1), higher means more diverse
        """
        # Extract keywords using MMR for diversity
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),  # Allow up to trigrams
            stop_words=list(self.custom_stop_words),
            use_maxsum=True,
            nr_candidates=30,
            top_n=top_n,
            diversity=diversity,
            use_mmr=True,  # Use Maximal Marginal Relevance
            seed_keywords=self.relevant_terms
        )
        
        # Filter and score keywords based on relevance to data access topics
        filtered_keywords = []
        for keyword, score in keywords:
            # Calculate similarity to relevant terms
            relevance_scores = [
                self.calculate_similarity(keyword, term)
                for term in self.relevant_terms
            ]
            max_relevance = max(relevance_scores)
            
            # Combine original score with relevance score
            combined_score = (score + max_relevance) / 2
            
            if combined_score > 0.2:  # Minimum relevance threshold
                filtered_keywords.append((keyword, float(combined_score)))
        
        # Sort by combined score
        filtered_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_keywords[:top_n]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Use the sentence transformer model directly
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

def save_results(results: dict, output_dir: str = "results"):
    """Save the keyword extraction results to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"keywords_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_file

if __name__ == "__main__":
    pdf_folder = "/Users/lukabekavac/Desktop/HSG/PhD/Supervision/Mathis/DSA-letters"
    
    # Extract text from PDFs
    pdf_extractor = PDFExtractor(pdf_folder)
    documents, filenames = pdf_extractor.extract_texts()

    # Extract keywords
    keyword_extractor = KeywordExtractor()
    results = {}
    
    for filename, text in zip(filenames, documents):
        keywords = keyword_extractor.extract_keywords(text, top_n=15, diversity=0.7)
        results[filename] = {
            "keywords": [
                {
                    "word": keyword,
                    "score": float(score),
                    "category": "data_access" if any(term in keyword.lower() for term in ["data", "access", "privacy", "security"]) else "other"
                }
                for keyword, score in keywords
            ]
        }
        
        # Print results to console
        print(f"\nKeywords for {filename}:")
        for kw, score in keywords:
            print(f"- {kw}: {score:.4f}")

    # Save results to file
    output_file = save_results(results)
    print(f"\nResults saved to: {output_file}")
