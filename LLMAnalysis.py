import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from openai import OpenAI

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

class GPTKeywordExtractor:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4-turbo-preview"  # or your preferred model version
        
        self.system_prompt = """
        You are a research assistant specialized in analyzing documents about data access and digital platforms regulation.
        Your task is to extract relevant keywords and key phrases from the provided text, focusing on:
        1. Data access requirements and problems
        2. Technical and legal barriers
        3. Privacy and security concerns
        4. Implementation challenges
        5. Stakeholder concerns

        For each keyword/phrase:
        - Provide a relevance score (0-1)
        - Categorize it into one of these categories:
          * data_access: Issues directly related to accessing platform data
          * technical: Technical barriers or implementation challenges
          * legal: Legal requirements, compliance issues
          * privacy: Data protection and privacy concerns
          * stakeholder: Concerns from different stakeholders
          * other: Other relevant issues
        - Include a brief explanation of why it's relevant

        Format your response as a JSON object with a 'keywords' array containing objects with:
        - keyword: the extracted term
        - score: relevance score
        - category: one of the categories listed above
        - explanation: brief explanation of relevance to data access research
        """

    def extract_keywords(self, text: str, max_tokens: int = 8000) -> list:
        """
        Extracts keywords using OpenAI's GPT-4 with temperature=0 for consistency.
        """
        try:
            # Truncate text if needed to fit within token limits
            if len(text) > max_tokens:
                text = text[:max_tokens] + "..."

            response = self.client.chat.completions.create(
                model=self.model,  # Use the model specified in init
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this text and extract relevant keywords and issues related to data access:\n\n{text}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return result.get('keywords', [])

        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            return []

def save_results(results: dict, output_dir: str = "results"):
    """Save the keyword extraction results to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"llm_keywords_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_file

if __name__ == "__main__":
    pdf_folder = "/Users/lukabekavac/Desktop/HSG/PhD/Supervision/Mathis/DSA-letters"
    
    # Extract text from PDFs
    pdf_extractor = PDFExtractor(pdf_folder)
    documents, filenames = pdf_extractor.extract_texts()

    # Extract keywords using GPT-4
    keyword_extractor = GPTKeywordExtractor()
    results = {}
    
    for filename, text in zip(filenames, documents):
        print(f"\nProcessing: {filename}")
        keywords = keyword_extractor.extract_keywords(text)
        results[filename] = {
            "keywords": keywords
        }
        
        # Print results to console
        print(f"\nKeywords for {filename}:")
        for kw in keywords:
            print(f"- {kw['keyword']} ({kw['category']}): {kw['score']:.2f}")
            print(f"  {kw['explanation']}")

    # Save results to file
    output_file = save_results(results)
    print(f"\nResults saved to: {output_file}")