import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from models import CodingDocument, CodingKeyword, CodingLabel, SystemPrompt, init_db
from sqlalchemy.exc import IntegrityError

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

def check_document_exists(filename: str, session) -> bool:
    """Check if a document has already been processed."""
    return session.query(CodingDocument).filter(CodingDocument.filename == filename).first() is not None

class GPTCodingAnalyzer:
    def __init__(self, session):
        self.client = OpenAI()
        self.model = "gpt-4"
        self.session = session
        self._init_system_prompt()
        
    def _init_system_prompt(self):
        """Initialize or retrieve the system prompt."""
        system_prompt = self.session.query(SystemPrompt).filter_by(
            name='coding_analysis',
            is_active=True
        ).first()
        
        if not system_prompt:
            system_prompt = SystemPrompt(
                name='coding_analysis',
                content="""You are a research assistant specialized in analyzing documents from the European Commission. Your task is to make a systematic thematic analysis of documents using the Braun and Clark methodology. Braun and Clarke's thematic analysis is essentially a process for identifying patterns or themes within qualitative data. For our research we will only focus on the first two steps of the methodology. What you're doing first is familiarizing yourself with the data. Then identifying specific pieces of content relevant to your research topic: you're breaking the data down into smaller, more meaningful pieces. This means looking really closely at it and essentially sticking labels on particular pieces of the content. Those labels are codes. Note that you should process one document at a time, and that a label can be assigned to multiple pieces of content, whether within the same document or across different documents."""
            )
            self.session.add(system_prompt)
            self.session.commit()
        
        self.system_prompt = system_prompt
        
    def get_existing_labels(self):
        """Get all existing labels ordered by usage count."""
        labels = self.session.query(CodingLabel).order_by(CodingLabel.usage_count.desc()).all()
        return [{"name": label.name, "description": label.description} for label in labels]

    def extract_keywords(self, text: str, max_tokens: int = 8000) -> list:
        """Extracts keywords using OpenAI's GPT-4."""
        try:
            if len(text) > max_tokens:
                text = text[:max_tokens] + "..."

            # Get existing labels for context
            existing_labels = self.get_existing_labels()
            labels_context = ""
            if existing_labels:
                labels_context = "\nExisting labels (use these if applicable, create new ones only if necessary):\n"
                for label in existing_labels:
                    labels_context += f"- {label['name']}: {label['description']}\n"

            user_prompt = f"""The document here are about data access and digital platforms regulation. You are focusing on the step two of the Braun and Clark methodology: the emphasis here is on identifying relevant pieces of content, keywords, and key phrases within the documents and assigning labels to that content. The purpose of this stage is to organize the data and lay the foundations for later analysis.

{labels_context}

IMPORTANT: You must respond with a valid JSON object containing a 'pieces_of_content' array. Each item in the array must have these exact fields:
- "content": the extracted content (string)
- "score": relevance score between 0 and 1 (number)
- "label": use an existing label if applicable, or create a new descriptive one (string)
- "explanation": brief explanation of relevance (string)

Example response format:
{{
    "pieces_of_content": [
        {{
            "content": "example text",
            "score": 0.85,
            "label": "example label",
            "explanation": "example explanation"
        }}
    ]
}}

Analyze the following text and provide your response in the exact JSON format specified above:

"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt.content},
                    {"role": "user", "content": user_prompt + text}
                ],
                temperature=0
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                print("\nRaw LLM response:")
                print(json.dumps(result, indent=2))
                return result.get('pieces_of_content', [])
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print("Raw response:", response.choices[0].message.content)
                return []

        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            return []

def get_or_create_label(session, label_name: str, explanation: str = None) -> CodingLabel:
    """Get existing label or create a new one."""
    # Trim whitespace and convert to lower case for consistent matching
    label_name = label_name.strip().lower()
    
    label = session.query(CodingLabel).filter(func.lower(CodingLabel.name) == label_name).first()
    
    if label:
        # Update usage count
        label.usage_count += 1
    else:
        # Create new label
        label = CodingLabel(
            name=label_name,
            description=explanation,
            usage_count=1
        )
        session.add(label)
        session.commit()  # Commit after adding a new label to save it to the database
    
    return label

def save_to_db(filename: str, keywords: list, session):
    """Save the analysis results to the database."""
    try:
        # Create new document record
        doc = CodingDocument(filename=filename)
        session.add(doc)
        session.flush()  # Get the document ID
        
        saved_keywords = []
        
        # Add keywords
        for kw in keywords:
            label_name = kw.get('label', '')
            
            # Get or create label
            label = get_or_create_label(session, label_name, kw.get('explanation', ''))
            
            keyword = CodingKeyword(
                document=doc,
                content=kw.get('content', ''),
                score=float(kw.get('score', 0.0)),
                label=label,
                explanation=kw.get('explanation', '')
            )
            session.add(keyword)
            saved_keywords.append(keyword)
        
        session.commit()
        doc.keywords = saved_keywords  # Update the keywords relationship
        return doc
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {str(e)}")
        return None

if __name__ == "__main__":
    pdf_folder = "/Users/lukabekavac/Desktop/HSG/PhD/Supervision/Mathis/Letters (Call for Evidence)"
    
    # Initialize database
    engine = init_db()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Extract text from PDFs
    pdf_extractor = PDFExtractor(pdf_folder)
    documents, filenames = pdf_extractor.extract_texts()

    # Extract keywords using GPT
    analyzer = GPTCodingAnalyzer(session)
    
    # Keep track of processed and skipped files
    processed_files = []
    skipped_files = []
    
    for filename, text in zip(filenames, documents):
        print(f"\nChecking: {filename}")
        
        # Check if document already exists in database
        if check_document_exists(filename, session):
            print(f"Skipping {filename} - already processed")
            skipped_files.append(filename)
            continue
            
        print(f"Processing: {filename}")
        keywords = analyzer.extract_keywords(text)
        
        # Save to database
        doc = save_to_db(filename, keywords, session)
        
        if doc:
            # Print results to console
            print(f"\nKeywords for {filename}:")
            for keyword in doc.keywords:
                print(f"- {keyword.content} ({keyword.label.name}): {keyword.score:.2f}")
                print(f"  {keyword.explanation}")
            
            print(f"Results for {filename} saved to database.")
            processed_files.append(filename)

    # Print summary
    print("\nProcessing complete.")
    print(f"Processed files ({len(processed_files)}): {', '.join(processed_files)}")
    print(f"Skipped files ({len(skipped_files)}): {', '.join(skipped_files)}")
    session.close() 