import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from models import Document, Keyword, Label, init_db
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
    return session.query(Document).filter(Document.filename == filename).first() is not None

class GPTKeywordExtractor:
    def __init__(self, session):
        self.client = OpenAI()
        self.model = "gpt-4o-mini"
        self.session = session
        
    def get_existing_labels(self):
        """Get all existing labels ordered by usage count."""
        labels = self.session.query(Label).order_by(Label.usage_count.desc()).all()
        return [{"name": label.name, "description": label.description} for label in labels]
        
    def get_system_prompt(self):
        """Generate system prompt including existing labels."""
        existing_labels = self.get_existing_labels()
        
        labels_context = ""
        if existing_labels:
            labels_context = "\nExisting labels (use these if applicable, create new ones only if necessary):\n"
            for label in existing_labels:
                labels_context += f"- {label['name']}: {label['description']}\n"
        
        return f"""You are a research assistant specialized in analyzing documents about data access and digital platforms regulation.
        Your task is to make a systematic analysis of documents using the Braun and Clark methodology. You are now focusing on the step 2 of the Braun and Clark methodology: the emphasis here is on identifying relevant pieces of content, keywords, and key phrases within the documents and assigning labels to that content. The purpose of this stage is to organize the data and lay the foundations for later analysis.
        
        IMPORTANT: Each label can only be used ONCE per document. If you find multiple pieces of content that could use the same label, either:
        1. Choose the most representative piece for that label, or
        2. Create a more specific variant of the label to differentiate them.
        
        For each piece of content:
        - Provide a relevance score (0-1)
        - Assign an existing label if applicable, or create a new one if necessary
        - Include a brief explanation of why it's relevant
        {labels_context}
        Format your response as a JSON object with a 'pieces_of_content' array containing objects with:
        - content: the extracted content
        - score: relevance score (0-1)
        - label: use an existing label if applicable, or create a new descriptive one (remember: each label can only be used once per document)
        - explanation: brief explanation of relevance to data access research
        """

    def extract_keywords(self, text: str, max_tokens: int = 8000) -> list:
        """Extracts keywords using OpenAI's GPT with temperature=0 for consistency."""
        try:
            if len(text) > max_tokens:
                text = text[:max_tokens] + "..."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": f"Analyze this text and extract relevant keywords and issues related to data access:\n\n{text}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print("\nRaw LLM response:")
            print(json.dumps(result, indent=2))
            
            return result.get('pieces_of_content', [])

        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            return []

def get_or_create_label(session, label_name: str, explanation: str = None) -> Label:
    """Get existing label or create a new one."""
    label = session.query(Label).filter(func.lower(Label.name) == func.lower(label_name)).first()
    
    if label:
        # Update usage count
        label.usage_count += 1
    else:
        # Create new label
        label = Label(
            name=label_name,
            description=explanation,
            usage_count=1
        )
        session.add(label)
    
    return label

def save_to_db(filename: str, keywords: list, session):
    """Save the analysis results to the database."""
    try:
        # Create new document record
        doc = Document(filename=filename)
        session.add(doc)
        session.flush()  # Get the document ID
        
        # Track used labels for this document
        used_labels = set()
        saved_keywords = []
        
        # Add keywords
        for kw in keywords:
            label_name = kw.get('label', '')
            
            # Skip if label was already used in this document
            if label_name.lower() in used_labels:
                print(f"Warning: Label '{label_name}' was already used in this document. Skipping duplicate.")
                continue
            
            # Get or create label
            label = get_or_create_label(session, label_name, kw.get('explanation', ''))
            
            keyword = Keyword(
                document=doc,
                content=kw.get('content', ''),
                score=float(kw.get('score', 0.0)),
                label=label,
                explanation=kw.get('explanation', '')
            )
            session.add(keyword)
            saved_keywords.append(keyword)
            used_labels.add(label_name.lower())
        
        session.commit()
        doc.keywords = saved_keywords  # Update the keywords relationship
        return doc
    except IntegrityError as e:
        session.rollback()
        print(f"Error: Duplicate label used in document: {str(e)}")
        return None
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {str(e)}")
        return None

def save_to_json(results: dict, output_dir: str = "results"):
    """Save the keyword extraction results to a JSON file."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(output_dir, f"llm_keywords_{timestamp}.json")
        
        existing_results = {}
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        
        existing_results.update(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
        
        return output_file
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
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
    keyword_extractor = GPTKeywordExtractor(session)
    
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
        keywords = keyword_extractor.extract_keywords(text)
        
        # Save to database
        doc = save_to_db(filename, keywords, session)
        
        if doc:
            # Save to JSON after each file
            results = {filename: {"pieces_of_content": keywords}}
            json_file = save_to_json(results)
            
            # Print results to console
            print(f"\nKeywords for {filename}:")
            for keyword in doc.keywords:
                print(f"- {keyword.content} ({keyword.label.name}): {keyword.score:.2f}")
                print(f"  {keyword.explanation}")
            
            if json_file:
                print(f"Results for {filename} saved to JSON file: {json_file}")
            print(f"Results for {filename} saved to database.")
            processed_files.append(filename)

    # Print summary
    print("\nProcessing complete.")
    print(f"Processed files ({len(processed_files)}): {', '.join(processed_files)}")
    print(f"Skipped files ({len(skipped_files)}): {', '.join(skipped_files)}")
    session.close()

