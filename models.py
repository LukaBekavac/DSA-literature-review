from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, DateTime, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Label(Base):
    __tablename__ = 'labels'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)  # Optional description of what this label represents
    created_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=0)  # Track how often this label is used

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    keywords = relationship("Keyword", back_populates="document")

class Keyword(Base):
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    label_id = Column(Integer, ForeignKey('labels.id'))
    explanation = Column(Text)
    
    document = relationship("Document", back_populates="keywords")
    label = relationship("Label")
    
    # Ensure each label is used only once per document
    __table_args__ = (
        UniqueConstraint('document_id', 'label_id', name='unique_label_per_document'),
    )

class CodingLabel(Base):
    __tablename__ = 'coding_labels'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)  # Optional description of what this label represents
    created_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=0)  # Track how often this label is used

class CodingDocument(Base):
    __tablename__ = 'coding_documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    keywords = relationship("CodingKeyword", back_populates="document")

class CodingKeyword(Base):
    __tablename__ = 'coding_keywords'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('coding_documents.id'))
    content = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    label_id = Column(Integer, ForeignKey('coding_labels.id'))
    explanation = Column(Text)
    
    document = relationship("CodingDocument", back_populates="keywords")
    label = relationship("CodingLabel")

class SystemPrompt(Base):
    __tablename__ = 'system_prompts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)

def init_db(db_url='sqlite:///llm_analysis.db'):
    """Initialize the database, creating all tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine 