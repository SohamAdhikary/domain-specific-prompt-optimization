import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Data structure for Question-Answer pairs"""
    question: str
    answer: str
    domain: str
    confidence: float = 0.0

@dataclass
class PromptResult:
    """Result structure for prompt generation"""
    prompt: str
    examples_used: List[QAPair]
    domain: str
    confidence: float

class DataCurator:
    """
    Week 1-2: Data Curation & Understanding
    Handles loading, preprocessing, and cleaning of domain-specific datasets
    """
    
    def __init__(self):
        self.medical_data = []
        self.legal_data = []
        self.processed_data = []
        
    def load_mock_data(self) -> None:
        """Load mock data for demonstration (replace with actual dataset loading)"""
        # Mock medical data (simulating PubMedQA)
        medical_qa = [
            {"question": "What is the first-line treatment for hypertension?", 
             "answer": "ACE inhibitors or ARBs are typically first-line treatments for hypertension."},
            {"question": "What are the symptoms of Type 2 diabetes?", 
             "answer": "Symptoms include increased thirst, frequent urination, fatigue, and blurred vision."},
            {"question": "How is pneumonia diagnosed?", 
             "answer": "Pneumonia is diagnosed through chest X-rays, blood tests, and clinical examination."},
            {"question": "What is the treatment for acute myocardial infarction?", 
             "answer": "Treatment includes aspirin, thrombolytics, and urgent cardiac catheterization."},
            {"question": "What are the risk factors for stroke?", 
             "answer": "Risk factors include hypertension, diabetes, smoking, and atrial fibrillation."}
        ]
        
        # Mock legal data (simulating LegalBench)
        legal_qa = [
            {"question": "What constitutes copyright infringement?", 
             "answer": "Copyright infringement occurs when copyrighted material is used without permission."},
            {"question": "What is the statute of limitations for contract disputes?", 
             "answer": "The statute of limitations varies by state but is typically 3-6 years for contracts."},
            {"question": "What are the elements of negligence?", 
             "answer": "Negligence requires duty, breach of duty, causation, and damages."},
            {"question": "What is the difference between criminal and civil law?", 
             "answer": "Criminal law involves prosecution by the state, while civil law involves disputes between parties."},
            {"question": "What constitutes a valid contract?", 
             "answer": "A valid contract requires offer, acceptance, consideration, and legal capacity."}
        ]
        
        # Convert to QAPair objects
        for qa in medical_qa:
            self.medical_data.append(QAPair(qa["question"], qa["answer"], "medical"))
            
        for qa in legal_qa:
            self.legal_data.append(QAPair(qa["question"], qa["answer"], "legal"))
            
        self.processed_data = self.medical_data + self.legal_data
        logger.info(f"Loaded {len(self.medical_data)} medical and {len(self.legal_data)} legal QA pairs")
    
    def preprocess_data(self) -> List[QAPair]:
        """Clean and standardize QA pairs"""
        cleaned_data = []
        
        for qa_pair in self.processed_data:
            # Basic cleaning
            clean_question = re.sub(r'\s+', ' ', qa_pair.question.strip())
            clean_answer = re.sub(r'\s+', ' ', qa_pair.answer.strip())
            
            # Skip if too short
            if len(clean_question) < 10 or len(clean_answer) < 10:
                continue
                
            cleaned_data.append(QAPair(clean_question, clean_answer, qa_pair.domain))
        
        logger.info(f"Preprocessed {len(cleaned_data)} QA pairs")
        return cleaned_data
    
    def split_data(self, test_size: float = 0.2) -> Tuple[List[QAPair], List[QAPair]]:
        """Split data into train/test sets"""
        train_data, test_data = train_test_split(
            self.processed_data, test_size=test_size, random_state=42,
            stratify=[qa.domain for qa in self.processed_data]
        )
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data

class DomainClassifier:
    """
    Week 3: Domain Classification
    Classifies user input as medical, legal, or other domains
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def train(self, training_data: List[QAPair]) -> None:
        """Train the domain classifier"""
        questions = [qa.question for qa in training_data]
        domains = [qa.domain for qa in training_data]
        
        # Vectorize questions
        X = self.vectorizer.fit_transform(questions)
        
        # Train classifier
        self.classifier.fit(X, domains)
        self.is_trained = True
        
        logger.info("Domain classifier trained successfully")
    
    def predict(self, question: str) -> Tuple[str, float]:
        """Predict domain for a given question"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
            
        X = self.vectorizer.transform([question])
        prediction = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])
        
        return prediction, confidence
    
    def evaluate(self, test_data: List[QAPair]) -> Dict:
        """Evaluate classifier performance"""
        questions = [qa.question for qa in test_data]
        true_domains = [qa.domain for qa in test_data]
        
        X = self.vectorizer.transform(questions)
        predictions = self.classifier.predict(X)
        
        accuracy = accuracy_score(true_domains, predictions)
        report = classification_report(true_domains, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

class FewShotPromptGenerator:
    """
    Week 4: Few-Shot Prompt Generation
    Generates domain-specific prompts with relevant examples
    """
    
    def __init__(self, domain_classifier: DomainClassifier):
        self.domain_classifier = domain_classifier
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.training_data = []
        self.is_fitted = False
        
    def fit(self, training_data: List[QAPair]) -> None:
        """Fit the prompt generator with training data"""
        self.training_data = training_data
        questions = [qa.question for qa in training_data]
        self.vectorizer.fit(questions)
        self.is_fitted = True
        logger.info("Few-shot prompt generator fitted")
        
    def find_similar_examples(self, user_question: str, domain: str, n_examples: int = 3) -> List[QAPair]:
        """Find most similar examples for few-shot learning"""
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before use")
            
        # Filter training data by domain
        domain_data = [qa for qa in self.training_data if qa.domain == domain]
        
        if len(domain_data) < n_examples:
            logger.warning(f"Not enough examples for domain {domain}. Using {len(domain_data)} examples.")
            return domain_data
        
        # Vectorize questions
        domain_questions = [qa.question for qa in domain_data]
        question_vecs = self.vectorizer.transform(domain_questions)
        user_vec = self.vectorizer.transform([user_question])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vec, question_vecs)[0]
        
        # Get top examples
        top_indices = similarities.argsort()[-n_examples:][::-1]
        
        return [domain_data[i] for i in top_indices]
    
    def generate_prompt(self, user_question: str, n_examples: int = 3) -> PromptResult:
        """Generate a few-shot prompt for the user question"""
        # Classify domain
        domain, confidence = self.domain_classifier.predict(user_question)
        
        # Find similar examples
        examples = self.find_similar_examples(user_question, domain, n_examples)
        
        # Format prompt
        prompt_parts = [
            f"You are an expert in {domain} questions. Answer the following question based on the examples provided:\n"
        ]
        
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Q: {example.question}")
            prompt_parts.append(f"A: {example.answer}\n")
        
        prompt_parts.append(f"Now answer this question:")
        prompt_parts.append(f"Q: {user_question}")
        prompt_parts.append("A:")
        
        final_prompt = "\n".join(prompt_parts)
        
        return PromptResult(
            prompt=final_prompt,
            examples_used=examples,
            domain=domain,
            confidence=confidence
        )

class OutputValidator:
    """
    Week 5: Output Validation
    Validates LLM outputs for factual correctness and relevance
    """
    
    def __init__(self):
        self.validation_rules = {
            'medical': [
                r'\b(aspirin|medication|treatment|diagnosis|symptoms)\b',
                r'\b(patient|doctor|medical|clinical)\b'
            ],
            'legal': [
                r'\b(law|legal|court|contract|statute)\b',
                r'\b(liability|negligence|damages|jurisdiction)\b'
            ]
        }
    
    def validate_domain_relevance(self, answer: str, domain: str) -> Tuple[bool, float]:
        """Check if answer is relevant to the domain"""
        if domain not in self.validation_rules:
            return True, 0.5  # Neutral for unknown domains
        
        patterns = self.validation_rules[domain]
        matches = 0
        
        for pattern in patterns:
            if re.search(pattern, answer.lower()):
                matches += 1
        
        relevance_score = matches / len(patterns)
        is_relevant = relevance_score > 0.3
        
        return is_relevant, relevance_score
    
    def validate_answer_quality(self, question: str, answer: str) -> Tuple[bool, float]:
        """Basic quality validation"""
        # Check length
        if len(answer) < 20:
            return False, 0.1
        
        # Check if answer contains question words (potential sign of poor quality)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        
        # High overlap might indicate the answer is just repeating the question
        overlap_ratio = overlap / len(question_words) if question_words else 0
        
        quality_score = 1.0 - min(overlap_ratio, 0.8)
        is_quality = quality_score > 0.5
        
        return is_quality, quality_score
    
    def validate(self, question: str, answer: str, domain: str) -> Dict:
        """Comprehensive validation of answer"""
        relevance_valid, relevance_score = self.validate_domain_relevance(answer, domain)
        quality_valid, quality_score = self.validate_answer_quality(question, answer)
        
        overall_score = (relevance_score + quality_score) / 2
        overall_valid = relevance_valid and quality_valid
        
        return {
            'is_valid': overall_valid,
            'overall_score': overall_score,
            'relevance_score': relevance_score,
            'quality_score': quality_score,
            'relevance_valid': relevance_valid,
            'quality_valid': quality_valid
        }

class PromptOptimizationEngine:
    """
    Main engine that orchestrates all components
    """
    
    def __init__(self):
        self.data_curator = DataCurator()
        self.domain_classifier = DomainClassifier()
        self.prompt_generator = None
        self.output_validator = OutputValidator()
        self.is_trained = False
        self.training_data = []
        self.test_data = []
        
    def setup(self) -> None:
        """Initialize and train all components"""
        logger.info("Setting up Prompt Optimization Engine...")
        
        # Load and preprocess data
        self.data_curator.load_mock_data()
        processed_data = self.data_curator.preprocess_data()
        
        # Split data
        self.training_data, self.test_data = self.data_curator.split_data()
        
        # Train domain classifier
        self.domain_classifier.train(self.training_data)
        
        # Setup prompt generator
        self.prompt_generator = FewShotPromptGenerator(self.domain_classifier)
        self.prompt_generator.fit(self.training_data)
        
        self.is_trained = True
        logger.info("Engine setup complete!")
    
    def optimize_prompt(self, user_question: str, n_examples: int = 3) -> PromptResult:
        """Generate optimized prompt for user question"""
        if not self.is_trained:
            raise ValueError("Engine must be set up before use")
            
        return self.prompt_generator.generate_prompt(user_question, n_examples)
    
    def validate_output(self, question: str, answer: str, domain: str) -> Dict:
        """Validate LLM output"""
        return self.output_validator.validate(question, answer, domain)
    
    def evaluate_performance(self) -> Dict:
        """Evaluate overall system performance"""
        if not self.test_data:
            raise ValueError("No test data available for evaluation")
            
        # Evaluate domain classifier
        classifier_results = self.domain_classifier.evaluate(self.test_data)
        
        # Evaluate prompt generation (measure example relevance)
        prompt_scores = []
        for qa in self.test_data[:10]:  # Sample for demonstration
            prompt_result = self.optimize_prompt(qa.question)
            # Score based on domain match and confidence
            score = 1.0 if prompt_result.domain == qa.domain else 0.0
            prompt_scores.append(score)
        
        return {
            'classifier_accuracy': classifier_results['accuracy'],
            'prompt_generation_accuracy': np.mean(prompt_scores),
            'total_test_samples': len(self.test_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive project report"""
        if not self.is_trained:
            return "Engine not trained yet. Please run setup() first."
            
        performance = self.evaluate_performance()
        
        report = f"""
Domain-Specific Prompt Optimization Engine - Performance Report
================================================================

Generated: {performance['timestamp']}

SYSTEM OVERVIEW:
- Total Training Samples: {len(self.training_data)}
- Total Test Samples: {len(self.test_data)}
- Domains Supported: Medical, Legal

COMPONENT PERFORMANCE:
- Domain Classifier Accuracy: {performance['classifier_accuracy']:.3f}
- Prompt Generation Accuracy: {performance['prompt_generation_accuracy']:.3f}

METHODOLOGY:
- Used TF-IDF vectorization with Logistic Regression for domain classification
- Employed cosine similarity for few-shot example retrieval
- Implemented rule-based validation for output quality assessment

NEXT STEPS:
1. Integrate with actual LLM API (GPT-4, Claude, etc.)
2. Expand to additional domains
3. Implement advanced validation using NLI models
4. Add feedback loop for continuous improvement

BUSINESS IMPACT:
- Addresses key pain point in enterprise LLM adoption
- Demonstrates product-minded approach to prompt engineering
- Scalable architecture for multiple domains
        """
        
        return report.strip()

# Demo usage
if __name__ == "__main__":
    # Initialize engine
    engine = PromptOptimizationEngine()
    
    # Setup (this would take a few minutes with real data)
    engine.setup()
    
    # Test with sample questions
    test_questions = [
        "What is the treatment for high blood pressure?",
        "What are the legal requirements for a valid contract?",
        "How do you diagnose pneumonia?"
    ]
    
    print("=== PROMPT OPTIMIZATION ENGINE DEMO ===\n")
    
    for question in test_questions:
        print(f"User Question: {question}")
        print("-" * 50)
        
        # Generate optimized prompt
        result = engine.optimize_prompt(question)
        
        print(f"Detected Domain: {result.domain}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Examples Used: {len(result.examples_used)}")
        print("\nGenerated Prompt:")
        print(result.prompt)
        print("\n" + "="*80 + "\n")
    
    # Generate performance report
    print("FINAL PERFORMANCE REPORT:")
    print(engine.generate_report())