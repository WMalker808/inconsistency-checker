#!/usr/bin/env python3
import os
import re
import nltk
import string
from collections import defaultdict, Counter
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except LookupError:
    # Fallback for stopwords if NLTK resources aren't available
    ENGLISH_STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
import pandas as pd
from flask import Flask, render_template, request
import json

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    # Ensure punkt_tab is downloaded as well
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")
    print("If you continue to have NLTK data issues, run this in a Python interpreter:")
    print("import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')")

class StyleInconsistencyDetector:
    def __init__(self, min_term_length=3, min_occurrences=5, similarity_threshold=0.8):
        self.min_term_length = min_term_length
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold
        try:
            self.stop_words = set(stopwords.words('english'))
        except (LookupError, NameError):
            # Use fallback stopwords if NLTK resources aren't available
            self.stop_words = ENGLISH_STOPWORDS
            print("Using fallback stopwords due to NLTK resource issues")
        self.punctuation = set(string.punctuation)

    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase for some processing steps
        text_lower = text.lower()
        
        return text, text_lower

    def extract_ngrams(self, text, max_n=4):
        """Extract n-grams from text, preserving case."""
        # Use a simpler tokenization method to avoid punkt_tab dependency
        try:
            tokens = word_tokenize(text)
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback tokenization if NLTK resources aren't available
            print("Using fallback tokenization method due to NLTK resource issues")
            tokens = text.split()
            sentences = re.split(r'[.!?]+', text)
        
        # Extract all n-grams up to max_n
        all_ngrams = []
        
        # Process individual tokens (unigrams)
        for token in tokens:
            if (len(token) >= self.min_term_length and 
                token.lower() not in self.stop_words and
                token not in self.punctuation and
                not token.isdigit() and
                not all(c in self.punctuation for c in token)):
                all_ngrams.append(token)
        
        # Process sentences to extract multi-word phrases
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            # Extract n-grams (n > 1)
            for n in range(2, min(max_n + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    
                    # Filter short ngrams and those with too many stop words
                    if (len(ngram) >= self.min_term_length and
                        not all(word.lower() in self.stop_words for word in ngram.split()) and
                        sum(1 for word in ngram.split() if word.lower() in self.stop_words) <= n/2):
                        all_ngrams.append(ngram)
        
        return all_ngrams

    def are_likely_variants(self, term1, term2):
        """Check if two terms are likely variants of each other."""
        # Compare terms without case
        if term1.lower() == term2.lower():
            return True
            
        # Check for hyphenation differences
        if term1.replace('-', ' ').lower() == term2.replace('-', ' ').lower():
            return True
            
        # Check for possessive forms
        if term1.lower() + "'s" == term2.lower() or term1.lower() == term2.lower() + "'s":
            return True
            
        # Check for plural forms
        if term1.lower() + "s" == term2.lower() or term1.lower() == term2.lower() + "s":
            return True
            
        return False

    def find_inconsistencies(self, corpus):
        """
        Find terms with inconsistent styling across the corpus.
        Returns a dictionary mapping terms to their variants and counts.
        """
        # Dictionary to store all term variants
        term_variants = defaultdict(Counter)
        
        # Process each document in the corpus
        for document in corpus:
            text, text_lower = self.preprocess_text(document)
            ngrams = self.extract_ngrams(text)
            
            # Count unique occurrences in this document
            doc_terms = set(ngrams)
            
            for term in doc_terms:
                # Use lowercase as a normalized form for grouping
                term_lower = term.lower()
                term_variants[term_lower][term] += 1
        
        # Filter for terms with multiple variants and minimum occurrences
        inconsistencies = {}
        for term_key, variants in term_variants.items():
            # Check if there are multiple variants
            if len(variants) > 1:
                total_occurrences = sum(variants.values())
                
                # Only include terms that occur frequently enough
                if total_occurrences >= self.min_occurrences:
                    inconsistencies[term_key] = dict(variants)
        
        return inconsistencies

    def group_similar_terms(self, inconsistencies):
        """Group terms that are likely to be variants of each other."""
        # First, create a flat list of all variant forms
        all_variants = []
        for term_key, variants in inconsistencies.items():
            all_variants.extend([(term_key, variant) for variant in variants])
        
        # Initialize groups with the first term
        if not all_variants:
            return []
            
        groups = []
        grouped_keys = set()
        
        # For each term key not yet grouped
        for term_key, _ in all_variants:
            if term_key in grouped_keys:
                continue
                
            # Create a new group
            group = {term_key: inconsistencies[term_key]}
            grouped_keys.add(term_key)
            
            # Find similar terms
            for other_key in inconsistencies:
                if other_key in grouped_keys:
                    continue
                    
                # Check if terms are similar
                if self.are_likely_variants(term_key, other_key):
                    group[other_key] = inconsistencies[other_key]
                    grouped_keys.add(other_key)
            
            groups.append(group)
        
        return groups

    def analyze_corpus(self, corpus):
        """
        Main method to analyze the corpus and return inconsistencies.
        """
        inconsistencies = self.find_inconsistencies(corpus)
        grouped_inconsistencies = self.group_similar_terms(inconsistencies)
        
        # Format results
        results = []
        for group in grouped_inconsistencies:
            group_result = []
            for term_key, variants in group.items():
                # Add all variants with their counts
                for variant, count in variants.items():
                    group_result.append({
                        'variant': variant,
                        'count': count
                    })
            
            # Only include groups with actual inconsistencies
            if len(group_result) > 1:
                # Sort variants by count (descending)
                group_result.sort(key=lambda x: x['count'], reverse=True)
                
                # Calculate total occurrences
                total_occurrences = sum(item['count'] for item in group_result)
                
                results.append({
                    'term': group_result[0]['variant'],  # Use most common form as the term name
                    'total_occurrences': total_occurrences,
                    'variants': group_result
                })
        
        # Sort results by total occurrences
        results.sort(key=lambda x: x['total_occurrences'], reverse=True)
        
        return results


# Flask web application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        # Check if a file or directory was submitted
        if 'file' in request.files and request.files['file'].filename:
            # Process the uploaded file
            file = request.files['file']
            content = file.read().decode('utf-8')
            corpus = [content]
        elif 'directory' in request.form and request.form['directory']:
            # Process all files in the specified directory
            directory = request.form['directory']
            corpus = []
            
            if os.path.isdir(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.html')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            corpus.append(f.read())
        else:
            # Process the entered text
            text = request.form.get('text', '')
            corpus = [text]
        
        min_term_length = int(request.form.get('min_term_length', 3))
        min_occurrences = int(request.form.get('min_occurrences', 5))
        
        # Analyze the corpus
        detector = StyleInconsistencyDetector(
            min_term_length=min_term_length, 
            min_occurrences=min_occurrences
        )
        results = detector.analyze_corpus(corpus)
        
    return render_template('index.html', results=results)

def setup_nltk():
    """Ensure all required NLTK data is available"""
    try:
        # Try to create necessary directories if they don't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required NLTK resources with explicit paths
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        
        # Verify downloads were successful
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        stopwords.words('english')
        word_tokenize("Test sentence.")
        
        print("NLTK setup completed successfully.")
    except Exception as e:
        print(f"NLTK setup warning: {e}")
        print("The program will attempt to continue with fallback methods.")

def main():
    # Handle command-line usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect style inconsistencies in a corpus of text.')
    parser.add_argument('--dir', type=str, help='Directory containing text files to analyze')
    parser.add_argument('--file', type=str, help='Single file to analyze')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum term length')
    parser.add_argument('--min-occurrences', type=int, default=5, help='Minimum number of occurrences')
    parser.add_argument('--web', action='store_true', help='Start the web interface')
    parser.add_argument('--output', type=str, help='Output file for results (CSV format)')
    
    args = parser.parse_args()
    
    if args.web:
        # Start the web application
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
        return
    
    # Collect the corpus
    corpus = []
    
    if args.dir:
        # Read all files in directory
        if os.path.isdir(args.dir):
            for filename in os.listdir(args.dir):
                file_path = os.path.join(args.dir, filename)
                if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.html')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        corpus.append(f.read())
        else:
            print(f"Directory {args.dir} not found.")
            return
    
    elif args.file:
        # Read single file
        if os.path.isfile(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                corpus.append(f.read())
        else:
            print(f"File {args.file} not found.")
            return
    
    else:
        print("No input specified. Use --dir, --file, or --web options.")
        return
    
    # Analyze the corpus
    detector = StyleInconsistencyDetector(
        min_term_length=args.min_length, 
        min_occurrences=args.min_occurrences
    )
    results = detector.analyze_corpus(corpus)
    
    # Print or save results
    if args.output:
        # Convert to DataFrame and save as CSV
        rows = []
        for result in results:
            term = result['term']
            for variant in result['variants']:
                rows.append({
                    'term': term,
                    'variant': variant['variant'],
                    'count': variant['count'],
                    'total_occurrences': result['total_occurrences']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        # Print to console
        for result in results:
            print(f"Term: {result['term']} (Total: {result['total_occurrences']})")
            for variant in result['variants']:
                print(f"  - {variant['variant']}: {variant['count']}")
            print()

if __name__ == "__main__":
    # Setup NLTK first
    setup_nltk()
    main()
