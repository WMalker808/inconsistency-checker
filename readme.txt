#!/Users/max_walker/inconsistency/bin/python3


To run
python3 -m venv inconsistency ; source inconsistency/bin/activate ; cd /Volumes/Cradle/Python/InconsistencyChecker ; python guardian_style_detector.py --web

python3 -m venv inconsistency ; source inconsistency/bin/activate





# Analyze a directory of text files
python guardian_style_detector.py --dir /path/to/articles

# Analyze a single file
python guardian_style_detector.py --file single_article.txt

# Set custom parameters
python guardian_style_detector.py --dir /path/to/articles --min-length 4 --min-occurrences 10

# Save results to CSV
python guardian_style_detector.py --dir /path/to/articles --output results.csv

python guardian_style_detector.py --web

from guardian_style_detector import StyleInconsistencyDetector

# Create corpus (list of document strings)
corpus = ["Document 1 content...", "Document 2 content..."]

# Initialize detector with custom parameters
detector = StyleInconsistencyDetector(
    min_term_length=3,
    min_occurrences=5,
    similarity_threshold=0.8
)

# Analyze corpus
results = detector.analyze_corpus(corpus)

# Process results
for result in results:
    print(f"Term: {result['term']} (Total: {result['total_occurrences']})")
    for variant in result['variants']:
        print(f"  - {variant['variant']}: {variant['count']}")