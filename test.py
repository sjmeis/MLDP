import sys
import os
import MLDP
import time
import random
from tabulate import tabulate
import faiss
import numpy
import ray

print("Loaded MLDP and data files")

# Set the correct paths for the data files
MLDP.VOCAB = set([x.strip() for x in open('data/vocab.txt', 'r').readlines()])
MLDP.EMBED = 'data/glove.840B.300d_filtered.txt'

# Generate test words of different lengths
def generate_test_words(length):
    # Get a list of all words from the vocabulary
    all_words = list(MLDP.VOCAB)
    # Randomly sample words of the specified length
    return random.sample(all_words, length)

# Initialize mechanisms with and without FAISS
# Only initialize FAISS for mechanisms that support it
mechanisms = {
    "MultivariateCalibrated": {
        "with_faiss": MLDP.MultivariateCalibrated(epsilon=1, use_faiss=True),
        "without_faiss": MLDP.MultivariateCalibrated(epsilon=1, use_faiss=False)
    },
    "VickreyMechanism": {
        "with_faiss": MLDP.VickreyMechanism(epsilon=1, use_faiss=True),
        "without_faiss": MLDP.VickreyMechanism(epsilon=1, use_faiss=False)
    },
    "TEM": {
        "with_faiss": MLDP.TEM(epsilon=1, use_faiss=True),
        "without_faiss": MLDP.TEM(epsilon=1, use_faiss=False)
    },
    "Mahalanobis": {
        "with_faiss": MLDP.Mahalanobis(epsilon=1, use_faiss=True),
        "without_faiss": MLDP.Mahalanobis(epsilon=1, use_faiss=False)
    }
}

# Initialize SynTF with the full vocabulary as the corpus
mechanisms["SynTF"] = {
    "with_faiss": None,  # SynTF doesn't use FAISS
    "without_faiss": MLDP.SynTF(epsilon=1, data=list(MLDP.VOCAB))
}

# Define the Ray remote function BEFORE using it!
@ray.remote
def evaluate_mechanism(mechanism, test_words, use_faiss, epsilon):
    results = []
    start_time = time.time()
    for word in test_words:
        try:
            perturbed_word = mechanism.replace_word(word, epsilon=epsilon)
        except Exception as e:
            print(f"Error with word '{word}' ({'with' if use_faiss else 'without'} FAISS): {str(e)}")
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_words)
    return total_time, avg_time

# Set the test length
test_length = 10

print(f"\nTesting with {test_length} words:")
print("=" * 50)

# Generate test words for this length
test_words = generate_test_words(test_length)
print("\nInput words:", test_words)

# Dictionary to store results for this length
length_results = []

futures = []
future_labels = []
epsilon = 1  # Set your desired epsilon value here, or make it variable per word if needed
for mechanism_name, mechanism_versions in mechanisms.items():
    # Test with FAISS if supported
    if mechanism_versions["with_faiss"] is not None:
        futures.append(
            evaluate_mechanism.remote(mechanism_versions["with_faiss"], test_words, True, epsilon)
        )
        future_labels.append((mechanism_name, "with_faiss"))
    # Test without FAISS
    if mechanism_versions["without_faiss"] is not None:
        futures.append(
            evaluate_mechanism.remote(mechanism_versions["without_faiss"], test_words, False, epsilon)
        )
        future_labels.append((mechanism_name, "without_faiss"))

results = ray.get(futures)

# Prepare a dict to store results by (mechanism, faiss_label)
results_dict = {}
for (mechanism_name, faiss_label), (total_time, avg_time) in zip(future_labels, results):
    results_dict[(mechanism_name, faiss_label)] = (
        round(total_time, 6) if isinstance(total_time, float) else total_time,
        round(avg_time, 6) if isinstance(avg_time, float) else avg_time
    )

# For the table, collect results for each mechanism
for mechanism_name in mechanisms.keys():
    total_time_with_faiss, avg_time_with_faiss = results_dict.get((mechanism_name, "with_faiss"), ("N/A", "N/A"))
    total_time_without_faiss, avg_time_without_faiss = results_dict.get((mechanism_name, "without_faiss"), ("N/A", "N/A"))
    length_results.append([
        mechanism_name,
        total_time_with_faiss,
        avg_time_with_faiss,
        total_time_without_faiss,
        avg_time_without_faiss
    ])

# Print comparison table
print("\n\nPerformance Comparison Table:")
print("=" * 100)
headers = [
    "Mechanism",
    "Total Time with FAISS (s)",
    "Avg Time/Word with FAISS (s)",
    "Total Time without FAISS (s)",
    "Avg Time/Word without FAISS (s)"
]
table = tabulate(length_results, headers=headers, tablefmt="grid", floatfmt=".6f")
print(table)