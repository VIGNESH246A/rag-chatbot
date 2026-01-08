#!/usr/bin/env python3
"""
Evaluation script for RAG chatbot performance
Tests the system with sample questions and evaluates responses
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import load_vector_store
from rag_pipeline import create_pipeline
import time
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample test questions
TEST_QUESTIONS = [
    {
        "question": "What is the return policy for small electronics?",
        "category": "policy",
        "expected_keywords": ["60-day", "return", "original condition", "RMA"]
    },
    {
        "question": "How do I set up the Smart Thermostat 300?",
        "category": "technical",
        "expected_keywords": ["C-wire", "breaker", "HomeCentral", "app"]
    },
    {
        "question": "Tell me about the Smart Refrigerator specifications",
        "category": "product",
        "expected_keywords": ["AP-FRZ-100", "21-inch", "touchscreen", "$2,899"]
    },
    {
        "question": "My security camera won't connect to WiFi",
        "category": "troubleshooting",
        "expected_keywords": ["2.4 GHz", "MAC filtering", "reset", "15 seconds"]
    },
    {
        "question": "What payment methods do you accept?",
        "category": "policy",
        "expected_keywords": ["Visa", "Mastercard", "Affirm", "$500"]
    },
    {
        "question": "How do I cancel my order?",
        "category": "policy",
        "expected_keywords": ["2 hours", "small electronics", "$150", "cancellation fee"]
    },
    {
        "question": "What is the warranty on the Smart Washing Machine?",
        "category": "product",
        "expected_keywords": ["1-year", "warranty", "extended", "3-year"]
    },
    {
        "question": "How do I fix excessive vibration on my washing machine?",
        "category": "troubleshooting",
        "expected_keywords": ["shipping bolts", "leveling feet", "level"]
    }
]


def evaluate_response(question: str, response: str, expected_keywords: List[str]) -> Dict:
    """Evaluate a single response"""
    # Check for keyword presence
    response_lower = response.lower()
    found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    keyword_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
    
    # Basic quality checks
    has_helpful_info = len(response) > 50
    not_error_message = "error" not in response_lower or "apologize" not in response_lower
    
    return {
        "keyword_score": keyword_score,
        "found_keywords": found_keywords,
        "missing_keywords": [kw for kw in expected_keywords if kw not in found_keywords],
        "has_helpful_info": has_helpful_info,
        "not_error": not_error_message,
        "response_length": len(response)
    }


def run_evaluation():
    """Run full evaluation"""
    print("\n" + "=" * 70)
    print("RAG CHATBOT EVALUATION")
    print("=" * 70 + "\n")
    
    # Load pipeline
    print("Loading pipeline...")
    try:
        vector_store = load_vector_store()
        pipeline = create_pipeline(vector_store)
        print("✓ Pipeline loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        return
    
    # Run tests
    results = []
    total_time = 0
    
    print("Running test questions...\n")
    print("-" * 70)
    
    for idx, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\nTest {idx}/{len(TEST_QUESTIONS)}")
        print(f"Question: {test['question']}")
        print(f"Category: {test['category']}")
        
        # Time the query
        start_time = time.time()
        result = pipeline.query_with_scores(test['question'])
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Evaluate
        eval_result = evaluate_response(
            test['question'],
            result['response'],
            test['expected_keywords']
        )
        
        # Store results
        results.append({
            "question": test['question'],
            "category": test['category'],
            "response": result['response'],
            "latency": elapsed,
            "num_chunks": result['num_chunks_retrieved'],
            "relevance_scores": result.get('relevance_scores', []),
            "evaluation": eval_result
        })
        
        # Print summary
        print(f"✓ Response generated in {elapsed:.2f}s")
        print(f"  Keyword Score: {eval_result['keyword_score']:.2%}")
        print(f"  Found Keywords: {', '.join(eval_result['found_keywords']) or 'None'}")
        if eval_result['missing_keywords']:
            print(f"  Missing Keywords: {', '.join(eval_result['missing_keywords'])}")
        print(f"  Chunks Retrieved: {result['num_chunks_retrieved']}")
        print(f"  Response Length: {eval_result['response_length']} chars")
        print("-" * 70)
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70 + "\n")
    
    avg_keyword_score = sum(r['evaluation']['keyword_score'] for r in results) / len(results)
    avg_latency = total_time / len(results)
    success_rate = sum(1 for r in results if r['evaluation']['keyword_score'] > 0.5) / len(results)
    
    print(f"Total Questions: {len(TEST_QUESTIONS)}")
    print(f"Average Keyword Score: {avg_keyword_score:.2%}")
    print(f"Success Rate (>50% keywords): {success_rate:.2%}")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    
    # Category breakdown
    print("\nCategory Breakdown:")
    categories = set(r['category'] for r in results)
    for category in categories:
        cat_results = [r for r in results if r['category'] == category]
        cat_score = sum(r['evaluation']['keyword_score'] for r in cat_results) / len(cat_results)
        print(f"  {category.capitalize()}: {cat_score:.2%}")
    
    # Save detailed results
    output_file = Path("evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()