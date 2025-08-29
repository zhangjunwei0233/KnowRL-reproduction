#!/usr/bin/env python3
"""
Test script to verify DeepSeek-v3 API integration works with KnowRL reward functions.
Run this before starting training to ensure API configuration is correct.

Usage:
    python test_deepseek_api.py
"""

import os
import sys
import logging
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_deepseek_direct_call():
    """Test direct DeepSeek API call"""
    print("=" * 50)
    print("🧪 Testing Direct DeepSeek API Call")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY_FACTSCORE")
    base_url = os.environ.get("OPENAI_BASE_URL_FACTSCORE")

    if not api_key or api_key == "your_deepseek_api_key_here":
        print("❌ DeepSeek API key not configured!")
        print("   Please update OPENAI_API_KEY_FACTSCORE in train.sh")
        return False

    print(f"📡 API Base URL: {base_url}")
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "What is 2+2? Answer briefly."}
            ],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()
        print(f"✅ DeepSeek Response: {result}")
        return True

    except Exception as e:
        print(f"❌ DeepSeek API Error: {str(e)}")
        return False


def test_fact_reward_integration():
    """Test FActScore integration with DeepSeek"""
    print("\n" + "=" * 50)
    print("🧪 Testing FActScore Integration")
    print("=" * 50)

    try:
        # Test if knowledge base is available
        db_path = os.environ.get("FACTSCORE_DB_PATH")
        if not db_path or not os.path.exists(db_path):
            print(f"❌ Knowledge base not found at: {db_path}")
            print("   Please download knowledge base first")
            return False

        print(f"✅ Knowledge base found: {db_path}")

        # Test FactualityScorer initialization
        from reward_function.fact_reward import FactualityScorer

        scorer = FactualityScorer()
        if scorer.get_fact_scorer() is None:
            print("❌ Failed to initialize FactScorer with DeepSeek")
            return False

        print("✅ FactScorer initialized successfully with DeepSeek")

        # Test a simple factuality evaluation
        test_prompts = ["What is the capital of France?"]
        test_completions = [
            "<think>Paris is the capital of France.</think><answer>Paris</answer>"]

        print("🧪 Testing factuality evaluation...")
        rewards = scorer.factuality_count_reward_func(
            test_prompts, test_completions)

        if rewards and len(rewards) > 0:
            print(f"✅ Factuality reward: {rewards[0]:.3f}")
            return True
        else:
            print("❌ Failed to get factuality rewards")
            return False

    except Exception as e:
        print(f"❌ FActScore Integration Error: {str(e)}")
        return False


def test_correct_reward_integration():
    """Test Correctness Reward integration with DeepSeek"""
    print("\n" + "=" * 50)
    print("🧪 Testing Correctness Reward Integration")
    print("=" * 50)

    try:
        from reward_function.correct_reward import llm_eval_reward_func

        # Test simple correctness evaluation
        test_prompts = ["What is 2+2?"]
        test_completions = ["<think>2+2 equals 4</think><answer>4</answer>"]
        test_best_answers = ["4"]

        print("🧪 Testing correctness evaluation...")
        rewards = llm_eval_reward_func(
            prompts=test_prompts,
            completions=test_completions,
            best_answer=test_best_answers
        )

        if rewards and len(rewards) > 0:
            print(f"✅ Correctness reward: {rewards[0]:.3f}")
            return True
        else:
            print("❌ Failed to get correctness rewards")
            return False

    except Exception as e:
        print(f"❌ Correctness Reward Integration Error: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🚀 KnowRL DeepSeek-v3 API Integration Test")
    print("=" * 60)

    # Check environment variables
    required_vars = [
        "OPENAI_API_KEY_FACTSCORE",
        "OPENAI_BASE_URL_FACTSCORE",
        "OPENAI_API_KEY_JUDGE",
        "OPENAI_API_BASE_JUDGE",
        "FACTSCORE_DB_PATH"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please run: source train.sh")
        sys.exit(1)

    # Run tests
    test_results = []

    test_results.append(("Direct DeepSeek API", test_deepseek_direct_call()))
    test_results.append(
        ("FActScore Integration", test_fact_reward_integration()))
    test_results.append(
        ("Correctness Reward", test_correct_reward_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! DeepSeek-v3 integration ready.")
        print("   You can now start training with: bash train.sh")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        print("   Check API keys, knowledge base, and network connectivity.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

