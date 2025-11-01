# import requests
# from dummy_inputs import generate_dummy_input
# import time

# # URL of your running Reasoning Agent
# REASONING_URL = "http://127.0.0.1:5007/reason"

# def send_dummy_request():
#     # Generate dummy input simulating imaging + clinical + KG
#     dummy_data = generate_dummy_input()
#     payload = {
#         "imaging_result": dummy_data["image_features"],
#         "clinical_text": dummy_data["clinical_notes"]
#     }

#     response = requests.post(REASONING_URL, json=payload)
#     return dummy_data, response.json()


# def simulate_multiple_patients(n=5, delay=1):
#     """
#     Simulate multiple patient cases to test reasoning agent.
#     :param n: number of dummy patients
#     :param delay: delay in seconds between requests
#     """
#     for i in range(n):
#         input_data, result = send_dummy_request()
#         print(f"\n--- Patient {i+1} ---")
#         print("Input Sent:")
#         print(input_data)
#         print("Reasoning Agent Response:")
#         print(result)
#         time.sleep(delay)


# if __name__ == "__main__":
#     simulate_multiple_patients(n=3, delay=0.5)
# test_reasoning_agent.py - Comprehensive testing for your Reasoning Agent

import requests
import json
from dummy_inputs import get_all_test_cases, generate_api_test_payload
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:5007"

def test_health_check():
    """Test if the agent is running"""
    print("\n" + "=" * 70)
    print("🏥 TEST 1: Health Check")
    print("=" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to agent. Is it running on port 5007?")
        print("   Run: python app.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_reasoning_with_dummy():
    """Test with random dummy input (agent generates its own)"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Reasoning with Dummy Input (Agent-Generated)")
    print("=" * 70)
    
    try:
        # Send empty payload - agent will use dummy data
        response = requests.post(
            f"{BASE_URL}/reason",
            json={},
            timeout=30
        )
        
        print(f"✅ Status Code: {response.status_code}")
        result = response.json()
        
        print("\n📊 DIAGNOSIS RESULT:")
        print(f"  • Diagnosis: {result.get('diagnosis')}")
        print(f"  • Confidence: {result.get('confidence')}")
        print(f"  • Primary Concern: {result.get('primary_concern', 'N/A')}")
        print(f"  • Recommended Action: {result.get('recommended_action', 'N/A')}")
        
        print("\n🧠 REASONING:")
        print(f"  {result.get('reasoning_text', 'No reasoning provided')}")
        
        print("\n🔍 KG CONTEXT USED:")
        kg_context = result.get('kg_context_used', {})
        if kg_context.get('differential_diagnoses'):
            for diag in kg_context['differential_diagnoses']:
                print(f"  • {diag['disease_name']} (score: {diag['total_score']:.2f})")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_specific_case(case_name: str):
    """Test with a specific predefined test case"""
    print("\n" + "=" * 70)
    print(f"🧪 TEST: {case_name.upper().replace('_', ' ')}")
    print("=" * 70)
    
    try:
        payload = generate_api_test_payload(case_name)
        
        print("📥 INPUT DATA:")
        print(f"  Imaging: {payload['imaging_result']}")
        print(f"  Clinical: {payload['clinical_text']}")
        
        response = requests.post(
            f"{BASE_URL}/reason",
            json=payload,
            timeout=30
        )
        
        result = response.json()
        
        print("\n📊 DIAGNOSIS RESULT:")
        print(f"  • Diagnosis: {result.get('diagnosis')}")
        print(f"  • Confidence: {result.get('confidence')}")
        print(f"  • Primary Concern: {result.get('primary_concern', 'N/A')}")
        print(f"  • Recommended Action: {result.get('recommended_action', 'N/A')}")
        
        print("\n🧠 REASONING:")
        reasoning = result.get('reasoning_text', 'No reasoning provided')
        # Wrap text for readability
        for line in reasoning.split('\n'):
            print(f"  {line}")
        
        print("\n🔍 KG CONTEXT:")
        kg_context = result.get('kg_context_used', {})
        
        if kg_context.get('identified_findings'):
            print(f"  Findings: {', '.join(kg_context['identified_findings'])}")
        if kg_context.get('identified_symptoms'):
            print(f"  Symptoms: {', '.join(kg_context['identified_symptoms'])}")
        if kg_context.get('identified_risk_factors'):
            print(f"  Risk Factors: {', '.join(kg_context['identified_risk_factors'])}")
        
        if kg_context.get('differential_diagnoses'):
            print("\n  Differential Diagnoses:")
            for i, diag in enumerate(kg_context['differential_diagnoses'], 1):
                print(f"    {i}. {diag['disease_name']} ({diag['malignancy']})")
                print(f"       Score: {diag['total_score']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_test_cases():
    """Run all predefined test cases"""
    print("\n" + "=" * 70)
    print("🧪 RUNNING ALL TEST CASES")
    print("=" * 70)
    
    test_cases = [
        "malignant_idc",
        "benign_fibroadenoma",
        "dcis",
        "cyst",
        "high_risk",
        "uncertain"
    ]
    
    results = {}
    for case in test_cases:
        success = test_specific_case(case)
        results[case] = "✅ PASSED" if success else "❌ FAILED"
        print("\n" + "-" * 70)
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    for case, result in results.items():
        print(f"{result} - {case.replace('_', ' ').title()}")


def test_kg_directly():
    """Test the Knowledge Graph directly (without API)"""
    print("\n" + "=" * 70)
    print("🧪 TEST: Knowledge Graph Direct Access")
    print("=" * 70)
    
    try:
        from kg_utils import get_kg_context
        
        test_imaging = {
            "mass_size": "large",
            "density": "high",
            "calcifications": "micro"
        }
        test_clinical = "Patient noticed a lump. Family history of breast cancer."
        
        print("📥 INPUT:")
        print(f"  Imaging: {test_imaging}")
        print(f"  Clinical: {test_clinical}")
        
        kg_result = get_kg_context(test_imaging, test_clinical)
        
        print("\n📊 KG OUTPUT:")
        print(json.dumps(kg_result, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_test():
    """Interactive testing mode"""
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE TEST MODE")
    print("=" * 70)
    
    print("\nEnter test case name (or 'random' for random input):")
    print("Options: malignant_idc, benign_fibroadenoma, dcis, cyst, high_risk, uncertain")
    
    case_name = input("Test case: ").strip().lower()
    
    if case_name == "random":
        test_reasoning_with_dummy()
    elif case_name:
        test_specific_case(case_name)
    else:
        print("❌ Invalid input")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        BREAST CANCER REASONING AGENT - TEST SUITE            ║
    ║                    with Knowledge Graph                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Health check
    if not test_health_check():
        print("\n❌ Agent is not running. Please start it with: python app.py")
        exit(1)
    
    # Step 2: Test KG directly
    test_kg_directly()
    
    # Step 3: Test with dummy input
    test_reasoning_with_dummy()
    
    # Step 4: Run all test cases
    print("\n\nWould you like to run all test cases? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        run_all_test_cases()
    else:
        # Run specific test
        print("\nRunning sample test case: Malignant IDC")
        test_specific_case("malignant_idc")
    
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print("\nTo test individual cases, run:")
    print("  python test_reasoning_agent.py")
    print("\nOr test via API:")
    print('  curl -X POST http://localhost:5007/reason -H "Content-Type: application/json" -d \'{"imaging_result": {...}, "clinical_text": "..."}\'')
    print("=" * 70)