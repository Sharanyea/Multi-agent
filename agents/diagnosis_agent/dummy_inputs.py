# import random

# def generate_dummy_image_features():
#     """Simulate features extracted from mammogram images."""
#     sizes = ["small", "medium", "large"]
#     densities = ["low", "medium", "high"]
#     calcifications = ["none", "micro", "macro"]

#     return {
#         "mass_size": random.choice(sizes),
#         "density": random.choice(densities),
#         "calcifications": random.choice(calcifications)
#     }

# def generate_dummy_clinical_notes():
#     """Simulate simple clinical notes."""
#     notes = [
#         "Patient reports mild breast pain",
#         "No family history of breast cancer",
#         "Patient noticed a lump in the left breast",
#         "Routine screening, no symptoms reported"
#     ]
#     return random.choice(notes)

# def generate_dummy_kg_insights():
#     """Simulate outputs from a knowledge graph."""
#     risk_factors = ["family_history", "age_over_50", "hormone_therapy", "BRCA1_positive"]
#     return {
#         "risk_factor": random.choice(risk_factors),
#         "related_conditions": ["fibroadenoma", "cyst", "benign_tumor"]
#     }

# def generate_dummy_input():
#     return {
#         "image_features": generate_dummy_image_features(),
#         "clinical_notes": generate_dummy_clinical_notes(),
#         "kg_insights": generate_dummy_kg_insights()
#     }

# if __name__ == "__main__":
#     from pprint import pprint
#     pprint(generate_dummy_input())
# dummy_inputs.py - Enhanced with realistic breast cancer test cases

import random

def generate_dummy_image_features():
    """Simulate features extracted from mammogram images."""
    sizes = ["small", "medium", "large"]
    densities = ["low", "medium", "high"]
    calcifications = ["none", "micro", "macro"]

    return {
        "mass_size": random.choice(sizes),
        "density": random.choice(densities),
        "calcifications": random.choice(calcifications)
    }

def generate_dummy_clinical_notes():
    """Simulate simple clinical notes."""
    notes = [
        "Patient reports mild breast pain",
        "No family history of breast cancer",
        "Patient noticed a lump in the left breast",
        "Routine screening, no symptoms reported",
        "Family history of breast cancer. Patient concerned about recent lump.",
        "Patient on hormone therapy for 5 years. Palpable mass detected.",
        "Young patient (28 years old) with mobile, painless lump.",
        "Post-menopausal patient. Screening mammogram shows suspicious findings."
    ]
    return random.choice(notes)

def generate_dummy_kg_insights():
    """Simulate outputs from a knowledge graph (legacy - for backward compatibility)."""
    risk_factors = ["family_history", "age_over_50", "hormone_therapy", "BRCA1_positive"]
    return {
        "risk_factor": random.choice(risk_factors),
        "related_conditions": ["fibroadenoma", "cyst", "benign_tumor"]
    }

def generate_dummy_input():
    """Generate basic random dummy input"""
    return {
        "image_features": generate_dummy_image_features(),
        "clinical_notes": generate_dummy_clinical_notes(),
        "kg_insights": generate_dummy_kg_insights()
    }


# ===== NEW: REALISTIC TEST CASES =====

def get_test_case_malignant_idc():
    """
    Test Case 1: Invasive Ductal Carcinoma (IDC)
    High suspicion malignant case
    """
    return {
        "image_features": {
            "mass_size": "large",
            "density": "high",
            "calcifications": "micro"
        },
        "clinical_notes": "Patient noticed a lump in the left breast. Family history of breast cancer. Palpable irregular mass on examination."
    }

def get_test_case_benign_fibroadenoma():
    """
    Test Case 2: Fibroadenoma
    Likely benign case
    """
    return {
        "image_features": {
            "mass_size": "small",
            "density": "low",
            "calcifications": "none"
        },
        "clinical_notes": "Young patient (28 years old) with mobile, painless lump. No family history."
    }

def get_test_case_dcis():
    """
    Test Case 3: DCIS (Pre-malignant)
    Microcalcifications on screening
    """
    return {
        "image_features": {
            "mass_size": "small",
            "density": "medium",
            "calcifications": "micro"
        },
        "clinical_notes": "Routine screening mammogram. Asymptomatic. Patient is 58 years old."
    }

def get_test_case_cyst():
    """
    Test Case 4: Simple Cyst
    Benign fluid-filled mass
    """
    return {
        "image_features": {
            "mass_size": "medium",
            "density": "low",
            "calcifications": "macro"
        },
        "clinical_notes": "Patient reports breast pain. Round, well-circumscribed mass on imaging."
    }

def get_test_case_high_risk():
    """
    Test Case 5: High Risk Patient
    Multiple risk factors with suspicious findings
    """
    return {
        "image_features": {
            "mass_size": "large",
            "density": "high",
            "calcifications": "micro"
        },
        "clinical_notes": "Patient on hormone therapy for 10 years. Strong family history of breast cancer. Sister diagnosed with BRCA1 mutation. Noticed skin changes and palpable lump."
    }

def get_test_case_uncertain():
    """
    Test Case 6: Uncertain/Ambiguous
    Mixed findings requiring further investigation
    """
    return {
        "image_features": {
            "mass_size": "medium",
            "density": "medium",
            "calcifications": "none"
        },
        "clinical_notes": "Screening mammogram. Dense breast tissue makes assessment difficult. No symptoms."
    }


def get_all_test_cases():
    """Return all predefined test cases"""
    return {
        "malignant_idc": get_test_case_malignant_idc(),
        "benign_fibroadenoma": get_test_case_benign_fibroadenoma(),
        "dcis": get_test_case_dcis(),
        "cyst": get_test_case_cyst(),
        "high_risk": get_test_case_high_risk(),
        "uncertain": get_test_case_uncertain()
    }


def get_random_test_case():
    """Get a random realistic test case"""
    test_cases = [
        get_test_case_malignant_idc,
        get_test_case_benign_fibroadenoma,
        get_test_case_dcis,
        get_test_case_cyst,
        get_test_case_high_risk,
        get_test_case_uncertain
    ]
    return random.choice(test_cases)()


# ===== FORMATTED INPUT FOR API TESTING =====

def generate_api_test_payload(test_case_name: str = None):
    """
    Generate properly formatted payload for /reason endpoint
    
    Args:
        test_case_name: One of ["malignant_idc", "benign_fibroadenoma", "dcis", 
                                 "cyst", "high_risk", "uncertain"] or None for random
    """
    if test_case_name:
        test_cases = get_all_test_cases()
        if test_case_name not in test_cases:
            raise ValueError(f"Unknown test case: {test_case_name}")
        test_data = test_cases[test_case_name]
    else:
        test_data = get_random_test_case()
    
    return {
        "imaging_result": test_data["image_features"],
        "clinical_text": test_data["clinical_notes"]
    }


if __name__ == "__main__":
    from pprint import pprint
    
    print("=" * 60)
    print("RANDOM DUMMY INPUT (backward compatible):")
    print("=" * 60)
    pprint(generate_dummy_input())
    
    print("\n" + "=" * 60)
    print("ALL REALISTIC TEST CASES:")
    print("=" * 60)
    
    all_cases = get_all_test_cases()
    for name, case in all_cases.items():
        print(f"\n📋 {name.upper().replace('_', ' ')}:")
        pprint(case)
    
    print("\n" + "=" * 60)
    print("API TEST PAYLOAD (for POST /reason):")
    print("=" * 60)
    pprint(generate_api_test_payload("malignant_idc"))