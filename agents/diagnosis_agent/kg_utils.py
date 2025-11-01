# # kg_utils.py

# from rdflib import Graph, Namespace, RDF, RDFS, Literal

# # Step 1: Define a small domain ontology
# EX = Namespace("http://example.org/medical#")

# def build_knowledge_graph():
#     g = Graph()

#     # Define conditions and relationships
#     g.add((EX.Fibroadenoma, RDF.type, EX.BenignCondition))
#     g.add((EX.Cyst, RDF.type, EX.BenignCondition))
#     g.add((EX.BreastCancer, RDF.type, EX.MalignantCondition))
#     g.add((EX.Microcalcifications, EX.associatedWith, EX.BreastCancer))
#     g.add((EX.Macrocalcifications, EX.associatedWith, EX.BenignCondition))
#     g.add((EX.HighDensity, EX.associatedWith, EX.BreastCancer))
#     g.add((EX.LowDensity, EX.associatedWith, EX.BenignCondition))
#     g.add((EX.FamilyHistory, EX.riskFactorFor, EX.BreastCancer))
#     g.add((EX.HormoneTherapy, EX.riskFactorFor, EX.BreastCancer))
#     g.add((EX.AgeOver50, EX.riskFactorFor, EX.BreastCancer))
#     g.add((EX.YoungerAge, EX.riskFactorFor, EX.BenignCondition))

#     return g


# def get_kg_context(imaging_result, clinical_text):
#     """
#     Query the knowledge graph for related conditions/risk factors dynamically.
#     """
#     g = build_knowledge_graph()
#     context = {"related_conditions": set(), "risk_factors": set()}

#     # Map clinical terms and imaging features to KG nodes
#     feature_map = {
#         "micro": EX.Microcalcifications,
#         "macro": EX.Macrocalcifications,
#         "high": EX.HighDensity,
#         "low": EX.LowDensity,
#         "family_history": EX.FamilyHistory,
#         "hormone_therapy": EX.HormoneTherapy,
#         "age_over_50": EX.AgeOver50
#     }

#     # Extract matching nodes based on inputs
#     for feature in imaging_result.values():
#         if feature in feature_map:
#             feature_node = feature_map[feature]
#             for _, _, related in g.triples((feature_node, EX.associatedWith, None)):
#                 context["related_conditions"].add(related.split("#")[-1])

#     for risk in feature_map:
#         if risk in clinical_text.lower():
#             risk_node = feature_map[risk]
#             for _, _, disease in g.triples((risk_node, EX.riskFactorFor, None)):
#                 context["risk_factors"].add(disease.split("#")[-1])

#     # Convert sets to lists for JSON compatibility
#     context["related_conditions"] = list(context["related_conditions"])
#     context["risk_factors"] = list(context["risk_factors"])

#     return context
# kg_utils.py - Enhanced Knowledge Graph for Breast Cancer Diagnosis
# COMPLETE AND CORRECTED VERSION

import networkx as nx
import pickle
from typing import List, Dict, Any
from pathlib import Path

class BreastCancerKG:
    """
    Knowledge Graph for Breast Cancer Diagnosis
    Using NetworkX for efficient querying and reasoning
    """
    
    def __init__(self, load_from_file: bool = False):
        self.graph = nx.DiGraph()
        
        if load_from_file and Path("breast_cancer_kg.pkl").exists():
            self.load()
        else:
            self._build_kg()
    
    def _build_kg(self):
        """Build comprehensive breast cancer knowledge graph"""
        
        # ===== DISEASES =====
        diseases = [
            {"id": "IDC", "name": "Invasive Ductal Carcinoma", "type": "Disease", 
             "malignancy": "malignant", "prevalence": 0.80, "birads": "5"},
            {"id": "DCIS", "name": "Ductal Carcinoma In Situ", "type": "Disease", 
             "malignancy": "pre-malignant", "prevalence": 0.15, "birads": "4"},
            {"id": "ILC", "name": "Invasive Lobular Carcinoma", "type": "Disease", 
             "malignancy": "malignant", "prevalence": 0.10, "birads": "5"},
            {"id": "FIBROADENOMA", "name": "Fibroadenoma", "type": "Disease", 
             "malignancy": "benign", "prevalence": 0.25, "birads": "2"},
            {"id": "CYST", "name": "Simple Cyst", "type": "Disease", 
             "malignancy": "benign", "prevalence": 0.30, "birads": "2"},
        ]
        
        # ===== IMAGING FINDINGS (aligned with your dummy data) =====
        findings = [
            {"id": "MASS_IRREGULAR", "name": "Irregular Mass", "type": "Finding", 
             "suspicion": "high", "birads": "4-5"},
            {"id": "MASS_ROUND", "name": "Round/Oval Mass", "type": "Finding", 
             "suspicion": "low", "birads": "2-3"},
            {"id": "MICROCALC", "name": "Microcalcifications", "type": "Finding", 
             "suspicion": "medium-high", "birads": "4"},
            {"id": "MACROCALC", "name": "Macrocalcifications", "type": "Finding", 
             "suspicion": "low", "birads": "2"},
            {"id": "ARCHITECTURAL_DISTORTION", "name": "Architectural Distortion", "type": "Finding", 
             "suspicion": "high", "birads": "4-5"},
            {"id": "HIGH_DENSITY", "name": "High Density Mass", "type": "Finding", 
             "suspicion": "medium", "birads": "3-4"},
            {"id": "LOW_DENSITY", "name": "Low Density Mass", "type": "Finding", 
             "suspicion": "low", "birads": "2"},
        ]
        
        # ===== SYMPTOMS =====
        symptoms = [
            {"id": "PALPABLE_LUMP", "name": "Palpable Lump", "type": "Symptom"},
            {"id": "BREAST_PAIN", "name": "Breast Pain", "type": "Symptom"},
            {"id": "NIPPLE_DISCHARGE", "name": "Nipple Discharge", "type": "Symptom"},
            {"id": "SKIN_CHANGES", "name": "Skin Changes/Dimpling", "type": "Symptom"},
        ]
        
        # ===== RISK FACTORS =====
        risk_factors = [
            {"id": "FAMILY_HISTORY", "name": "Family History", "type": "RiskFactor", 
             "risk_multiplier": 2.0},
            {"id": "BRCA_MUTATION", "name": "BRCA Gene Mutation", "type": "RiskFactor", 
             "risk_multiplier": 5.0},
            {"id": "AGE_OVER_50", "name": "Age Over 50", "type": "RiskFactor", 
             "risk_multiplier": 1.8},
            {"id": "HORMONE_THERAPY", "name": "Hormone Therapy", "type": "RiskFactor", 
             "risk_multiplier": 1.3},
            {"id": "YOUNGER_AGE", "name": "Younger Age (<30)", "type": "RiskFactor", 
             "risk_multiplier": 0.5},
        ]
        
        # Add all nodes
        for item_list in [diseases, findings, symptoms, risk_factors]:
            for item in item_list:
                self.graph.add_node(item["id"], **item)
        
        # ===== DISEASE-FINDING RELATIONSHIPS =====
        disease_findings = [
            # IDC (most common malignant)
            ("IDC", "MASS_IRREGULAR", "SHOWS_FINDING", {"confidence": 0.85, "specificity": 0.70}),
            ("IDC", "ARCHITECTURAL_DISTORTION", "SHOWS_FINDING", {"confidence": 0.65, "specificity": 0.75}),
            ("IDC", "HIGH_DENSITY", "SHOWS_FINDING", {"confidence": 0.70, "specificity": 0.60}),
            ("IDC", "MICROCALC", "SHOWS_FINDING", {"confidence": 0.60, "specificity": 0.65}),
            
            # DCIS (pre-malignant)
            ("DCIS", "MICROCALC", "SHOWS_FINDING", {"confidence": 0.90, "specificity": 0.80}),
            
            # ILC (malignant, harder to detect)
            ("ILC", "ARCHITECTURAL_DISTORTION", "SHOWS_FINDING", {"confidence": 0.80, "specificity": 0.70}),
            ("ILC", "MASS_IRREGULAR", "SHOWS_FINDING", {"confidence": 0.70, "specificity": 0.65}),
            
            # Fibroadenoma (benign)
            ("FIBROADENOMA", "MASS_ROUND", "SHOWS_FINDING", {"confidence": 0.85, "specificity": 0.75}),
            ("FIBROADENOMA", "LOW_DENSITY", "SHOWS_FINDING", {"confidence": 0.70, "specificity": 0.60}),
            
            # Cyst (benign)
            ("CYST", "MASS_ROUND", "SHOWS_FINDING", {"confidence": 0.80, "specificity": 0.70}),
            ("CYST", "LOW_DENSITY", "SHOWS_FINDING", {"confidence": 0.75, "specificity": 0.65}),
            ("CYST", "MACROCALC", "SHOWS_FINDING", {"confidence": 0.40, "specificity": 0.50}),
        ]
        
        # ===== DISEASE-SYMPTOM RELATIONSHIPS =====
        disease_symptoms = [
            ("IDC", "PALPABLE_LUMP", "HAS_SYMPTOM", {"confidence": 0.70}),
            ("IDC", "SKIN_CHANGES", "HAS_SYMPTOM", {"confidence": 0.40}),
            ("ILC", "PALPABLE_LUMP", "HAS_SYMPTOM", {"confidence": 0.65}),
            ("FIBROADENOMA", "PALPABLE_LUMP", "HAS_SYMPTOM", {"confidence": 0.60}),
            ("CYST", "BREAST_PAIN", "HAS_SYMPTOM", {"confidence": 0.50}),
        ]
        
        # ===== RISK FACTOR RELATIONSHIPS =====
        risk_relationships = [
            ("FAMILY_HISTORY", "IDC", "INCREASES_RISK", {"relative_risk": 2.0}),
            ("BRCA_MUTATION", "IDC", "INCREASES_RISK", {"relative_risk": 5.0}),
            ("AGE_OVER_50", "IDC", "INCREASES_RISK", {"relative_risk": 1.8}),
            ("HORMONE_THERAPY", "IDC", "INCREASES_RISK", {"relative_risk": 1.3}),
            ("YOUNGER_AGE", "FIBROADENOMA", "INCREASES_RISK", {"relative_risk": 2.0}),
        ]
        
        # Add all edges
        for edges in [disease_findings, disease_symptoms, risk_relationships]:
            for source, target, relation, attrs in edges:
                self.graph.add_edge(source, target, relationship=relation, **attrs)
    
    # ===== MAPPING FUNCTIONS (for your dummy input format) =====
    
    def _map_imaging_features(self, imaging_result: Dict) -> List[str]:
        """
        Map your dummy imaging format to KG finding IDs
        Example: {"mass_size": "large", "density": "high", "calcifications": "micro"}
        """
        finding_ids = []
        
        # Map calcifications
        calc_type = imaging_result.get("calcifications", "").lower()
        if calc_type == "micro":
            finding_ids.append("MICROCALC")
        elif calc_type == "macro":
            finding_ids.append("MACROCALC")
        
        # Map density
        density = imaging_result.get("density", "").lower()
        if density == "high":
            finding_ids.append("HIGH_DENSITY")
        elif density == "low":
            finding_ids.append("LOW_DENSITY")
        
        # Map mass characteristics (simplified)
        mass_size = imaging_result.get("mass_size", "").lower()
        if mass_size in ["large", "medium"]:
            finding_ids.append("MASS_IRREGULAR")  # Assume large masses are more concerning
        elif mass_size == "small":
            finding_ids.append("MASS_ROUND")
        
        return finding_ids
    
    def _extract_symptoms_from_text(self, clinical_text: str) -> List[str]:
        """Extract symptom IDs from clinical notes text"""
        text_lower = clinical_text.lower()
        symptom_ids = []
        
        if "lump" in text_lower:
            symptom_ids.append("PALPABLE_LUMP")
        if "pain" in text_lower:
            symptom_ids.append("BREAST_PAIN")
        if "discharge" in text_lower:
            symptom_ids.append("NIPPLE_DISCHARGE")
        if "skin" in text_lower or "dimpl" in text_lower:
            symptom_ids.append("SKIN_CHANGES")
        
        return symptom_ids
    
    def _extract_risk_factors_from_text(self, clinical_text: str) -> List[str]:
        """Extract risk factor IDs from clinical notes text"""
        text_lower = clinical_text.lower()
        risk_ids = []
        
        if "family history" in text_lower or "family_history" in text_lower:
            risk_ids.append("FAMILY_HISTORY")
        if "brca" in text_lower:
            risk_ids.append("BRCA_MUTATION")
        if "hormone" in text_lower:
            risk_ids.append("HORMONE_THERAPY")
        
        # Age inference (simplified - in real system, extract from structured data)
        if "screening" in text_lower or "routine" in text_lower:
            risk_ids.append("AGE_OVER_50")
        
        return risk_ids
    
    def _get_diseases_by_findings(self, finding_ids: List[str]) -> List[Dict]:
        """Find diseases associated with imaging findings"""
        disease_scores = {}
        
        for finding in finding_ids:
            if finding not in self.graph:
                continue
                
            for node in self.graph.nodes():
                if self.graph.nodes[node].get("type") == "Disease":
                    if self.graph.has_edge(node, finding):
                        edge = self.graph.get_edge_data(node, finding)
                        if edge.get("relationship") == "SHOWS_FINDING":
                            if node not in disease_scores:
                                disease_scores[node] = {
                                    "disease_id": node,
                                    "disease_name": self.graph.nodes[node]["name"],
                                    "malignancy": self.graph.nodes[node]["malignancy"],
                                    "finding_score": 0,
                                    "matched_findings": []
                                }
                            
                            confidence = edge.get("confidence", 0.5)
                            disease_scores[node]["finding_score"] += confidence * 10
                            disease_scores[node]["matched_findings"].append(
                                self.graph.nodes[finding]["name"]
                            )
        
        return list(disease_scores.values())
    
    def _check_symptom_match(self, disease_id: str, symptom_ids: List[str]) -> Dict:
        """Check how well symptoms match a disease"""
        matched = []
        score = 0
        
        for symptom in symptom_ids:
            if self.graph.has_edge(disease_id, symptom):
                edge = self.graph.get_edge_data(disease_id, symptom)
                if edge.get("relationship") == "HAS_SYMPTOM":
                    matched.append(self.graph.nodes[symptom]["name"])
                    score += edge.get("confidence", 0.5) * 5
        
        return {"matched_symptoms": matched, "symptom_score": score}
    
    def _check_risk_factors(self, disease_id: str, risk_factor_ids: List[str]) -> Dict:
        """Check risk factors for a disease"""
        matched = []
        multiplier = 1.0
        
        for risk in risk_factor_ids:
            if self.graph.has_edge(risk, disease_id):
                edge = self.graph.get_edge_data(risk, disease_id)
                if edge.get("relationship") == "INCREASES_RISK":
                    matched.append(self.graph.nodes[risk]["name"])
                    multiplier *= edge.get("relative_risk", 1.0)
        
        return {"matched_risk_factors": matched, "risk_multiplier": multiplier}
    
    # ===== MAIN QUERY FUNCTION (FIXED - only defined once) =====
    
    def get_kg_context(self, imaging_result: Dict, clinical_text: str) -> Dict:
        """
        Main function called by your API
        Returns structured KG context for LLM reasoning with total_score calculation
        """
        # Map inputs to KG entities
        finding_ids = self._map_imaging_features(imaging_result)
        symptom_ids = self._extract_symptoms_from_text(clinical_text)
        risk_factor_ids = self._extract_risk_factors_from_text(clinical_text)
        
        # Query KG for differential diagnoses
        differential_diagnoses = self._get_diseases_by_findings(finding_ids)
        
        # Enhance with symptom and risk factor analysis
        for diagnosis in differential_diagnoses:
            symptom_data = self._check_symptom_match(diagnosis["disease_id"], symptom_ids)
            risk_data = self._check_risk_factors(diagnosis["disease_id"], risk_factor_ids)
            
            diagnosis["symptom_match"] = symptom_data
            diagnosis["risk_factors"] = risk_data
            
            # Calculate total score
            base_score = diagnosis["finding_score"]
            symptom_score = symptom_data["symptom_score"]
            risk_multiplier = risk_data["risk_multiplier"]
            
            diagnosis["total_score"] = (base_score + symptom_score) * risk_multiplier
        
        # Sort by total score
        differential_diagnoses.sort(key=lambda x: x["total_score"], reverse=True)
        
        return {
            "differential_diagnoses": differential_diagnoses[:3],  # Top 3
            "identified_findings": [self.graph.nodes[f]["name"] for f in finding_ids if f in self.graph],
            "identified_symptoms": [self.graph.nodes[s]["name"] for s in symptom_ids if s in self.graph],
            "identified_risk_factors": [self.graph.nodes[r]["name"] for r in risk_factor_ids if r in self.graph],
        }
    
    def save(self, filepath: str = "breast_cancer_kg.pkl"):
        """Save KG to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load(self, filepath: str = "breast_cancer_kg.pkl"):
        """Load KG from disk"""
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)


# ===== SINGLETON INSTANCE =====
_kg_instance = None

def get_kg_instance() -> BreastCancerKG:
    """Get or create singleton KG instance"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = BreastCancerKG()
    return _kg_instance


# ===== MAIN API FUNCTION (used by app.py) =====
def get_kg_context(imaging_result: Dict, clinical_text: str) -> Dict:
    """
    Main function called from your FastAPI app
    Compatible with your existing code
    """
    kg = get_kg_instance()
    return kg.get_kg_context(imaging_result, clinical_text)


# ===== TEST =====
if __name__ == "__main__":
    # Test with your dummy format
    test_imaging = {
        "mass_size": "large",
        "density": "high",
        "calcifications": "micro"
    }
    test_clinical = "Patient noticed a lump in the left breast. Family history of breast cancer."
    
    context = get_kg_context(test_imaging, test_clinical)
    
    import json
    print(json.dumps(context, indent=2))