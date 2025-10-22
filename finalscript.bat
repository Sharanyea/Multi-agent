
@echo off
setlocal

echo Starting Diagnosis_agent...
start "Run Diagnosis_agent Server" cmd /k python L:\multi-agent\multi_agent_breast_cancer_diagnosis\agents\diagnosis_agent\app.py

echo Starting imageing_agent...
start "Run imageing_agent Server" cmd /k python L:\multi-agent\multi_agent_breast_cancer_diagnosis\agents\imaging_agent\app.py

echo Starting knowledge_agent...
start "Run knowledge_agent Server" cmd /k python L:\multi-agent\multi_agent_breast_cancer_diagnosis\agents\knowledge_agent\app.py

echo Starting coordinator_agent...
start "Run Coordinator_agent Server" cmd /k python L:\multi-agent\multi_agent_breast_cancer_diagnosis\agents\coordinator_agent\app.py


echo Starting streamlit server..
start "stremlit server" cmd /k streamlit run L:\multi-agent\multi_agent_breast_cancer_diagnosis\ui\app.py

echo All servers launched in separate terminals.
endlocal
