# Persona construction and questionnaire assessment

`generate_persona.py` samples candidate biographies, generates third-person and first-person descriptions, and assigns qualitative Big Five labels. It uses PERSONA_CANDIDATES_CSV, PERSONA_SAMPLE_SIZE (original default5000), PERSONA_LLM_URL, PERSONA_GENERATION_DATA and PERSONA_GENERATION_OUTPUT. This script does not implement the complete final600-persona cohort selection procedure.

`bigfive_test.py` performs repeated questionnaire assessment; configure PERSONA_PROFILE_CSV, PERSONA_GENERATION_DATA, PERSONA_GENERATION_OUTPUT, PERSONA_LLM_URL and PERSONA_BIG5_REPEATS. Prompt and scoring content are retained from the supplied code. Required instruments and real-name candidate records are not included in the public repository.

Install repository-root requirements-generation.txt. Provide a running model endpoint through PERSONA_LLM_URL. Execution can invoke paid APIs or GPU inference; it is not started by the offline analysis wrapper.
