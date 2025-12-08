#! /bin/bash
# make pesudo label
python src/main.py --config-name label_qwen_FHM
# conduct retreival
python src/model/Insight/make_embeddings.py
python src/model/Insight/conduct_retrieval.py
# gather experience
python src/main.py --config-name insight_qwen_FHM
# refine reference
python src/main.py --config-name manage_qwen_FHM
# conduct prediction with reference
python src/main.py --config-name inpredict_qwen_FHM