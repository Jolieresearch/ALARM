#! /bin/bash
# make pesudo label
python src/main.py --config-name label_qwen_ToxiCN
# conduct retreival
python src/model/Experience/make_embeddings.py
python src/model/Experience/conduct_retrieval.py
# gather experience
python src/main.py --config-name insight_qwen_ToxiCN
# refine reference
python src/main.py --config-name manage_qwen_ToxiCN
# conduct prediction with reference
python src/main.py --config-name inpredict_qwen_ToxiCN
