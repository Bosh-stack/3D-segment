#!/bin/bash
set -e

ROOT=/data/Open3DSG_trainset
OUT_GRAPH=open3dsg/output/graphs/myset
OUT_PREPROC=open3dsg/output/preprocessed/myset

python open3dsg/data/gen_myset_subgraphs.py --root $ROOT --out $OUT_GRAPH --split train
python open3dsg/data/get_object_frame_myset.py --root $ROOT --out $OUT_PREPROC/frames --top_k 5
python open3dsg/data/preprocess_myset.py --root $ROOT \
  --graphs $OUT_GRAPH/graphs/train.json --frames $OUT_PREPROC/frames \
  --out $OUT_PREPROC --max_edges_per_node 10 --max_nodes 10

python open3dsg/scripts/run.py --dump_features --dataset myset \
  --scales 3 --top_k_frames 5 --clip_model OpenSeg --blip \
  --max_nodes 10 --max_edges 100

python open3dsg/scripts/run.py \
  --epochs 100 --batch_size 4 --gpus 4 --workers 8 \
  --use_rgb --dataset myset --clip_model OpenSeg --blip \
  --load_features open3dsg/output/features/myset \
  --mixed_precision --max_nodes 10 --max_edges 100
