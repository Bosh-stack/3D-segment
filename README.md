<
## Setup

```bash
conda env create --name open3dsg python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

> **Note**: This software is tested and developed for CUDA 11.8 & tested with an NVIDIA V100 32GB.

### Data Preparation

1. Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Unpack the image sequences for each scan. And include the 3DSSG files as a subdirectory in 3RScan.
2. Download [ScanNet](http://www.scan-net.org/ScanNet/) and split the scans into ```scannet_2d``` and ```scannet_3d```. We use the pre-processed data from [ScanNet ETH preprocessed 3D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip) & [ScanNet ETH preprocessed 2D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip), when using the pre-processed version make sure that you have acknowledged the ScanNet license. When using processed ScanNet ETH preprocessed 2D frames, use the matching [intrinsics](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
3. Download the [3DSSG_subset.zip](http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip) and extract the files in the 3RScan directory for training and evaluation. Additional meta files can be found [here](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
4. Download 3RScan & ScanNet meta data files using ```scripts/download_scannet_meta.sh``` and ```scripts/download_scannet_meta.sh``` and place them in their data directories.
5. Set the path to your data in ```config/config.py```

### Data Preprocessing

3DSSG provides pre-constructed scene graphs with ground-truth labels for training and validation. ScanNet does not. To train our model on ScanNet, we first have to build up a similar graph structure for ScanNet. You can use the following command to generate the graphs for ScanNet

```bash
python open3dsg/data/gen_scannet_subgraphs.py --type [train/test/validation]
```

For the 2D-3D distillation training, we have to align the 2D frames to the 3D point clouds. Using this script we generate matching frames for each 3D instance.
All projection utilities interpret camera coordinates as **y-up** and automatically flip the axis so that pixel rows grow downward.

```bash
python open3dsg/data/get_object_frame.py --mode [train/test] --dataset [R3SCAN/SCANNET]
python -m open3dsg.data.merge_instance_masks --scan <scan_id> --dataset <R3SCAN/SCANNET> --masks_dir <path_to_2d_masks>
```

We pre-process the data before the training for faster data processing in the training loop.

```bash
python open3dsg/data/preprocess_3rscan.py
python open3dsg/data/preprocess_scannet.py
```

The pre-processed features can be used directly for training and testing.

## Reconstruct Point Clouds

To combine per-object point sets from the preprocessing step into a single point cloud, run:

```bash
python open3dsg/scripts/reconstruct_pointcloud.py --graph <data_dict.pkl> --out <scene.ply>
```

The script expects the graph dictionaries produced by [open3dsg/data/preprocess_scannet.py](open3dsg/data/preprocess_scannet.py) or [open3dsg/data/preprocess_3rscan.py](open3dsg/data/preprocess_3rscan.py). The output format is inferred from the `--out` extension (`.ply` or `.npz`).

## Visualize Instance Masks

Overlay projected pixel indices from `*_object2frame.pkl` files onto their
corresponding RGB frames:

```bash
python open3dsg/scripts/visualize_instance_masks.py --scan_dir <scan_dir> \
    --object2frame <scan_id>_object2frame.pkl --out vis --top_k 3
```

This writes files like `42.png` to the output directory.

## Save Frame Projections

Compute visibility statistics for a single instance across all frames of a
scan and store annotated images:

```bash
python scripts/save_frame_projections.py --scan_dir <scan_dir> --instance <inst.ply> --out_dir projections
```

The command writes bounding boxes onto each frame and generates a
`frames.json` file sorted by visibility.

## Find Missing Object Frames

List object IDs without frame associations in the preprocessed dataset:

```bash
python scripts/find_missing_obj_frames.py
```

Use `--out results.json` to save the output instead of printing it.

### Model Downloads

Download the [OpenSeg Checkpoint](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg), [BLIP2 Positional Embedding](https://drive.google.com/file/d/1BfvxB6eo3XksE6AfMUgoBHwzVYce1ed1/view?usp=sharing) & pre-trained [PointNet/PointNet2 weights](https://drive.google.com/drive/folders/1PrnJVMpJVVh4MAV4yPRuRByhBu-DuXwH?usp=sharing) and put them the checkpoints directory selected in the config file.

## Precompute 2D features

This is an **optional** step to accelerate the forward pass in the training loop. This command will dump the VLM features for each training sample to disk. Storing the features requires about 300GB per dataset.

```bash
python open3dsg/scripts/run.py --dump_features --dataset [scannet/3rscan] --scales 3 --top_k_frames 5 --clip_model OpenSeg --blip
```

Running with `--dump_features` only extracts the features without training and disables the learning rate scheduler.

In case of out of memory issues, use `--skip_edge_features` to export only the node
features first. Afterwards run the command again with `--blip` to dump relation
embeddings.
The relation masking step now processes relations in chunks. Adjust
`--rel_chunk_size` (default: 512) to reduce peak memory usage at the cost of
additional passes over the relation features.

### Precompute Full Graphs

Scenes with many objects can exceed GPU memory when dumping features. The helper
script below splits each ScanNet scene into subgraphs, dumps features per
subgraph and merges them back into a full graph.

```bash
python open3dsg/scripts/precompute_full_graph.py --split train --out_dir <output_dir>
```

The script writes `relationships_<split>_full.json` and the merged feature
files into the chosen output directory. These can be used with
`--load_features` for training large scenes.

## Query Node Features

After running `precompute_2d_features`, node embeddings can be searched with a
text query and exported as a colourised point cloud:

```bash
python open3dsg/scripts/query_node_features.py --features <feature_dir> --graph <data_dict.pkl> --scene <scene_id> --word "chair" --topk 5 --out_ply query.ply --log query.txt
```

The command selects the top matching nodes, writes them to `query.ply` with
unique colours and logs their indices with similarity scores to `query.txt`.

## Batch OpenSeg Image Processing

To encode a directory of images with the OpenSeg model and store per-pixel features and a colourised segmentation map run:

```bash
python open3dsg/scripts/openseg_batch_jpeg.py <image_dir> <output_dir> --model_dir checkpoints/openseg --clusters 20
```

The script recursively processes `.jpg`, `.jpeg`, `.png` and `.bmp` files, preserving the input directory structure in the output.

## Profile GPU Memory

To inspect GPU memory usage of each pretrained component run:

```bash
python open3dsg/scripts/vram_profile.py --dataset scannet --clip_model OpenSeg --blip
```

The profiler also supports `--dump_features` to measure memory
consumption during 2D feature precomputing and `--load_features` to
profile training when precomputed features are loaded from disk.
Use `--gpus 2` (or `--gpu_ids 0,1`) to profile all available GPUs
sequentially.


## Train

To train Open3DSG on ScanNet you can use:

```bash
python open3dsg/scripts/run.py --epochs 100 --batch_size 4 --gpus 4 --workers 8 --use_rgb --dataset scannet --clip_model OpenSeg --blip --load_features [path to precomputed 2D features]
```

Change hyperparameters according to you hardware availability. In [run.py](open3dsg/scripts/run.py) you can find more model and data hyperparameters.
Use ```--mixed_precision``` to optimize GPU memory during training.

## Test

To evaluate a trained model on the 3RSCAN dataset with ground-truth labels, use the following command:

```bash
python open3dsg/script/run.py --test --dataset 3rscan --checkpoint [path to checkpoint] --n_beams 5 --weight_2d 0.5 --clip_model OpenSeg --node_model ViT-L/14@336px --blip
```

We use the ```CLIP ViT-L/14@336px``` to query object classes from the node embedding. Use ```--n_beams``` to adjust the beam search for the LLM relationship output and ```--weight_2d``` to adjust the 2D-3D features fusion. A value of 0.0 indicates a prediction from 3D features only

## Citation

If you find our code or paper useful, please cite

```bibtex
@inproceedings{koch2024open3dsg,
      title={Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships},
      author={Koch, Sebastian  and Vaskevicius, Narunas and Colosi, Mirco and Hermosilla, Pedro and Ropinski, Timo},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month={June},
      year={2024},
  }
```

## License

Open3DSG is open-sourced under the AGPL-3.0 license. See the LICENSE file for details.

For a list of other open source components included in Open3DSG, see the file 3rd-party-licenses.txt.
