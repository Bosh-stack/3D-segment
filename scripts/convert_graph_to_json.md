# Convert Open3DSG Preprocessed Graphs to JSON

`convert_graph_to_json.py` converts a preprocessed Open3DSG scene graph and a TSV of edge relations into a compact JSON file.

## Usage

```bash
python scripts/convert_graph_to_json.py --graph_pkl path/to/data_dict.pkl \
    --edge_tsv path/to/edge_relations.tsv --out_json output.json
```

- `--graph_pkl`: Pickle file with a dictionary containing a list of nodes.
- `--edge_tsv`: Tab-separated values file with columns `src_id`, `tgt_id`, `relation` and optional `caption`.
- `--out_json`: Destination JSON file summarising objects and relationships.

The resulting JSON contains two arrays: `objects` and `relationships`, each with the minimal fields required for downstream language model consumption.
