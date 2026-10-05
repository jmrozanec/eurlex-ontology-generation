# Commands to merge ontologies

# merge all partitions not yet done + final global merge
uv run python scripts/run_ontology_merge.py

# merge only partitions, without global merge
uv run python scripts/run_ontology_merge.py --skip-global

# merges also the already merged partitions
uv run python scripts/run_ontology_merge.py --force

# merge of a chosen number of partitions (in this case 1)
uv run python scripts/run_ontology_merge.py --max-partitions 1