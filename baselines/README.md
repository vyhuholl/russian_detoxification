## Baselines
We provide two baselines:
* **Duplicate** — simple duplication of the input;
* **Delete** — removal of rude and toxic from the pre-defined [vocab](https://github.com/vyhuholl/russian_detoxification/blob/master/data/toxic_vocab.txt). You can run the script [`delete.py`](https://github.com/vyhuholl/russian_detoxification/blob/master/baselines/delete.py) with the following parameters:
	* `-i`, `--inputs` — the path to the input dataset written in `.txt` file;
	* `-v`, `--vocab` — the path to the pre-defined vocab of rude and toxic words written in `.txt` file (1 line — 1 word);
	* `-f`, `--file` — the path to the output file.
