Models for automatic detoxification of Russian texts.
## Data
Folder [`data`](https://github.com/vyhuholl/russian_detoxification/tree/master/data) consists of:
* [`data/train.csv`](https://github.com/vyhuholl/russian_detoxification/blob/master/data/train.csv) — train dataset, 262702 texts (213271 non-toxic, 37073 toxic);
* [`data/test.txt`](https://github.com/vyhuholl/russian_detoxification/blob/master/data/test.txt) — test dataset, 12358 toxic texts;
* [`data/parallel_corpus.csv`](https://github.com/vyhuholl/russian_detoxification/blob/master/data/parallel_corpus.csv) — parallel train dataset, 500 samples;
* [`data/toxic_vocab.txt`](https://github.com/vyhuholl/russian_detoxification/blob/master/data/toxic_vocab.txt) — pre-defined vocab of rude and toxic words, 139490 words.
## Models
### Baselines
We provide two baselines:
* **Duplicate** — simple duplication of the input;
* **Delete** ([`baselines/delete.py`](https://github.com/vyhuholl/russian_detoxification/blob/master/baselines/delete.py)) — removal of rude and toxic from the pre-defined [vocab](https://github.com/vyhuholl/russian_detoxification/blob/master/data/toxic_vocab.txt).
## Evaluation
The evaluation consists of three types of metrics:
* **style transfer accuracy (STA)** — the average confidence of the pre-trained BERT-based toxic/non-toxic text classifier (`SkolkovoInstitute/russian_toxicity_classifier`). We suppose that the resulted texts should be in non-toxic style);
* **cosine similarity (CS)** — the average distance of embeddings of the input and output texts. The embeddings are generated with the [FastText Skipgram](http://vectors.nlpl.eu/repository/20/213.zip) model;
* **fluency score (FL)** — the average difference in confidence of the pre-trained BERT-based corrupted/non-corrupted text classifier (`SkolkovoInstitute/rubert-base-corruption-detector`) between the input and output texts. We suppose that both the input and the output texts should not be corrupted.

Finally, **joint score (JS)**: the sentence-level multiplication of the **STA**, **SIM**, and **FL** scores.

You can run the [`metric.py`](https://github.com/vyhuholl/russian_detoxification/blob/master/metric.py) script for evaluation with the following parameters:
* `-i`, `--inputs` — the path to the input dataset written in `.txt` file;
* `-p`, `--preds` — the path to the file of model's prediction written in `.txt` file;
* `-b`, `--batch_size` — batch size for the classifiers, default value 32;
* `-m`, `--model` — the name of your model, is empty by default;
* `-f`, `--file` — the path to the output file. If not specified, results will not be written to a file.
## Results
Method | STA↑ | CS↑ | FL↑ | JS↑
------ | ---- | --- | --- | ---
**Baselines** |
Duplicate | 0.07 | 1.00 | 0.00 | **0.00**
