Models for automatic detoxification of Russian texts.
## Data
Folder `data` consists of:
* `data/train.csv` — train dataset, 262702 texts (213271 non-toxic, 37073 toxic);
* `data/test.txt` — test dataset, 12358 toxic texts;
* `data/parallel_corpus.csv` — parallel train dataset, 500 samples.
## Models
### Baselines
* **Duplicate** — simple duplication of the input;
* **Remove** — removal of rude and toxic from pre-defined [vocab](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/MAT_FINAL_with_unigram_inflections.txt);
* **Retrieve** — retrieval based on cosine similarity between word embeddings from non-toxic part of [RuToxic](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/ru_toxic_dataset.csv) dataset;
## Evaluation
The evaluation consists of three types of metrics:
* **style transfer accuracy (STA)** — accuracy based on toxic/non-toxic classifier (we suppose that the resulted text should be in non-toxic style);
* **cosine similarity (CS)** — similarity between vectors of the input and the resulted texts' embeddings
* **perplexity (PPL)** — perplexity based on language model.
Finally, **aggregation metric**: geometric mean between STA, CS and the inverse of PPL.
You can run the [`metric.py`](https://github.com/vyhuholl/russian_detoxification/blob/master/metric.py) script for evaluation.
## Results
Method | STA↑ | CS↑ | PPL↓ | GM↑
------ | ---- | --- | ----| ---
**Baselines** |
