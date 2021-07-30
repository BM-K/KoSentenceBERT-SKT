# Ko-Sentence-BERT-SKTBERT

- 다음 레파지토리의 ETRI KoBERT의 학습된 파일을 공개하지 못함 <br>
          - https://github.com/BM-K/KoSentenceBERT
- SKT KoBERT 사용 학습 <br>
          - https://github.com/SKTBrain/KoBERT

## Installation
- **huggingface transformer, sentence transformers, tokenizers** 라이브러리 코드를 직접 수정하므로 가상환경 사용을 권장합니다.  <br>
- 사용한 Docker image는 Docker Hub에 첨부합니다. <br>
    - https://hub.docker.com/r/klbm126/kosbert_image/tags <br>

```
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
cd ..
git clone https://github.com/BM-K/KoSentenceBERT_SKTBERT.git
pip install -r requirements.txt
```
 - transformer, tokenizers, sentence_transformers 디렉토리를 opt/conda/lib/python3.7/site-packages/ 로 이동합니다. <br>

## Train Models
 - 모델 학습을 원하시면 KoSentenceBERT 디렉토리 안에 KorNLUDatasets이 존재하여야 합니다. <br>
 - STS를 학습 시 모델 구조에 맞게 데이터를 수정하였으며, 데이터와 학습 방법은 아래와 같습니다 : <br><br>
KoSentenceBERT/KorNLUDatates/KorSTS/tune_test.tsv <br>
<img src="https://user-images.githubusercontent.com/55969260/93304207-97afec00-f837-11ea-88a2-7256f2f1664e.png"></img><br>
*STS test 데이터셋의 일부* <br>
```
python training_nli.py      # NLI 데이터로만 학습
python training_sts.py      # STS 데이터로만 학습
python con_training_sts.py  # NLI 데이터로 학습 후 STS 데이터로 Fine-Tuning
```

## Pre-Trained Models
**pooling mode**는 **MEAN-strategy**를 사용하였으며, 학습시 모델은 output 디렉토리에 저장 됩니다. <br>
|디렉토리|학습방법|
|-----------|:----:|
|training_**nli**|Only Train NLI|
|training_**sts**|Only Train STS|
|training_**nli_sts**|STS + NLI|
<br>
학습된 pt 파일은 다음 드라이브에 있습니다. <br>https://drive.google.com/drive/folders/1fLYRi7W6J3rxt-KdGALBXMUS2W4Re7II?usp=sharing 
<br>
<img src='https://user-images.githubusercontent.com/55969260/101112870-9cf42a00-3621-11eb-9bc0-788ba08638e1.png'>
각 폴더에 있는 result파일을 output 디렉토리에 넣으시면 됩니다. <br>
ex) sts 학습 파일 사용시 위 드라이브에서 sts/result.pt 파일을 output/training_sts/0_Transformer에 넣으시면 됩니다. <br>
output/training_sts/0_Transformer/result.pt <br>

## Performance
Seed 고정, test set
|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|NLl|65.05|68.48|68.81|68.18|68.90|68.20|65.22|66.81|
|STS|**80.42**|**79.64**|**77.93**|77.43|**77.92**|77.44|76.56|75.83|
|STS + NLI|78.81|78.47|77.68|**77.78**|77.71|**77.83**|75.75|75.22|

## Application Examples
 - 생성 된 문장 임베딩을 다운 스트림 애플리케이션에 사용할 수 있는 방법에 대한 몇 가지 예를 제시합니다.
 - STS pretrained 모델을 통해 진행합니다.

### Semantic Search
SemanticSearch.py는 주어진 문장과 유사한 문장을 찾는 작업입니다.<br>
먼저 Corpus의 모든 문장에 대한 임베딩을 생성합니다.
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_sts'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['한 남자가 파스타를 먹는다.',
           '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
           '치타가 들판을 가로 질러 먹이를 쫓는다.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
        
```
<br> 결과는 다음과 같습니다 :
```
======================


Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.6800)
한 남자가 빵 한 조각을 먹는다. (Score: 0.6735)
한 남자가 말을 탄다. (Score: 0.1256)
두 남자가 수레를 숲 솦으로 밀었다. (Score: 0.1077)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0968)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.6832)
한 여자가 바이올린을 연주한다. (Score: 0.2885)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2278)
그 여자가 아이를 돌본다. (Score: 0.2018)
한 남자가 말을 탄다. (Score: 0.1397)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.8141)
두 남자가 수레를 숲 솦으로 밀었다. (Score: 0.3707)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.1842)
한 남자가 말을 탄다. (Score: 0.1716)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1519)

```
### Clustering
Clustering.py는 문장 임베딩 유사성을 기반으로 유사한 문장을 클러스터링하는 예를 보여줍니다. <br>
이전과 마찬가지로 먼저 각 문장에 대한 임베딩을 계산합니다. <br>
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_sts'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.',
          '한 남자가 파스타를 먹는다.',
          '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
          '치타가 들판을 가로 질러 먹이를 쫓는다.']

corpus_embeddings = embedder.encode(corpus)

# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")

```
결과는 다음과 같습니다 :
 ```
Cluster  1
['그 여자가 아이를 돌본다.', '원숭이 한 마리가 드럼을 연주한다.', '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.']

Cluster  2
['한 남자가 음식을 먹는다.', '한 남자가 빵 한 조각을 먹는다.', '한 남자가 파스타를 먹는다.']

Cluster  3
['치타 한 마리가 먹이 뒤에서 달리고 있다.', '치타가 들판을 가로 질러 먹이를 쫓는다.']

Cluster  4
['한 남자가 말을 탄다.', '두 남자가 수레를 숲 솦으로 밀었다.', '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.']

Cluster  5
['한 여자가 바이올린을 연주한다.']

```

## Citing
### KorNLU Datasets
```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
### Sentence Transformers: Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}

@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```

