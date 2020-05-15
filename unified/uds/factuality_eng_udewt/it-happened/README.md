# It Happened (v1.0, 2.0)

This archive contains data collected using annotation protocols described in the following papers.

White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. [Universal decompositional semantics on universal dependencies](http://aclweb.org/anthology/D/D16/D16-1177.pdf). In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 1713–1723, Austin, Texas, November 1-5, 2016.

Rudinger, R., A.S. White, & B. Van Durme. 2018. [Neural models of factuality](http://aswhite.net/papers/rudinger_neural_2018.pdf). To appear in *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics*, New Orleans, Louisiana, June 1-6, 2018.

If you make use of these datasets in a presentation or publication, we ask that you please cite these papers.

## Contents

The file `it-happened_eng_ud1.2_10262016.tsv` corresponds to the data reported in White et al. 2016, and the file `it-happened_eng_ud1.2_07092017.tsv` corresponds to the data reported in Rudinger et al. 2018. The latter file contains a strict superset of the data found in the former. Both are stand-off annotations with pointers into the English Web Treebank portion of English Universal Dependencies v1.2. 

The column descriptions and values for `it-happened_eng_ud1.2_10262016.tsv` can be found below. 

| Column            | Description       | Values            |
|-------------------|-------------------|-------------------|
| Split             | The train-dev-test split from the English Web Treebank | `train`, `dev`, `test` |
| Annotator.ID      | The annotator that provided the response | `0`, ..., `23` |
| Display.Position  | The position in which the predicate was displayed in the HIT starting at 1 | `1`, ..., `10`  |
| Sentence.ID       | The file and sentence number of the sentence in the English Universal Dependencies v1.2 treebank with the format `LANGUAGE-CORPUS-SPLIT.ANNOTATION SENTNUM` | see data |
| Pred.Token        | The position of the predicate in the sentence starting at 1 | `1`, ..., `155` |
| Pred.Lemma        | The predicate lemma | see data |
| Is.Predicate      | Whether the word in question is a predicate | `true`, `false` |
| Is.Understandable | Whether the sentence in question is understandable | `true`, `false` |
| Happened          | Whether the eventuality described by the predicate happened or is happening | `yes`, `no`, `nan` |
| Confidence        | How confident the annotator was in the `Happened` judgment | `0`, `1`, `2`, `3`, `4`, `nan` |

The column descriptions and values for `it-happened_eng_ud1.2_07092017.tsv` can be found below. 

| Column            | Description       | Values            |
|-------------------|-------------------|-------------------|
| Split             | The train-dev-test split from the English Web Treebank | `train`, `dev`, `test` |
| Annotator.ID      | The annotator that provided the response | `0`, ..., `46` |
| Display.Position  | The position in which the predicate was displayed in the HIT starting at 1 | `1`, ..., `10`  |
| Sentence.ID       | The file and sentence number of the sentence in the English Universal Dependencies v1.2 treebank with the format `LANGUAGE-CORPUS-SPLIT.ANNOTATION SENTNUM` | see data |
| Pred.Token        | The position of the predicate in the sentence starting at 1 | `1`, ..., `155` |
| Pred.Lemma        | The predicate lemma | see data |
| Is.Predicate      | Whether the word in question is a predicate | `true`, `false` |
| Is.Understandable | Whether the sentence in question is understandable | `true`, `false` |
| Happened          | Whether the eventuality described by the predicate happened or is happening | `yes`, `no`, `nan` |
| Confidence        | How confident the annotator was in the `Happened` judgment | `0`, `1`, `2`, `3`, `4`, `nan` |