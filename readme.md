# Text ClassifiKation

helpers to search for best pipeline and classifier models

 
## Install

Available on pip

    pip install text_classifikation
    
from source

    git clone https://github.com/JarbasAl/text_classifikation
    cd text_classifikation
    pip install -r requirements.txt
    pip install .
    
## Usage

Create a Classifier base class for your problem and implement dataset loading

```python
from text_classifikation.classifiers import BaseClassifier

class QuestionClassifier(BaseClassifier):   
    

    @staticmethod    
    def load_data(filename=join(DATA_PATH, "questions.txt")):        
        train_data = []
        target_data = []
        with open(filename, 'r') as f:
            for line in f:
                # each line in dataset is LABEL SENTENCE 
                label = line.split(" ")[0]
                question = " ".join(line.split(" ")[1:])
                train_data.append(question.strip())
                target_data.append(label.strip())
        return train_data, target_data

    # optional, just ensuring correct file is chosen by default
    def load_test_data(self, filename=join(DATA_PATH, "questions_test.txt")):
        
        return self.load_data(filename)
        
        
```

Select a classifier model to try
```python

from text_classifikation.classifiers.svm import LinearSVCTextClassifier
    
class LinearSVCQuestionClassifier(QuestionClassifier, LinearSVCTextClassifier):
    pass
    
```

Use existing feature extraction pipelines

Default Feature Extractors:

- CountVectorizer with up to 3 ngrams (optionally input is lemmatized)
- TfidfVectorizer with up to 3 ngrams (optionally input is lemmatized)
- PosTagVectorizer with one-hot encoding
- NERVectorizer with one-hot encoding
- Word2VecVectorizer for word embeddings (optionally input is lemmatized)

```python
from text_classifikation.classifiers.pipelines import default_pipelines, \
    default_pipeline_unions
    
# different features
assert sorted(list(default_pipelines.keys())) == ['cv',
                                                  'cv2',
                                                  'cv2_lemma',
                                                  'cv3',
                                                  'cv3_lemma',
                                                  'cv_lemma',
                                                  'ner',
                                                  'postag',
                                                  'tfidf',
                                                  'tfidf2',
                                                  'tfidf2_lemma',
                                                  'tfidf3',
                                                  'tfidf3_lemma',
                                                  'tfidf_lemma',
                                                  'w2v',
                                                  'w2v_lemma']

# combinations of previous features
assert sorted(list(default_pipeline_unions.keys())[:8]) == ['tfidf2_cv3_lemma',
                                                            'tfidf2_lemma_cv3_w2v_lemma',
                                                            'tfidf3_cv3_w2v_lemma',
                                                            'tfidf3_lemma_cv2_lemma_w2v_lemma',
                                                            'tfidf_cv3_lemma_w2v_lemma',
                                                            'tfidf_lemma_cv2_w2v_lemma',
                                                            'w2v_lemma_cv3_tfidf3',
                                                            'w2v_tfidf3_lemma']
                                                            
                                                            

class QuestionClassifierPipelineX(QuestionClassifier):
    @property
    def pipeline(self):
        # text features = word2vec, count vectorizer with ngrams=(1,2), tfidf with with ngrams=(1,2)
        return [
            ('text', default_pipeline_unions["tfidf2_cv2_w2v"])
            ('clf', self.classifier_class)
        ]
```

Add new feature extractors

```python
from sklearn.pipeline import Pipeline
from little_questions.classifiers.features import DictTransformer
from sklearn.feature_extraction import DictVectorizer


pipeline__intent = Pipeline([('dict', DictTransformer()),
                             ('dict_vec', DictVectorizer())])

base_pipelines["intentdict"] = pipeline__intent
```

Experiment with pipelines to improve accuracy
```python

def find_best_pipeline(clf):
    train, train_label = clf.load_data()
    test, test_label = clf.load_test_data()
    best_score, best_pipeline, acs = clf.find_best_pipeline(train, train_label,
                                                           test, test_label)
    return best_score, best_pipeline
    
name = "questions_svc"
clf = LinearSVCQuestionClassifier(name)
best_score, best_pipeline = find_best_pipeline(clf)
print("BEST:", best_pipeline, "ACCURACY:", best_score)
```


Test specific feature combinations

```python
# select features to join
from text_classifikation.classifiers.pipelines import generate_unions

_independent_components = {
    "w2v": ("w2v", "w2v_lemma"),
    "postag": ("postag")
}
base_pipeline_unions = generate_unions(base_pipelines, _independent_components)
    

# search new best pipeline
def find_best_pipeline(clf, train_data, target_data, test_data, test_label, 
                        pipelines=None, unions=None, outfolder=None, 
                        save_all=True, skip_existing=True, verbose=True):
    pipelines = pipelines or base_pipelines
    unions = unions or base_pipeline_unions
    clf.find_best_pipeline(train_data, target_data, test_data, test_label, 
                            pipelines, unions, outfolder, save_all, 
                            skip_existing, verbose)   
                              
name = "questions_svc"
clf = LinearSVCQuestionClassifier(name)
best_score, best_pipeline = find_best_pipeline(clf)
print("BEST:", best_pipeline, "ACCURACY:", best_score)                 
```

Experiment with hyperparams to improve accuracy
```python
# TODO grid search
```

## Projects using this package

- [Little Questions](https://github.com/JarbasAl/little_questions) - classify questions