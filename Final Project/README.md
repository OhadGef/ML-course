#final project
    By Nadav Gilron & Ohad Gefen

## Description:

This project is about choosing a data-set and by using the RNN deep learing algorithem (Recurrent Neural Networks) we made a language model and generate new text.



### Install the requirments 

```python
#to run the priject you need to install all
 pip install -r requirments.txt
```

### Collecting Data

<dl>
The Data that we choosing to work was Louis CK jokes and that was out language model to learn from.
<dt>Process steps:</dt>
<dd>
1. we took a collection of Louis CK jokes and gather them in a .csv file.
</dd>
<dd>
2. we removed from the data-set unnecessary syntax like - " , ... and so, to make the result close and similar to the original. 
</dd>
</dl>

<br>

### Training the Data and Preprocessing 

After reading the data from CSV file we Tokenized it to words for geting predictions on a per-word basis, the tokenize is not only sliting each of the sentences by spaces.
We'll using [NLK's tool kit](http://www.nltk.org/) methods.

In this program we limit the vocabulary size to 1560 most common words, the wordes that not in the vocabulary we replace by UNKNOWN_TOKEN. we also want to learn which words tend to start and end a sentence, for that we have SENTENCE_START and SENTENCE_END token to each sentence.

The RNN algorithem will work with vectors so we are create a mapping between words and index by using INDEX_TOWORD and WORD_TO_INDEX.
X and Y is the vectores and the differance is that X is shifting by one position with the last element being the SENTENCE_END token.


```python
vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/LouisCK.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
    
    
    
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

```

### RNN initialization
 
```python
_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '1560'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
```

We set the hidden dimension to 100 that represent the memory of the network, lerning rate to 0.005 nad the nepoch to 50 to iterate the data, that's 158500 parameters.


### Create new text 

After we finished creating the modle we can make it create new text for us and save it to CSV file. the results: 
```
laundry good my nice anyone came my matter it screwed be ketchup .                                                                        
little ' to a ask 'm losing feels young have at .                                                                                         
positive my beat invented try really lied stage conversation been that .                                                                  
ruin lead writing yeast lines your earth n't goes degenerate really a vulgar was .                                                        
find that selfish to to reality them hurt was n't control .

```

### Similarity between sentence
To find the similarity we used cosine vector similarity between every line in the original CSV against the new and foun 4%-10% percent similarity.

```
sentence: 1 0.105112563825                                                                                                                  
sentence: 2 0.0482884316282                                                                                                                 
sentence: 3 0.0673356665992                                                                                                                 
sentence: 4 0.108316685607

```
### Summary and Conclusions

we find Louis CK a very talented and funny stand-up comedian that have a large vocabulary and complex sentences.
So The challenge in working with this data-set is to try and make new funny jokes (or at least reasonable sentences).
Our conclusion is that if you wan't to make funny jokes ask Louis CK to do that and dont use RNN algorithm



