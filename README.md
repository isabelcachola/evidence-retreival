# Question answering as evidence retrieval 
 This repo is the final project for Information Retrieval and Web Agents 601.666.

## Requirements

This code has been tested on Python 3.9.12.

To install required dependencies, run `pip install -r requirements.txt`.

## Data
To speed search, I include the documents preprocessed in `processed_docs`.
The `processed_docs/test.v2/` directory contains preprocessed documents (removed stopwords and stemmed). 
The `processed_docs/test.v2.freqs.pkl`contains precomputed document frequencies. 

The `tobacco_data` directory includes the following files:
#### `annotated_data_100.v2.tsv`
 The raw annotated data, including queries, stories, and highlighted answers.
#### `query.v2.evidence` 
Each line is a space separated annotation with the query id, relevant document id, start token index, and end token index. The token indexes represented the highlighted answers to the queries.

#### `query.v2.raw`
Contains queries, formatted as follows:
```
.I {query id}
.W
 {query text}
 ```
#### `tobacco_test.v2.raw`
Contains news data. formatted as follows:
```
.I {doc id}
.T 
{title}
.W 
{document text}
.A 
{source api}
```

## Usage

The `main.py` script includes 3 modes to run: command line, interactive, and experiment mode. Below are example usages. For full description of options, use `python main.py -h`.

### Command line mode
This mode allows the user to input a query using the `--query` argument, returning the top 20 documents and highlighted answers.

Usage example:
```
$ python main.py --query "What regulations has the FDA passed"
Loading processed_docs/test.v2.freqs.pkl
Query: what regulations has the fda passed ?
1) california declares electronic cigarettes a health threat
        '' the u.s. food and drug administration is also proposing regulations that include warning labels and ingredient lists on e-cigarettes , although enactment could take years . california health officials
2) altria has a bullish outlook - gurufocus.com ( registration )
        . there is a decline in smoking rates , and the regulations are increasing . it is safer to invest in this company since
...
```
### Interactive mode
This mode prompts the user to input a query, returning the top 20 documents and highlighted answers for each query. Enter `q` or `quit` to exit the program.

Usage example:
```
$ python main.py --interactive  
Query: What is the effect of second hand smoke?
Keywords: tobacco fda
Authors/API: 
Query: what is the effect of second hand smoke ?
1) fda tobacco director ignores 2.5 million 'anecdotal reports ' about e-cigarettes
        this huge tax increase . `` these data show a dramatic rise in usage of e-cigarettes by youth , and this is cause for great concern as we
2) eliminate the risk of secondhand smoke to your family
        that secondhand smoke exposure causes more than 41,000 deaths from heart disease and lung cancer among non-smoking adults each year . secondhand smoke contains more
...
Query: q
```

### Experiement mode

This mode is the default behvaior evaluates different parameters and outputs the performance. To use, simply run `python main.py`.
