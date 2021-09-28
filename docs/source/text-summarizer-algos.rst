Text Summarizer
===============

Overview
--------

Text summarization is a task of producing a concise and fluent summary
while preserving key information content and overall meaning. Assume
there is a document :math:`D` that consists of :math:`n` sentences: :math:`(S_0,
S_1, ..., S_{n-1})`. The problem of text summarization can be formulated
as creating another document :math:`D’`, where :math:`D’ = (S’_0, S’_1, ..., S’_{m-1})`
and :math:`m \le n`.

In general, there are two categories of text summarization:
**extractive summarization** and **abstractive summarization**. If for
each :math:`i` in :math:`[0, m), S’_i` is in :math:`{S_0, S_1, ...,
S_{n-1}}`, the summarization is **extractive summarization**. If there
exists :math:`j` in :math:`[0, m), S’_j` is not in :math:`{S_0, S_1, ..., S_{n-1}}`, the
summarization is **abstractive summarization**.

To create a summarization, the conventional approaches are to first rank
the sentences in :math:`D`. Then the top :math:`m` scoring sentences in :math:`D`
are selected as the summary :math:`D’`. The key to these approaches is how
to define a sorting metric.

Another summarization approach is k-means. Each sentence in :math:`D` is
represented as a numerical vector, and :math:`D` will be modeled as :math:`m`
clusters of numerical vectors. The distance metric can be Euclidean,
Cosine, or Manhattan distance. The sentences closest to the :math:`m` cluster
centroids will be chosen as the sentences in :math:`D’`. The numerical
representations of :math:`D’` sentences can be generated from sentences’
embeddings, for example, Gensim’s Doc2Vec.

The **extractive** method is more practical because the summaries it
creates are more grammatically correct and semantically relevant to the
document. So, the library's text summarizers take the **extractive** approach.

The library's text summarizer implements two versions of
extractive summarizers: **JaccardSummarizer** and
**KMedoidsSummarizer**.

Custom Vocabulary
-----------------

The library's text summarizer can be customized for specific use-cases.
For example, an analyst and might want to use
distinct summarizers with their own vocabulary.

To achieve this, you can simply specify a custom vocabulary list.
During a processing job of the library's summarizer,
only the words in the custom vocabulary are extracted.

This vocabulary customization feature is implemented in **JaccardSummarizer.**
You can
specify your own vocabulary when instantiating a :class:`~smjsindustry.Summarizer`
with :class:`~smjsindustry.JaccardSummarizerConfig`.

JaccardSummarizer
-----------------

This summarizer first preprocesses the document in question to obtain a
set of tokens for each sentence in the document. The preprocessing is
based on a bag of words model. The document is first segmented into a
list of sentences by `Natural Language Toolkit <https://www.nltk.org/>`_’s (NLTK)
``sent_tokenize`` method. Then each sentence is
further tokenized by NLTK’s ``regexp_tokenize`` method, which removes
numbers, punctuations, and spaces from the sentence. Next, stop words are
removed and stemming is applied to the remaining tokens.

JaccardSummarizer is a traditional summarizer. It scores the
sentences in a document measuring similarities. The sentences with higher
similarities to other sentences in the documents are ranked higher. The
top scoring sentences are selected as the summary of the document.

More specifically, the similarity is calculated in terms of `Jaccard
similarity <https://en.wikipedia.org/wiki/Jaccard_index>`__. The Jaccard
similarity of two sentences **A** and **B** is the ratio of the size of
intersection of tokens in **A** and **B** vs the size of union of tokens
in **A** and **B**.

.. math::

   J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A|+|B|-|A \cap B|}


Accordingly, it calculates a symmetric Jaccard similarity matrix for
the sentences in the document. Each row and column in the matrix
correspond a sentence in the document.

.. math::

   J_{similarity} = \left[ \begin{array}{cccc}
   j_{0,0} & j_{0,1} & ... & j_{0,n-1} \\
   j_{1,0} & j_{1,1} & ... & j_{1,n-1} \\
   ... & ... & ... & ... \\
   j_{n-1,0} & j_{n-1,1} & ... & j_{n-1,n-1} \\
   \end{array} \right]

Finally, the score of a sentence is the row sum of this sentence’s similarities
to all other sentences in the document.

.. math::

   score_i = \sum_k j_{ik}

Then the top :math:`m` sentences with the highest similarities are selected
via heap sorting or quick select.


KMedoidsSummarizer
------------------

KMedoidsSummarizer is a k-means based approach. It creates
the sentence embeddings using `Gensim’s Doc2Vec
<https://radimrehurek.com/gensim/models/doc2vec.html>`_. Then it uses the
`k-medoids <https://en.wikipedia.org/wiki/K-medoids>`_ algorithm to determine
the :math:`m` sentences in the document closest to the
cluster centroids.
