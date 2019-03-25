"""
=======================
4.2. Feature extraction
=======================
将数据转化成用于机器学习的特征矩阵

DictVectorizer: 字典转换成矩阵 DictVectorizer()
    vec = DictVectorizer()
    vec.fit_transform(measurements).toarray()

The Bag of Words representation： CountVectorizer()
    所有词在文档中发生的词数，形成的一条数据代表一条样例
    n-gram:
        n-gram是一种统计语言模型，用来根据前(n-1)个item来预测第n个item
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    缺点：常见词的比重较大

Tf–idf: 词频-逆文本频率 TfidfTransformer()
    一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章.

    词频 (term frequency, TF)  指的是某一个给定的词语在该文件中出现的次数
    逆向文件频率 (inverse document frequency, IDF)  IDF的主要思想是：如果包含词条t的文档越少, IDF越大，
        则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，
        再将得到的商取对数得到。

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

解码技巧：
    import chardet
    decoded = [x.decode(chardet.detect(x)['encoding'])
                   for x in (text1, text2, text3)]
"""


def example():
    measurements = [
        {'city': 'Dubai', 'temperature': 33.},
        {'city': 'London', 'temperature': 12.},
        {'city': 'San Francisco', 'temperature': 18.},
    ]

    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()

    print(vec.fit_transform(measurements).toarray())
    print(vec.get_feature_names())

    ###################################################
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
        'Is this the first document?',
    ]
    X = vectorizer.fit_transform(corpus)

    analyze = vectorizer.build_analyzer()
    print(analyze("Is this the first document?"))
    print(vectorizer.get_feature_names())
    print(X.toarray())

    print(vectorizer.transform(['Something completely new.']).toarray())
    #######################################################
    # 2 - grams
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                        token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    print(analyze('Bi-grams are cool!'))
    X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
    print(X_2)

    ##########################################################
    # tf - idf
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X_2)
    print(tfidf.toarray())

    #######################################################
    # 解码
    import chardet
    text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
    text2 = b"holdselig sind deine Ger\xfcche"
    text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
    decoded = [x.decode(chardet.detect(x)['encoding'])
               for x in (text1, text2, text3)]
    v = CountVectorizer().fit(decoded).vocabulary_
    for term in v:
        print(term)


if __name__ == '__main__':
    pass
