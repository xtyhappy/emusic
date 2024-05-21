import gensim
import numpy as np
import time
import datetime

found = 0


def compute_ngrams(word, min_n, max_n):
    extended_word = word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))


def word_vector(word, wv_from_text, min_n=1, max_n=3):
    # Confirm word vector dimensions
    word_size = wv_from_text.vectors[0].shape[0]
    # Compute ngrams for the word
    ngrams = compute_ngrams(word, min_n=min_n, max_n=max_n)
    # If the word is in the dictionary, return the word vector
    if word in wv_from_text.index_to_key:
        global found
        found += 1
        return wv_from_text[word]
    else:
        # If not in the dictionary, calculate word vectors similar to the word
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
        # Accept only word vectors with a length of more than 2 words
        for ngram in ngrams_more:
            if ngram in wv_from_text.index_to_key:
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
        # If no match, then consider single word vectors
        if ngrams_found == 0:
            for ngram in ngrams_single:
                if ngram in wv_from_text.index_to_key:
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
        if word_vec.any():  # As long as one is not zero
            return word_vec / max(1, ngrams_found)
        else:
            print('all ngrams for word %s absent from model' % word)
            return 0


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def part3_main(emotion_word, scene_words):
    # Inherit emotion_word and scene_words from another program
    # emotion_word = "happy"  # Replace with the actual emotion word
    # scene_words = {"sun", "rain", "beach"}  # Replace with the actual set of scene words

    # Create a set to store words that will always be part of input_set as its last few elements
    concrete_words = {"music", "emotion", "emotional", "genre", "music%20type", "musical%20style"}

    # Create a set to store words that should not appear in the output results
    forbidden_words = {"sound", "scene", "music%20genre", "the%20beat", "sound%20in", "music%20like", "music%20beat",
                       "kind%20of%20music", "music%20today", "music%20style", "type%20of%20music", "beat%20music",
                       "style%20of%20music", "genre%20of%20music", "musical%20genre", "form%20of%20music"}

    print("开始载入文件...")
    print("Now：", datetime.datetime.now())
    t1 = time.time()
    wv_from_text = gensim.models.KeyedVectors.load(r'ChineseEmbedding.bin', mmap='r')
    print("文件载入完毕")
    print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")

    # Construct the input set
    input_set = {emotion_word}
    input_set.update(scene_words)
    input_set.update(concrete_words)  # Add additional words to the input_set

    # Calculate the average vector for the input words
    avg_vec = np.mean([word_vector(keyword, wv_from_text, min_n=1, max_n=3) for keyword in input_set if
                       word_vector(keyword, wv_from_text, min_n=1, max_n=3) is not 0], axis=0)

    # Find the top 15 most similar words to the average vector, excluding those with cosine similarity above 90%
    if avg_vec.any():
        similar_word = wv_from_text.most_similar(positive=[avg_vec], topn=15 + len(input_set))
        result_word = [x[0] for x in similar_word if x[0] not in input_set and x[0] not in forbidden_words and all(
            cosine_similarity(avg_vec, word_vector(keyword, wv_from_text, min_n=1, max_n=3)) < 0.9 for keyword in
            input_set)][:15]
        print(result_word)

    print("词库覆盖比例：", found, "/", len(input_set))
    print("词库覆盖百分比：", 100 * found / len(input_set), "%")
    print("整个推荐过程耗费时间：", (time.time() - t1) / 60, "minutes")
    return result_word
