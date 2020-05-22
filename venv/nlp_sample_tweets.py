from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens): # tag contains position tag based on pos_tag()
        # remove any extra letter or mark from token
        # once this pattern is matched, replace with empty string ''
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)

        # lemmatize token with position tag
        token = lemmatize(token, tag)

        # filter punctuation and stop_words
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    # return list of cleaned tokens
    return cleaned_tokens

def lemmatize(token, tag):
    # from category determine root of word (being -> be) using wordNetLemmatizer()
    if tag.startswith("NN"): # noun
        pos = 'n'
    elif tag.startswith('VB'): # verb
        pos = 'v'
    else:
        pos = 'a'
    lemmatizer = WordNetLemmatizer()
    # return lemmatized word based on position tag and token
    return lemmatizer.lemmatize(token, pos)


def get_all_words(cleaned_tokens_list): # make tokens list into one
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list): # make a dictionary to define word model as dictionary
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def user_operation_message(classifier):
    end = False

    while not end:
        message = input("Enter your message to see sentiment: \n")
        custom_tokens = remove_noise(word_tokenize(message))

        print(f'Entered message: \n'
            f'{message}\n'
            f'is {classifier.classify(dict([token, True] for token in custom_tokens))}')

        valid_option = False
        while not valid_option:
            try:
                end_option = int(input("Enter (0) for continue\n"
                                        "(1) for Exit: "))
            except ValueError as err:
                print(f'{err}: Enter either 0 or 1')
            else:
                if end_option == 1:
                    end = True
                valid_option = True

def main():
    # get tweets sample of each positive and negative and tokenize using trained nltk tokenizer
    print('Gathering information to train ...\n')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json') # 5000 positive sample tweets
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json') # 5000 negative sample tweets

    # initialize cleaned tokens list for each
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    # set stop words
    stop_words = stopwords.words('english')

    print('Learning ...\n')
    # clean all positive tweets
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # clean all negative tweets
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # put all tweets tokens into one list
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    # get distribution of word frequency
    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    # make a model for each as dictionary
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    # attach label of Positive or Negative
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    # make it one list and shuffle
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    print('hmm ...\n')
    # 70% train and 30% test data
    train_data = dataset[:int(len(dataset) * .7)]
    test_data = dataset[int(len(dataset) * .3):]

    # train data to build model
    classifier = NaiveBayesClassifier.train(train_data)

    print(f'Accuracy of classify: {classify.accuracy(classifier, test_data) * 100} %')
    print(classifier.show_most_informative_features(10))

    user_operation_message(classifier)

if __name__ == "__main__":
    main()