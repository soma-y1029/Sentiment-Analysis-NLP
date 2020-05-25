from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


class Model:
    def __int__(self):
        pass

    def remove_noise(self, tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens): # tag contains position tag based on pos_tag()
            # remove any extra letter or mark from token
            # once this pattern is matched, replace with empty string ''
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub('(@[A-Za-z0-9_]+)','', token)

            # lemmatize token with position tag
            token = self.lemmatize(token, tag)

            # filter punctuation and stop_words
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        # return list of cleaned tokens
        return cleaned_tokens

    def lemmatize(self, token, tag):
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


    def get_all_words(self, cleaned_tokens_list): # make tokens list into one
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(self, cleaned_tokens_list): # make a dictionary to define word model as dictionary
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def user_operation_message(self, classifier):
        end = False

        while not end:
            message = input("Enter your message to see sentiment: \n")
            custom_tokens =  self.remove_noise(word_tokenize(message))
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

    def get_samples(self, source: string, stop_words):
        # get tweets sample of each positive and negative and tokenize using trained nltk tokenizer
        dest = twitter_samples.tokenized(source)
        cleaned_tokens = []

        # clean all tweets
        for tokens in dest:
            cleaned_tokens.append(self.remove_noise(tokens, stop_words))
        return cleaned_tokens

    def build_model(self):
        # set stop words
        stop_words = stopwords.words('english')

        print('Gathering information to train ...\n')

        # get tweets sample of each positive and negative and tokenize using trained nltk tokenizer
        positive_cleaned_tokens_list = self.get_samples('positive_tweets.json', stop_words)
        negative_cleaned_tokens_list = self.get_samples('negative_tweets.json', stop_words)
        print(len(positive_cleaned_tokens_list), len(negative_cleaned_tokens_list))

        # put all tweets tokens into one list
        all_pos_words = self.get_all_words(positive_cleaned_tokens_list)

        # get distribution of word frequency
        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(10))

        # make a model for each as dictionary
        positive_tokens_for_model = self.get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = self.get_tweets_for_model(negative_cleaned_tokens_list)

        # attach label of Positive or Negative
        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]
        # make it one list and shuffle
        dataset = positive_dataset + negative_dataset
        random.shuffle(dataset)

        print('Learning ...\n')
        # 70% train and 30% test data
        train_data = dataset[:int(len(dataset) * .7)]
        test_data = dataset[int(len(dataset) * .3):]

        # train data to build model
        classifier = NaiveBayesClassifier.train(train_data)

        print(f'Accuracy of classify: {classify.accuracy(classifier, test_data) * 100} %')
        print(classifier.show_most_informative_features(10))

        return classifier

if __name__ == "__main__":
    model = Model()
    classifier = model.build_model()
    model.user_operation_message(classifier)