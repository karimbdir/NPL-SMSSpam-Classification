from Preprocessing.preprocess import parse_text_file, describe, explore_class, messages_length, missing, \
    length_describe, longest_message, count_class, count_class_hist, pie_class, text_process, vectorization, \
    tfidf_function, split_data, NB, decision_tree, forest_tree, ada, support_vector_machine, logistic_regression

if __name__ == '__main__':


    messages = parse_text_file('data/SMSSpamCollection.txt')
    #print(messages)


    #messages_describe = describe(messages)
    #print(messages_describe)

    #label_describe = explore_class(messages)
    #print(label_describe)

    #new_feature = messages_length(messages)
    #print(new_feature)

    #check_missing = missing(messages)
    #print(check_missing)

    #desc_length = length_describe(messages)
    #print(desc_length)

    #explore_longest_message = longest_message(messages)
    #print(explore_longest_message)

    #count_class(messages)

    #count_class_hist(messages)

    #class_pie = pie_class(messages)

    messages['message'] = messages['message'].apply(text_process)
    #print(messages.head())

    bag_of_words = vectorization(text_process).fit(messages['message'])
    bow = bag_of_words.transform(messages['message']).toarray()

    tfidf = tfidf_function().fit(bow)

    X = bow
    y = messages['class']

    X_train, X_test, y_train, y_test = split_data(X,y)

    NB(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)
    forest_tree(X_train, X_test, y_train, y_test)
    ada(X_train, X_test, y_train, y_test)
    support_vector_machine(X_train, X_test, y_train, y_test)
    logistic_regression(X_train, X_test, y_train, y_test)


