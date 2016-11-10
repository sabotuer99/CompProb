import sys
import os.path
import numpy as np
from math import log, exp
from collections import Counter, defaultdict

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    word_counts = Counter()
    total = 0    
    for filename in file_list:
        words = util.get_words_in_file(filename)
        for word in words:
            count = word_counts[word]
            word_counts[word] = count + 1.0
            total += 1.0
    
    return word_counts

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    word_counts = get_counts(file_list)
    
    total_words = 0;
    for word, count in word_counts.items():
        total_words += count

    total_words += len(word_counts) #smoothing
    log_probs = defaultdict(lambda : log(1.0/total_words))
    for word, count in word_counts.items():
        log_probs[word] = log(word_counts[word]/total_words)
    
    return log_probs


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    spam = file_lists_by_category[0]
    ham = file_lists_by_category[1]
    spam_prob = get_log_probabilities(spam)
    ham_prob = get_log_probabilities(ham)
    
    log_prior_spam = log(len(spam) / (len(spam) + len(ham)))
    log_prior_ham = log(len(ham) / (len(spam) + len(ham)))
    
    return [spam_prob, ham_prob], [log_prior_spam, log_prior_ham]
    

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    email_words = util.get_words_in_file(email_filename)  
    test_words = set([])
    
    for list in log_probabilities_by_category:
      for word in list:
        test_words.add(word)
    
    spam = log_prior_by_category[0]
    ham = log_prior_by_category[1]
    
    print("spam prior:" + str(spam))
    print("ham prior:" + str(ham) + "\n")    
    
    spam_data = log_probabilities_by_category[0]    
    ham_data = log_probabilities_by_category[1] 
    
    for word in test_words:
      if word in email_words:
        spam += spam_data[word]
        ham += ham_data[word]
      else:
        spam += log(1 - exp(spam_data[word]))
        ham += log(1 - exp(ham_data[word]))
    
    
    print("spam:" + str(spam))
    print("ham:" + str(ham) + "\n\n\n")
    
    
    if spam > ham:
        label = "spam"
    else:
        label = "ham"
        
    return label

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
