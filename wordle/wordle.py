import numpy as np
import json
from tqdm import tqdm
import argparse

guess_lex_file = 'wordle_valid_guesses.json'
with open(guess_lex_file, 'r') as f:
    guess_list = json.load(f)
GUESS_LEXICON = np.array(guess_list)

answer_lex_file = 'wordle_valid_answers.json'
with open(answer_lex_file, 'r') as f:
    answer_list = json.load(f)
LEXICON = np.array(answer_list)

# Generate the word features (takes a few seconds)
# LEX_LETTER_FEATURES = np.array([get_letter_features(w) for w in LEXICON])
# LEX_CONTAINS_FEATURES = np.array([get_contains_features(w) for w in LEXICON])

# Load precalculated word features
LEX_LETTER_FEATURES = np.load('letter_features.npy')
LEX_CONTAINS_FEATURES = np.load('contains_features.npy')
LEX_FEATURES = (LEX_LETTER_FEATURES, LEX_CONTAINS_FEATURES)

# Represents the relative probabilities that any word in the lexicon would be the answer
# to a given puzzle, currently set to all equal although we should really discount 
# less common words like 'xylem', etc.
SAMPLING_RATES = np.ones(len(LEXICON))

FIRST_GUESS = 'roate'
MIN_GUESS_CUTOFF = 0.49


def char_to_int(c):
    return ord(c) - 97


def get_letter_features(word):
    features = np.zeros(len(word), dtype=np.uint8) # We'd have to use a different data type if word had too many letters
    for i, c in enumerate(word):  
        features[i] = char_to_int(c) + 26 * i
    return features


def get_contains_features(word):
    features = np.zeros(26, dtype=np.uint8)
    for c in word:
        features[char_to_int(c)] += 1
    return features


def find_matches(word_features, lex_features=LEX_FEATURES):
    lex_letter_features, lex_contains_features = lex_features
    word_letters, word_contains, word_no_contains = word_features
    valid_inds = np.arange(len(lex_letter_features))
    
    word_letters = word_letters.reshape(1, -1, 26) # [i, j] true is letter j could be in position i
    word_min_contains = word_contains.reshape(1, -1)
    word_max_contains = word_no_contains.reshape(1, -1)
    # Test for contains second, this is slower
    if np.any(word_min_contains) or np.any(word_max_contains < 5):
        contains_matches = np.all(
            (word_min_contains <= lex_contains_features)&(word_max_contains >= lex_contains_features), axis=1
        )
        valid_inds  = valid_inds[contains_matches]
        lex_letter_features = lex_letter_features[contains_matches]
        lex_contains_features = lex_contains_features[contains_matches]
    # Next test for letter placement, this is the most restrictive
    if ~np.all(word_letters):
        letter_matches = (word_letters.flatten()[lex_letter_features.flatten()]).reshape(-1, 5)
        letter_matches = np.all(letter_matches, axis=1)
        valid_inds  = valid_inds[letter_matches]
        lex_letter_features = lex_letter_features[letter_matches]
        lex_contains_features = lex_contains_features[letter_matches]
    mask = np.zeros(len(lex_features[0]), dtype=bool)
    mask[valid_inds] = True
    return mask


def starting_guess_features():
    l_f = np.zeros((5, 26), dtype=bool) + 1
    c_min_f = np.zeros(26, dtype=np.uint8)
    c_max_f = np.zeros(26, dtype=np.uint8) + 5
    
    return l_f, c_min_f, c_max_f


def evaluate_words(word_list, lexicon=LEXICON, lex_features=LEX_FEATURES, samples=None, sampling_rates=None, debug=False):
    mean_matches = []
    std_matches = []
    if debug:
        it = tqdm(word_list)
    else:
        it = word_list
    for guess_0 in it:
        if samples is None:
            word_samples = lexicon
            n_matches = np.zeros(len(lexicon))
        elif samples < len(lexicon):
            if sampling_rates is None:
                sampling_rates = np.ones(len(lexicon), dtype=float) / len(lexicon)
            word_samples = np.random.choice(lexicon, samples, replace=False, p=sampling_rates)
            n_matches = np.zeros(samples)
        else:
            word_samples = lexicon
            n_matches = np.zeros(len(lexicon))
        for i, true_word in enumerate(word_samples):
            # Do we care more about eliminating likely words?
#             n_matches[i] = np.dot(
#                 SAMPLING_RATES, 
#                 find_matches(update_features(guess_0, true_word), lex_features=lex_features))
#             )
            n_matches[i] = np.sum(
                find_matches(update_features(guess_0, true_word), lex_features=lex_features)
            )
        mean_matches.append(np.mean(n_matches))
        std_matches.append(np.std(n_matches))
    return mean_matches, std_matches


def update_features(guess, true_word, current_features=None):
    if current_features is None:
        current_features = starting_guess_features()
    l_f = current_features[0].copy()
    c_min_f = current_features[1].copy()
    c_max_f = current_features[2].copy()
    if true_word is None:
        results = input("Enter guess results here: 0 for no match (gray), "
                        "1 for letter match (yellow), 2 for letter and place match (green): ")
        n_letters_contained = np.zeros(26, dtype=np.uint8)
        letters_maxed = np.zeros(26, dtype=bool)
        for i, (c, r) in enumerate(zip(guess, results)):
            c_i = char_to_int(c)
            r_ind = int(r)
            if r_ind == 0:
                l_f[i, c_i] = False  # Letter is not in that position in the word.
                # Word contains exactly the observed count of this letter
                letters_maxed[c_i] = True
            elif r_ind == 1:
                l_f[i, c_i] = False # Letter is not in that position in the word.
                n_letters_contained[c_i] += 1
            else:
                l_f[i] = False
                l_f[i, c_i] = True # Letter is in that position in the word.
                n_letters_contained[c_i] += 1
        # There are at least this many instances of the letter in the word
        c_min_f = np.maximum(n_letters_contained, c_min_f) 
        c_max_f[letters_maxed] = n_letters_contained[letters_maxed]

    else:
        n_letters_contained = np.zeros(26, dtype=np.uint8)
        n_letters_true = np.zeros(26, dtype=np.uint8)
        letters_tested = np.zeros(26, dtype=bool)
        for i, (c, t_c) in enumerate(zip(guess, true_word)):
            c_i = char_to_int(c)
            t_c_i = char_to_int(t_c)
            n_letters_true[t_c_i] += 1
            n_letters_contained[c_i] += 1
            letters_tested[c_i] = True
            if c == t_c:
                l_f[i] = False
                l_f[i, c_i] = True
            else:
                l_f[i, c_i] = False
        n_yellows = np.minimum(n_letters_contained, n_letters_true)
        c_min_f[letters_tested] = np.maximum(c_min_f[letters_tested], n_yellows[letters_tested])
        maxed_inds = letters_tested.copy()
        maxed_inds[n_letters_contained <= n_letters_true] = False
        c_max_f[maxed_inds] = n_letters_true[maxed_inds]
    
    return l_f, c_min_f, c_max_f  


def play_game(true_word, first_guess=FIRST_GUESS, samples=20, max_turns=6, debug=True):
    if true_word is not None:
        assert true_word in LEXICON
    turn_ind = 1
    match_mask = np.ones_like(LEXICON)
    possible_words = LEXICON
    match_features = LEX_FEATURES
    features = None
    if true_word is None:
        guess = input("Enter guess used (recommendation is %s): " % first_guess)
    else:
        guess = first_guess
    while True:
        if debug:
            print("Turn %d guess: %s" % (turn_ind, guess))
        if guess == true_word:
            if debug:
                print("Turn: %d, guess: %s is correct!" % (turn_ind, guess))
            return guess, turn_ind
        # Test current_guess
        features = update_features(guess, true_word, features)
        
        match_mask = find_matches(features)
        if debug:
            print("Number of possible words remaining: %d" % match_mask.sum())

        # Come up with next guess
        possible_words = LEXICON[match_mask]
        if debug:
            print(possible_words)
        if true_word is None:
            if len(possible_words) == 0:
                print("True word not in lexicon")
                return None, None
            if len(possible_words) == 1:
                print("Only one possibility remains: %s" % (possible_words[0]))
                return possible_words[0], turn_ind + 1
            elif len(possible_words) == 2:
                print("Only two possibilities remain: %s" % possible_words)
            else:
                match_features = (LEX_LETTER_FEATURES[match_mask], LEX_CONTAINS_FEATURES[match_mask])
                sampling_rates = SAMPLING_RATES[match_mask]
                sampling_rates /= np.sum(sampling_rates)
                match_means, _ = evaluate_words(
                    GUESS_LEXICON, lexicon=possible_words, 
                    lex_features=match_features, samples=samples,
                    sampling_rates=sampling_rates,
                    debug=debug)
                print("Top 20 guesses: ")
                print(GUESS_LEXICON[np.argsort(match_means)[:20]])
            guess = input("Enter guess used: ")
        else:

            possibility_freqs = SAMPLING_RATES[match_mask]
            possibility_freqs /= np.sum(possibility_freqs)
            if possibility_freqs.max() > MIN_GUESS_CUTOFF:
                if debug:
                    print("Guessing answer with p=%f" % possibility_freqs.max())
                guess = possible_words[np.argmax(possibility_freqs)]
            else:
                match_features = (LEX_LETTER_FEATURES[match_mask], LEX_CONTAINS_FEATURES[match_mask])
                sampling_rates = SAMPLING_RATES[match_mask]
                sampling_rates /= np.sum(sampling_rates)
                match_means, _ = evaluate_words(
                    GUESS_LEXICON, lexicon=possible_words, 
                    lex_features=match_features, samples=samples,
                    sampling_rates=sampling_rates,
                    debug=debug)
                guess = GUESS_LEXICON[np.argmin(match_means)]
            
        turn_ind += 1
        if turn_ind > max_turns:
            return None, -1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='', help="Solution to the wordle, leave empty to play interactively.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--first_guess', type=str, default=FIRST_GUESS)
    
    args = parser.parse_args()
    print(args)
    if args.word:
        word = args.word
    else:
        word = None
    play_game(word, first_guess=args.first_guess, samples=args.max_samples, max_turns=6, debug=args.debug)
    