import numpy as np

suits = ['clubs',
         'hearts',
         'diamonds',
         'spades']

names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


def find_sets(cards):
    set_sums = np.sum(cards, axis=1)
    set_mask = set_sums >= 3
    set_vals = values[set_mask]
    set_inds = np.arange(13)[set_mask]
    set_cards = cards * set_mask.reshape(-1, 1)
    set_lengths = set_sums[set_mask]
    return set_cards.astype(bool), set_lengths, set_vals, set_inds


def find_runs(cards):
    run_cards = np.zeros((13, 4))
    run_lengths = []
    run_vals = []
    run_inds = []
    for i in range(4):
        on_run = False
        run_length = 0
        for j in range(11):
            if not on_run:
                if np.all(cards[j: j + 3, i]):
                    run_cards[j: j + 3, i] = 1
                    on_run = True
                    run_length = 3
                    run_vals.append(values[j])
                    run_inds.append((j, i))
            else:
                if cards[j + 2, i]:
                    run_length += 1
                    run_cards[j + 2, i] = 1
                else:
                    run_lengths.append(run_length)
                    on_run=False
                    run_length = 0
        if run_length:
            run_lengths.append(run_length)


    return run_cards.astype(bool), run_lengths, run_vals, run_inds


def find_deadwoods(cards):
    used_cards = find_runs(cards)[0]|find_sets(cards)[0]

    return cards&(~used_cards)


def fix_hand(allocations):
    sets = find_sets(allocations == 2)[0]
    runs = find_runs(allocations == 3)[0]
    unused_cards = (allocations != 0)&(~(sets|runs))
    allocations[unused_cards] = 1

    return allocations


def evaluate_hand(allocations, n_cards=None):
    sets, set_lengths, set_vals, set_inds = find_sets(allocations == 2)
    runs, run_lengths, run_vals, run_inds = find_runs(allocations == 3)
    deadwoods = (allocations > 0)&(~(sets|runs))
    allocations[deadwoods] = 1
    deadwood_vals = np.array([values] * 4).T[allocations == 1]
    total_cards = np.sum(allocations != 0)
    if n_cards is None:
        n_cards = total_cards
    used_cards = sets|runs
    n_used_cards = np.sum(used_cards)
    # We have the exact right number of cards used
    if n_used_cards == n_cards:
        return 0
    # We have too few cards used
    elif n_used_cards < n_cards:
        return sum(sorted(list(deadwood_vals))[:n_cards - n_used_cards])
    # We need to take only a subset of the cards used
    if n_cards >= 3:
        set_leftovers = np.array(set_lengths) - 3
        run_leftovers = np.array(run_lengths) - 3
        all_leftovers = sorted(list(set_leftovers) + list(run_leftovers))
        pieces_available = min(int(n_cards / 3), len(all_leftovers)) # That is the total number of sets and runs
        leftovers_available = sum(all_leftovers[-pieces_available:])
        cards_needed = n_cards - 3 * pieces_available
        if leftovers_available >= cards_needed:
            return 0
        n_cards = cards_needed - leftovers_available
        assert n_cards < 3

    if len(run_vals):
        min_run = np.min(run_vals)
    else:
        min_run = 1000
    if len(set_vals):
        min_set = np.min(set_vals)
    else:
        min_set = 1000
    min_deadwoods = (sorted(list(deadwood_vals)) + [1000, 1000])[:2]
    if n_cards == 1:
        return min(min_run, min_set, min_deadwoods[0])
    elif n_cards == 2:
        return min(2 * min_run + 1,
                   2 * min_set,
                   min_run + min_deadwoods[0],
                   min_set + min_deadwoods[0],
                   min_deadwoods[0] + min_deadwoods[1])
    print(allocations)
    print(n_cards)
    assert False


def get_inds(num, n_digits):
    inds = np.zeros(n_digits).astype(bool)
    for i in range(n_digits):
        if num % 2:
            inds[i] = True
        num = int(num / 2)
    return inds


def solve_hand(cards, n_cards=None):
    if n_cards is None:
        n_cards = np.sum(cards)
    sets = find_sets(cards)[0]
    runs = find_runs(cards)[0]
    used_cards = sets|runs
    deadwoods = cards&(~used_cards)

    allocations = np.zeros((13, 4))

    # 0 for empty
    # 1 for deadwoods
    # 2 for sets
    # 3 for runs
    # 4 for uncertain
    allocations[deadwoods] = 1
    allocations[sets] = 2
    allocations[runs] = 3
    uncertain = sets&runs
    allocations[uncertain] = 4

    uncertain_inds = np.where(allocations.ravel() == 4)[0]
    if not len(uncertain_inds):
        return allocations, evaluate_hand(allocations, n_cards=n_cards)
    # Range through all permutations of each uncertain card being in a set or a run
    n_uncertain = len(uncertain_inds)
    best_allocation = None
    best_points = 100000
    for i in range(2 ** n_uncertain):
        set_inds = get_inds(i, n_uncertain)
        run_inds = ~set_inds
        allocation_i = allocations.ravel().copy()
        allocation_i[uncertain_inds[set_inds]] = 2
        allocation_i[uncertain_inds[run_inds]] = 3
        allocation_i = allocation_i.reshape((13, 4))
        points_i = evaluate_hand(allocation_i, n_cards=n_cards)
        if points_i == 0:
            best_allocation = allocation_i
            best_points = points_i
            break
        if points_i < best_points:
            best_points = points_i
            best_allocation = allocation_i
    return fix_hand(best_allocation), best_points


class Hand():
    def __init__(self, cards=None, deck=None):
        if cards is None:
            self.cards = np.zeros((13, 4), dtype=bool)
        else:
            self.cards = cards.astype(bool)

        self.deck = deck

    def deal(self, n_cards, remove=True):
        dealable_cards = np.where(self.cards.flatten())[0]
        dealt_cards = np.random.choice(dealable_cards, n_cards, False)
        deal = np.zeros(52)
        deal[dealt_cards] = 1
        deal = deal.reshape((13, 4)).astype(bool)
        if remove:
            self.cards[deal] = False
        dealt_hand = Hand(deal)

        return dealt_hand

    def add(self, new_cards, remove=False):
        if type(new_cards) is Hand:
            assert ~np.any(self.cards&new_cards.cards)
            self.cards = self.cards|new_cards.cards
            if remove:
                new_cards.cards[:] = False
        else:
            number, suit = new_cards
            assert ~self.cards[number, suit]
            self.cards[number, suit] = True

    def remove(self, new_cards, remove=False):
        if type(new_cards) is Hand:
            assert np.all(self.cards >= new_cards.cards)
            self.cards = self.cards^new_cards.cards
        else:
            number, suit = new_cards
            assert self.cards[number, suit]
            self.cards[number, suit] = False

    def check_integrity(self):
        if self.deck is not None:
            return self.deck.check_integrity()
        return True

    def print_hand(self):
        out_str = ''
        for i in range(13):
            for j in range(4):
                if self.cards[i, j]:
                    out_str += '%s%s ' % (names[i], suits[j][0])

        return out_str.strip()

    def show_cards(self):
        return self.cards.copy()

    def length(self):
        return np.sum(self.cards)

    def copy(self):
        return Hand(self.cards.copy())


class Deck():
    def __init__(self):
        self.base = Hand(np.ones((13, 4)))
        self.hands = [self.base]

    def deal(self, n_cards, recipient=None, track=True, remove=True):
        dealt_hand = self.base.deal(n_cards, remove=remove)
        if recipient is not None:
            recipient.add(dealt_hand)
        if track:
            if recipient is None:
                dealt_hand.deck = self
                self.hands.append(dealt_hand)
            else:
                if recipient not in self.hands:
                    recipient.deck = self
                    self.hands.append(recipient)

        return dealt_hand

    def check_integrity(self):
        counts = np.zeros((13, 4))
        for hand in self.hands:
            counts += hand.cards.astype(int)
        if ~np.all(counts == 1):
            raise IntegrityError()


def evaluate_hand_montecarlo(hand, deck, n_cards=None, cards_to_take=5, iterations=100):
    points = np.zeros(iterations)
    if n_cards is None:
        n_cards = hand.length()
    for i in range(iterations):
        test_hand = hand.copy()
        test_hand.add(deck.deal(cards_to_take, remove=False), remove=False)
        points[i] = solve_hand(test_hand.cards, n_cards)[1]

    return points


class DeckEstimate():
    def __init__(self, suspicion_exponent=2):
        # -1: definitely not occupied
        # 0: maybe occupied
        # 1: definitely occupied
        self.deck = np.zeros((13, 4))
        self.opponents_hand = np.zeros((13, 4))

        # Number of observations suggesting occupied -
        # number of observations suggesting unoccupied
        self.opponents_hand_suspicions = np.zeros((13, 4))
        self.suspicion_exponent = suspicion_exponent

    def print_estimate(self, log_fn=None):
        if log_fn is None:
            log_fn = print
        log_fn("Cards definitely in opponents hand: ")
        log_fn(Hand(self.opponents_hand == 1).print_hand())
        log_fn("Cards definitely not in opponents hand: ")
        log_fn(Hand(self.opponents_hand == -1).print_hand())
        log_fn("Cards suspected in opponents hand: ")
        log_fn(Hand(self.opponents_hand_suspicions > 0).print_hand())
        log_fn("Cards suspected not in opponents hand: ")
        log_fn(Hand(self.opponents_hand_suspicions < 0).print_hand())
        log_fn("Cards definitely in deck: ")
        log_fn(Hand(self.deck == 1).print_hand())
        log_fn("Cards definitely not in deck: ")
        log_fn(Hand(self.deck == -1).print_hand())

    def remove_from_deck(self, hand):
        self.deck[hand.cards] = -1
        self.clean_suspicions()

    def remove_from_opponents_hand(self, hand):
        self.opponents_hand[hand.cards] = -1
        self.clean_suspicions()

    def remove_from_both(self, hand):
        self.remove_from_deck(hand)
        self.remove_from_opponents_hand(hand)

    def add_to_deck(self, hand):
        self.deck[hand.cards] = 1
        self.remove_from_opponents_hand(hand)
        self.clean_suspicions()

    def add_to_opponents_hand(self, hand):
        self.opponents_hand[hand.cards] = 1
        self.remove_from_deck(hand)
        self.add_suspicion(hand)
        self.clean_suspicions()

    # Take into consideration fact that opponents
    # will often discard high point-value cards even though
    # they have adjacent cards?
    def add_suspicion(self, hand):
        cards = hand.cards
        assert np.sum(cards) == 1
        num_a, suit_a = np.where(cards)
        num = num_a[0]
        suit = suit_a[0]
        self.opponents_hand_suspicions[num, :] += 1
        if num:
            self.opponents_hand_suspicions[num - 1, suit] += 1
        if num < len(cards) - 1:
            self.opponents_hand_suspicions[num + 1, suit] += 1
        self.clean_suspicions()

    def remove_suspicion(self, hand):
        cards = hand.cards
        assert np.sum(cards) == 1
        num_a, suit_a = np.where(cards)
        num = num_a[0]
        suit = suit_a[0]
        self.opponents_hand_suspicions[num, :] -= 1
        if num:
            self.opponents_hand_suspicions[num - 1, suit] -= 1
        if num < len(cards) - 1:
            self.opponents_hand_suspicions[num + 1, suit] -= 1
        self.clean_suspicions()

    def clean_suspicions(self):
        self.opponents_hand_suspicions[self.opponents_hand != 0] = 0
        self.opponents_hand_suspicions[self.deck == 1] = 0

    def deck_sampler(self):
        hand = Hand(self.deck != -1)
        return hand

    def sample_opponents_hand(self, n_cards, cards_to_add=None):
        # Returns a sampling of where there may or may not be cards in the opponents hand
        hand_frequencies = (self.opponents_hand == 0) * np.power(self.suspicion_exponent, self.opponents_hand_suspicions)
        if cards_to_add is not None:
            hand_frequencies[cards_to_add.cards] = 0
        hand_frequencies = hand_frequencies.flatten() / np.sum(hand_frequencies)
        sample_inds = np.random.choice(np.arange(52), n_cards, replace=False, p=hand_frequencies)
        hand_sample = np.zeros(52, dtype=bool)
        hand_sample[sample_inds] = True

        return Hand(hand_sample.reshape(13, 4))

    def evaluate_opponents_hand_montecarlo(self, n_cards, cards_to_add=None, cards_to_take=5, iterations=100):
        hand = Hand(self.opponents_hand == 1)
        if cards_to_add is not None:
            hand.add(cards_to_add)
        unknown_cards_in_hand = n_cards - hand.length()
        points = np.zeros(iterations)
        for i in range(iterations):
            test_hand = hand.copy()
            test_deck = Hand(self.deck >= 0)
            hand_sample = self.sample_opponents_hand(unknown_cards_in_hand, cards_to_add)
            test_deck.remove(hand_sample)
            deck_sample = test_deck.deal(cards_to_take)
            test_hand.add(hand_sample)
            test_hand.add(deck_sample)
            points[i] = solve_hand(test_hand.cards, n_cards)[1]
        return points

    def check_integrity(self):
        assert ~np.any((self.deck == 1)&(self.opponents_hand == 1))
