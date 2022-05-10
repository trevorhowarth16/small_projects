import numpy as np
from backend import DeckEstimate, evaluate_hand_montecarlo, Hand, solve_hand


class Agent():
    def __init__(self):
        pass

    def pass_hand(self, see_hand):
        self.see_hand = see_hand

    def pass_upcard(self, see_upcard):
        self.see_upcard = see_upcard

    def pass_discard_pile(self, see_discard_pile):
        self.see_discard_pile = see_discard_pile

    def log(self, message):
        print('[%s] %s' % (self.name, message))


class HumanAgent(Agent):
    def __init__(self, name):
        self.name = name

    def print_discard_pile(self):
        pile = Hand(self.see_discard_pile())
        self.log("Discard pile:")
        self.log(pile.print_hand())

    def check_upcard(self):
        hand = Hand(self.see_hand())
        upcard = Hand(self.see_upcard())
        self.log("Your hand")
        self.log(hand.print_hand())
        self.log("Upcard")
        self.log(upcard.print_hand())
        while True:
            accept = input("Would you like to take the upcard (y/n or d to see discard pile) ") # Give option to see discard_pile
            if accept.lower() == 'y':
                return True
            if accept.lower() == 'n':
                return False
            if accept.lower() == 'd':
                self.print_discard_pile()
                continue
            self.log("Invalid input")

    def draw_from_deck(self, draw_card):
        card = Hand(draw_card())
        self.log("You draw %s from the deck." % card.print_hand())

    def discard(self):
        hand = Hand(self.see_hand())
        self.log("Choose one of the following to discard: ")
        discard_str = ''
        for i, card in enumerate(hand.print_hand().split(' ')):
            self.log(str(i) + ': ' + card)
        while True:
            discard = input("Enter a card index to discard (0 - %d or d to see discard pile): " % (hand.length() - 1))
            if discard.lower() == 'd':
                self.print_discard_pile()
                continue
            try:
                discard_ind = int(discard)
            except ValueError:
                self.log("Invalid input")
                continue

            card_idxs = np.where(hand.cards.flatten())[0]
            try:
                discard_idx = card_idxs[discard_ind]
            except IndexError:
                self.log("Invalid input")
                continue

            discard_hand = np.zeros(52)
            discard_hand[discard_idx] = 1
            discard_hand = Hand(discard_hand.reshape(13, 4))
            self.log("You discard %s" % discard_hand.print_hand())
            return discard_hand


    def query_go_down(self):
        cards = self.see_hand()
        _, points = solve_hand(cards)
        if points:
            if points <= 10:
                while True:
                    go_down_str = input("You have %d points, would you like to knock? (y/n) " % points)
                    if go_down_str.lower() == 'y':
                        return True
                    if go_down_str.lower() == 'n':
                        return False
                    self.log("Invalid input")
            else:
                return False
        else:
            while True:
                go_down_str = input("You have gin, would you like to go down? (y/n) ")
                if go_down_str.lower() == 'y':
                    return True
                if go_down_str.lower() == 'n':
                    return False
                self.log("Invalid input")

    def view_opponent_draw(self, took_upcard):
        if took_upcard:
            upcard = Hand(self.see_upcard())
            self.log("Your opponent takes the upcard: %s" % upcard.print_hand())
        else:
            self.log("Your opponent draws a card from the deck")

    def view_opponent_discard(self):
        upcard = Hand(self.see_upcard())
        self.log("Your opponent discards: %s" % upcard.print_hand())


class AIAgent(Agent):
    def __init__(self, name, mc_extra_cards=5, mc_iterations=100, debug=False):
        self.name = name
        self.mc_extra_cards = mc_extra_cards
        self.mc_iterations = mc_iterations
        self.deck_opponent_estimate = DeckEstimate()

        self.debug = debug

        self.deck_prepped = False

    def check_upcard(self):
        hand = Hand(self.see_hand())
        upcard = Hand(self.see_upcard())

        # Remove hand from deck if first turn
        if not self.deck_prepped:
            self.deck_opponent_estimate.remove_from_both(hand)
            self.deck_prepped = True

        # Remove upcard from deck
        self.deck_opponent_estimate.remove_from_both(upcard)

        hand_w_upcard = hand.copy()
        hand_w_upcard.add(upcard)
        hand_pts = evaluate_hand_montecarlo(
            hand,
            self.deck_opponent_estimate.deck_sampler(),
            n_cards=hand.length(),
            cards_to_take=self.mc_extra_cards,
            iterations=self.mc_iterations)

        hand_w_upcard_pts = evaluate_hand_montecarlo(
            hand_w_upcard,
            self.deck_opponent_estimate.deck_sampler(),
            n_cards=hand.length(),
            cards_to_take=self.mc_extra_cards - 1,
            iterations=self.mc_iterations)

        if self.debug:
            self.log("Hand: %s" % hand.print_hand())
            self.log("Hand points: %f" % hand_pts.mean())
            self.log("Upcard: %s" % upcard.print_hand())
            self.log("Hand w/ upcard points: %f" % hand_w_upcard_pts.mean())

        if hand_pts.mean() > hand_w_upcard_pts.mean():
            return True  # Take upcard
        return False  # Don't take upcard

    def draw_from_deck(self, draw_card):
        self.deck_opponent_estimate.remove_from_both(Hand(draw_card()))
        if self.debug:
            print("Drawn card: %s" % Hand(draw_card()).print_hand())

    def discard(self):
        hand = Hand(self.see_hand())
        card_idxs = np.where(hand.cards.flatten())[0]
        # Want to maximize this
        discard_points = np.zeros(hand.length())
        default_opponent_points = self.deck_opponent_estimate.evaluate_opponents_hand_montecarlo(hand.length()).mean()
        for i, card_idx in enumerate(card_idxs):
            discard_card = np.zeros(52)
            discard_card[card_idx] = 1
            discard_hand = Hand(discard_card.reshape(13, 4))
            remaining_hand = hand.copy()
            remaining_hand.remove(discard_hand)
            points = evaluate_hand_montecarlo(
                remaining_hand,
                self.deck_opponent_estimate.deck_sampler(),
                n_cards=remaining_hand.length(),
                cards_to_take=self.mc_extra_cards,
                iterations=self.mc_iterations).mean()
            opponent_points_w_discard = self.deck_opponent_estimate.evaluate_opponents_hand_montecarlo(
                hand.length(), cards_to_add=discard_hand).mean()
            opponent_points = min(default_opponent_points, opponent_points_w_discard)
            discard_points[i] = opponent_points - points
            if self.debug > 1:
                self.log("Remaining Hand %d: %s" % (i, remaining_hand.print_hand()))
                self.log("Discard: %s" % discard_hand.print_hand())
                self.log("Points: %f\n" % points)
                self.log("Estimated opponent_points: %f\n" % opponent_points)

        best_discard_idx = card_idxs[np.argmax(discard_points)]
        best_discard_card = np.zeros(52)
        best_discard_card[best_discard_idx] = 1
        best_discard_hand = Hand(best_discard_card.reshape(13, 4))
        if self.debug:
            self.log("Chosen Discard: %s" % best_discard_hand.print_hand())
            self.deck_opponent_estimate.print_estimate(self.log)
        return best_discard_hand

    def query_go_down(self):
        cards = self.see_hand()
        _, points = solve_hand(cards)

        if points:
            if points <= 10:
                return True
            return False
        else:
            return True

    def view_opponent_draw(self, took_upcard):
        hand = Hand(self.see_upcard())
        self.deck_opponent_estimate.remove_from_deck(hand)
        if took_upcard:
            self.deck_opponent_estimate.add_to_opponents_hand(hand)
            self.deck_opponent_estimate.add_suspicion(hand)
        else:
            self.deck_opponent_estimate.remove_from_opponents_hand(hand)
            self.deck_opponent_estimate.remove_suspicion(hand)

    def view_opponent_discard(self):
        hand = Hand(self.see_upcard())
        self.deck_opponent_estimate.remove_suspicion(hand)