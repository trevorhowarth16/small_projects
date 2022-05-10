from backend import Deck, Hand, solve_hand


class Gin():
    def __init__(self, agent1, agent2, cards_in_hand=10):
        self.cards_in_hand = cards_in_hand
        self.agent1 = agent1
        self.agent2 = agent2

        self.deck = Deck()

        self.hand1 = Hand()
        self.hand2 = Hand()
        self.upcard = Hand()
        self.discard_pile = Hand()

        self.agent1.pass_hand(self.hand1.show_cards)
        self.agent1.pass_upcard(self.upcard.show_cards)
        self.agent1.pass_discard_pile(self.discard_pile.show_cards)

        self.agent2.pass_hand(self.hand2.show_cards)
        self.agent2.pass_upcard(self.upcard.show_cards)
        self.agent2.pass_discard_pile(self.discard_pile.show_cards)

        self.turn = 0

    def print_game_state(self, show_deck=False):
        hands = [self.hand1, self.hand2, self.upcard, self.discard_pile]
        names = ['hand1', 'hand2', 'upcard', 'discard_pile']
        print("Turn: %d" % self.turn)
        for hand, name in zip(hands, names):
            print(name)
            print(hand.print_hand())
        if show_deck:
            print('deck')
            print(self.deck.base.print_hand())

    def check_win(self):
        if self.turn == 0:
            down_hand = self.hand1
            down_name = self.agent1.name
            opponent_hand = self.hand2
            opponent_name = self.agent2.name
        else:
            down_hand = self.hand2
            down_name = self.agent2.name
            opponent_hand = self.hand1
            opponent_name = self.agent1.name

        _, down_points = solve_hand(down_hand.cards)
        _, opponent_points = solve_hand(opponent_hand.cards)
        print("%s's hand:" % down_name)
        print(down_hand.print_hand())
        print("%s's hand:" % opponent_name)
        print(opponent_hand.print_hand())
        if down_points > 10:
            print("%s has more than 10 points (%d) and is not allowed to go down." % (down_name, down_points))
            return False
        elif down_points:
            if opponent_points >= down_points:
                print("%s knocks with %d points and %s has %d points, so %s nets %d points" % (
                    down_name, down_points, opponent_name, opponent_points, down_name, opponent_points - down_points))
        else:
            print("%s has gin and %s has %d points, so %s nets %d points" % (
                    down_name, opponent_name, opponent_points, down_name, opponent_points + 25))
        return True

    def play(self):
        # Setup, agent1 always goes first
        self.deck.deal(self.cards_in_hand, recipient=self.hand1)
        self.deck.deal(self.cards_in_hand, recipient=self.hand2)
        self.deck.deal(1, recipient=self.upcard)
        self.deck.hands.append(self.discard_pile)

        self.deck.check_integrity()
        self.turn = 0
        playing = True
        while playing:
            for agent, other_agent, hand in zip([self.agent1, self.agent2], [self.agent2, self.agent1], [self.hand1, self.hand2]):
                accepted_up_card = agent.check_upcard()
                if accepted_up_card:
                    other_agent.view_opponent_draw(True)
                    hand.add(self.upcard, remove=True)
                else:
                    other_agent.view_opponent_draw(False)
                    self.discard_pile.add(self.upcard, remove=True)
                    drawn_card = self.deck.deal(1, recipient=hand)
                    agent.draw_from_deck(drawn_card.show_cards)

                self.upcard.add(agent.discard())
                other_agent.view_opponent_discard()
                hand.remove(self.upcard)

                go_down = agent.query_go_down()
                if go_down:
                    valid_win = self.check_win()
                    if valid_win:
                        playing = False
                        break

                self.turn = (self.turn + 1) % 2
                self.deck.check_integrity()
                if self.deck.base.length() == 0:
                    # Reshuffle deck
                    pass