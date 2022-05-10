from gin import Gin
from agents import AIAgent, HumanAgent

game = Gin(HumanAgent("Player 1"), AIAgent("Player 2", debug=False))
game.play()