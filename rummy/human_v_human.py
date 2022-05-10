from gin import Gin
from agents import HumanAgent

game = Gin(HumanAgent("Player 1"), HumanAgent("Player 2"))
game.play()