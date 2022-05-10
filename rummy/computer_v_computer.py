from gin import Gin
from agents import AIAgent

game = Gin(AIAgent("Player 1", debug=True), AIAgent("Player 2", debug=True))
game.play()