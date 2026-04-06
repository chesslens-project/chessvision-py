import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish")
board = chess.Board()
result = engine.analyse(board, chess.engine.Limit(depth=15))
print("Stockfish is working. Score:", result["score"])
engine.quit()
