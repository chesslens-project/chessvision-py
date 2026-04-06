import chess
import chess.engine

def test_stockfish_connection():
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    board = chess.Board()
    result = engine.analyse(board, chess.engine.Limit(depth=10))
    assert result["score"] is not None
    engine.quit()
