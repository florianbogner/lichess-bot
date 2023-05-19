import pickle
import torch
import chess
import chess.engine
import numpy as np

from legal_filtering import filter_legal

def board2tensor(board):
    n_channels = 17
    board_tensor = np.zeros((n_channels, 8, 8), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        offset = 0 if color == chess.WHITE else 6
        # first/seventh channel: white king
        board_tensor[offset+0, :, :] = np.reshape(board.pieces(chess.KING, color).tolist(), (8, 8))
        # second/eighth channel: white queen
        board_tensor[offset+1, :, :] = np.reshape(board.pieces(chess.QUEEN, color).tolist(), (8, 8))
        # third/nineth channel: white rook
        board_tensor[offset+2, :, :] = np.reshape(board.pieces(chess.ROOK, color).tolist(), (8, 8))
        # fourth/tenth channel: white bishop
        board_tensor[offset+3, :, :] = np.reshape(board.pieces(chess.BISHOP, color).tolist(), (8, 8))
        # fifth/eleventh channel: white knight
        board_tensor[offset+4, :, :] = np.reshape(board.pieces(chess.KNIGHT, color).tolist(), (8, 8))
        # sixth/twelveth channel: white pawn
        board_tensor[offset+5, :, :] = np.reshape(board.pieces(chess.PAWN, color).tolist(), (8, 8))

    # thirteenth channel: castling right white
    board_tensor[12, :, :] = (board.castling_rights & chess.BB_H1 != 0)
    # fourteenth channel: castling left white
    board_tensor[13, :, :] = (board.castling_rights & chess.BB_A1 != 0)
    # fifteenth channel: castling right black
    board_tensor[14, :, :] = (board.castling_rights & chess.BB_H8 != 0)
    # sixteenth channel: castling left black
    board_tensor[15, :, :] = (board.castling_rights & chess.BB_A8 != 0)

    # seventeenth channel: whites turn
    board_tensor[16, :, :] = (board.turn == chess.WHITE)

    # # move encoding
    # # 18th channel: from square
    # board_tensor[17, chess.square_rank(move.from_square), chess.square_file(move.from_square)] = 1
    # # 19th channel: to square
    # board_tensor[18, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1
    
    if board.turn == chess.BLACK:
        board_tensor = np.flip(np.flip(board_tensor, axis=2), axis=1).copy()

    return torch.tensor(board_tensor, device="cuda")

def flip_move(move):
    return move.translate(str.maketrans("abcdefgh12345678", "hgfedcba87654321"))

board = chess.Board()

MODEL_PATH = './../chess-teacher/output/maia-move-05-19-23_3.43391.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))
model.eval()
with torch.no_grad():  
    model_output = model(torch.tensor(np.expand_dims(board2tensor(board).cpu(), 0 ), device="cuda"))

if board.turn == chess.WHITE:
    prediction = filter_legal(board, model_output)
else:
    prediction = flip_move(filter_legal(board.mirror(), model_output))

print(chess.engine.PlayResult(move=chess.Move.from_uci(prediction), ponder = None))
