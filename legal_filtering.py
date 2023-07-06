import chess
import numpy as np
from policy_indices import policy_index, move2ix
import torch

def filter_legal(board, model_output):
    sorted_moves = create_move_dict(model_output)

    for i, move in enumerate(list(sorted_moves.keys())):
        if chess.Move.from_uci(move) in board.legal_moves:
            return move
        # print(move, chess.Move.from_uci(move) in board.legal_moves)
    
    raise AttributeError("Network generated no legal moves.")


def create_move_dict(model_output):
    move_dict = dict()
    
    for i, move in enumerate(model_output.squeeze()):
        move_dict[policy_index[i]] = move.cpu().numpy()

    sorted_dict = dict(sorted(move_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

def get_legal_moves(board):
    legal_moves = torch.zeros(1858)

    for move in board.legal_moves:
        move_uci = move.uci()
        if len(move_uci) > 4 and move_uci[4] == 'n':
            move_uci = move_uci[:4]
        if board.turn == chess.WHITE:
            legal_moves[move2ix(move_uci, 'white')] = 1
        else:
            legal_moves[move2ix(move_uci, 'black')] = 1
    
    return legal_moves