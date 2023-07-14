"""
Some example strategies for people who want to create a custom, homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult
import random
from engine_wrapper import MinimalEngine
from typing import Any
import pickle
import io
from policy_indices import policy_index
from legal_filtering import get_legal_moves
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class DLVCInterface(MinimalEngine):
    def __init__(self, commands, options, stderr, draw_or_resign, cwd=None):
        super(DLVCInterface, self).__init__(commands, options, stderr, draw_or_resign, cwd=cwd)
        self.model = 'multi-task-move'
        self.move_history = 8

    def board2tensor(self, board):
        if self.model == 'multi-task-move':
            n_channels = 33
        else:
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

        counter = 17
        move_history = board.move_stack
        for move in reversed(move_history):
            board_tensor[counter, chess.square_rank(move.from_square), chess.square_file(move.from_square)] = 1
            board_tensor[counter+1, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1
            counter += 2
            if counter >= 33:
                break

        # # move encoding
        # # 18th channel: from square
        # board_tensor[17, chess.square_rank(move.from_square), chess.square_file(move.from_square)] = 1
        # # 19th channel: to square
        # board_tensor[18, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1
        
        if board.turn == chess.BLACK:
            board_tensor = np.flip(np.flip(board_tensor, axis=2), axis=1).copy()

        return torch.tensor(board_tensor, device=DEVICE)

    def flip_move(self, move):
        return move.translate(str.maketrans("abcdefgh12345678", "hgfedcba87654321"))

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        MODEL_PATH = 'models/multi-task-move-07-06-23_22-15-52_0.95858.pkl'
        model = CPU_Unpickler(open(MODEL_PATH, 'rb')).load()
        model.to(DEVICE)
        model.eval()
        #print([module for module in model.modules() if not isinstance(module, nn.Sequential)])

        with torch.no_grad():  
            model_output = model(torch.tensor(np.expand_dims(self.board2tensor(board).cpu(), 0), device=DEVICE))
            if self.model == 'multi-task-move':
                model_output = model_output[0]
        
        model_output = torch.nn.functional.softmax(model_output, dim=-1)[0]

        prediction_full = get_legal_moves(board).to(DEVICE) * model_output
        prediction_full = prediction_full / prediction_full.sum()

        sampled_index = np.random.choice(np.arange(len(prediction_full)), p=prediction_full.numpy())

        prediction = policy_index[sampled_index]

        if board.turn == chess.BLACK:
            prediction = self.flip_move(prediction)

        prediction = prediction[:4]
        
        #if board.turn == chess.WHITE:
        #    prediction = filter_legal(board, model_output)
        #else:
        #    prediction = self.flip_move(filter_legal(board.mirror(), model_output))

        return chess.engine.PlayResult(move=chess.Move.from_uci(prediction), ponder = None)
        
        # return PlayResult(random.choice(list(board.legal_moves)), None)




class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass

# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)
