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
from policy_indices import policy_index
from legal_filtering import get_legal_moves
import numpy as np
import torch
import re


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

class DLVCInterface(MinimalEngine):
    def board2tensor(self, board):
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

    def flip_move(self, move):
        return move.translate(str.maketrans("abcdefgh12345678", "hgfedcba87654321"))

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        MODEL_PATH = './../chess-teacher/output/maia-move-05-19-23_2.51139.pkl'
        model = pickle.load(open(MODEL_PATH, 'rb'))
        model.eval()
        with torch.no_grad():  
            model_output = model(torch.tensor(np.expand_dims(self.board2tensor(board).cpu(), 0 ), device="cuda"))

        prediction_full = get_legal_moves(board).to('cuda') * model_output
        prediction = policy_index[torch.argmax(prediction_full)]

        if board.turn == chess.BLACK:
            prediction = self.flip_move(prediction)
        
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
    
class LLM(ExampleEngine):
    def generate_san(self, move_stack):
        play_count = 1
        san_string = ""
        board = chess.Board()
        for move in move_stack:
            if board.turn == chess.WHITE:
                san_string += str(play_count) + "."
                play_count += 1
            san_string += board.san(move) + " "
            board.push(move)
        return san_string

    def get_move_from_llm(self, output_str, board):
        # print("Output str: ", output_str)

        splitted = str(output_str).replace(".", " ").replace(":", " ").split(" ")
        cleaned_moves = [x for x in splitted if has_numbers(x) and len(x) > 1 and len(x) <= 5 and not x.isnumeric()]

        # print("Cleaned moves: ", cleaned_moves)
        for move in cleaned_moves:
            try:
                move = board.parse_san(move)
                # print("trying move", move)
                if move in board.legal_moves:
                    return move
            except:
                pass
        return None

    def search(self, board: chess.Board, *args: Any) -> PlayResult:

        san = self.generate_san(board.move_stack)
        print(san)

        turn = "white" if board.turn == chess.WHITE else "black"
        prompt = f'In the following chess game, you play {turn}: [Event "Rated Classical game https://lichess.org/tounament/xxxx"] [Date „??“] [Round "-"] [White "???"] [Black "???"] [Result "1-0"] [WhiteElo "2800"] [BlackElo „2800"] [WhiteRatingDiff "??"] [BlackRatingDiff "??"] [ECO "??"] [Opening „??"] [TimeControl "300+0"] [Termination "Time forfeit"] {san}  Choose your best next move.'
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True,
        )
        token = outputs.sequences[0, input_length:]
        output_str = self.tokenizer.decode(token)

        move = self.get_move_from_llm(output_str, board)
        print("Move from LLM: ", move)
        if move != None:
            return PlayResult(move, None)
        else:   
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
