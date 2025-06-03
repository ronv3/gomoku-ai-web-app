"""
gomoku_ai.py

Gomoku AI using Iterative Deepening Minimax with Alpha–Beta Pruning.

Algorithm overview:
This AI agent uses a minimax search with alpha–beta pruning and iterative deepening so that it gradually
searches deeper within a fixed time limit (here set to 5 seconds). The evaluation function scores the board
by counting sequences of the AI’s stones (and subtracting the opponent’s score) based on their length and
whether the sequence is open on one or both sides. A sequence of exactly five stones (and only five, as
required by the rules) is considered a winning line and assigned a very high value.

Moves to be considered are generated from empty cells that are adjacent to existing stones, reducing the search
space. With iterative deepening, the algorithm starts at a shallow depth and increases the search depth until
the time limit is reached. This ensures that even when time is short, the AI returns a valid (if not optimal) move.
Alpha–beta pruning is used to cut off branches of the game tree that cannot affect the final decision.
"""

import time
import math
import random
import copy

# ------------------------------
# Helper Functions
# ------------------------------

def get_possible_moves(board):
    """
    Returns a list of candidate moves (row, col) from empty squares that are adjacent to any stone.
    If the board is empty, the center is returned.
    """
    board_size = len(board)
    moves = set()
    # if board is completely empty, return the center of the board
    if all(board[r][c] == 0 for r in range(board_size) for c in range(board_size)):
        return [(board_size // 2, board_size // 2)]
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] != 0:
                # examine neighbors (including diagonals)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == 0:
                            moves.add((nr, nc))
    return list(moves)


def check_win(board, player, move):
    """
    Checks if placing a stone for 'player' at the given move (row, col)
    results in exactly five in a row (horizontal, vertical, or diagonal).
    Note that if the sequence extends beyond five, it is not counted as a win.
    """
    board_size = len(board)
    r, c = move
    # only four principal directions need to be checked
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for dr, dc in directions:
        # Move back to the start of the potential sequence
        start_r, start_c = r, c
        while (0 <= start_r - dr < board_size and 0 <= start_c - dc < board_size and
               board[start_r - dr][start_c - dc] == player):
            start_r -= dr
            start_c -= dc
        count = 0
        nr, nc = start_r, start_c
        while 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == player:
            count += 1
            nr += dr
            nc += dc
        # Only return True if the count is exactly five and not extendable
        if count == 5:
            if (0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == player):
                continue  # sequence extends to 6 or more, so do not count as win
            return True
    return False


def is_terminal_node(board):
    """
    Checks whether the board is terminal (i.e. no moves are possible)
    or if one side has already won.
    This simplified test checks whether the board is completely full.
    """
    return not any(0 in row for row in board)


def evaluate(board, player):
    """
    Returns a heuristic evaluation value for the board from the perspective
    of 'player'. A positive score indicates an advantage.
    """
    # The final evaluation is the AI's score minus opponent's score.
    return score_for_player(board, player) - score_for_player(board, 3 - player)


def score_for_player(board, player):
    """
    Computes a score for the given player by scanning for consecutive stone sequences.
    Scores are based on the number of stones in sequence and whether the sequence is
    open on one or both ends.
    """
    board_size = len(board)
    score = 0
    # Four directions are considered for sequences
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == player:
                for dr, dc in directions:
                    # avoid recounting sequences—only count if this is the first stone in the line
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < board_size and 0 <= pc < board_size and board[pr][pc] == player:
                        continue
                    # count consecutive stones in direction (dr, dc)
                    length = 0
                    nr, nc = r, c
                    while 0 <= nr < board_size and 0 <= nc < board_size and board[nr][nc] == player:
                        length += 1
                        nr += dr
                        nc += dc
                    # check for open ends (if the cell before or after the sequence is empty or off-board)
                    open_ends = 0
                    pr, pc = r - dr, c - dc
                    if not (0 <= pr < board_size and 0 <= pc < board_size) or board[pr][pc] == 0:
                        open_ends += 1
                    if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr][nc] == 0:
                        open_ends += 1
                    # assign scores based on length and openness; winning sequence gets a very high score
                    if length >= 5:
                        score += 1000000
                    elif length == 4:
                        score += 10000 if open_ends == 2 else 1000
                    elif length == 3:
                        score += 1000 if open_ends == 2 else 100
                    elif length == 2:
                        score += 100 if open_ends == 2 else 10
                    elif length == 1:
                        score += 10
    return score


# ------------------------------
# Minimax Search With Alpha–Beta Pruning
# ------------------------------

def minimax(board, depth, alpha, beta, maximizingPlayer, player, start_time, time_limit):
    """
    Recursively computes a minimax value using alpha–beta pruning.
    'player' is the AI’s number and 'maximizingPlayer' is True when searching
    for the AI's best score. A TimeoutError is raised if the search exceeds the allowed time.
    """
    if time.time() - start_time > time_limit:
        raise TimeoutError
    if depth == 0 or is_terminal_node(board):
        return evaluate(board, player)
    moves = get_possible_moves(board)
    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            board[move[0]][move[1]] = player
            if check_win(board, player, move):
                board[move[0]][move[1]] = 0
                return 1000000
            eval_score = minimax(board, depth - 1, alpha, beta, False, player, start_time, time_limit)
            board[move[0]][move[1]] = 0
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # prune branch
        return max_eval
    else:
        opp = 3 - player
        min_eval = math.inf
        for move in moves:
            board[move[0]][move[1]] = opp
            if check_win(board, opp, move):
                board[move[0]][move[1]] = 0
                return -1000000
            eval_score = minimax(board, depth - 1, alpha, beta, True, player, start_time, time_limit)
            board[move[0]][move[1]] = 0
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # prune branch
        return min_eval


def iterative_deepening(board, player, depth, start_time, time_limit):
    """
    Performs one iteration of minimax at a given depth.
    Returns the best move found and its evaluation value.
    """
    best_move = None
    best_value = -math.inf
    moves = get_possible_moves(board)
    for move in moves:
        board[move[0]][move[1]] = player
        if check_win(board, player, move):
            board[move[0]][move[1]] = 0
            return move, 1000000
        try:
            value = minimax(board, depth - 1, -math.inf, math.inf, False, player, start_time, time_limit)
        except TimeoutError:
            board[move[0]][move[1]] = 0
            raise TimeoutError
        board[move[0]][move[1]] = 0
        if value > best_value:
            best_value = value
            best_move = move
    return best_move, best_value


# ------------------------------
# Main function: getTurn()
# ------------------------------

def getTurn(board, player):
    """
    Returns the next move for the given board and player as a tuple (row, col).
    The function uses iterative deepening minimax search with alpha–beta pruning.
    """
    start_time = time.time()
    time_limit = 10  # maximum search time in seconds (adjustable)
    best_move = None
    depth = 4
    try:
        while True:
            move, value = iterative_deepening(board, player, depth, start_time, time_limit)
            if move is not None:
                best_move = move
            depth += 1
    except TimeoutError:
        pass
    if best_move is None:
        # Fallback: choose a random move if none was found (should not normally happen)
        possible = get_possible_moves(board)
        best_move = random.choice(possible)
    return best_move
