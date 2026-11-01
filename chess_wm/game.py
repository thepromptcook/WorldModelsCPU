"""Chess as a grid of tokens for the world model: 9x8 = the 64 squares + one status row.

Board rows 0-7 are ranks 8..1 (as you'd look at the board), columns are files a..h.
Status row: [turn, white O-O, white O-O-O, black O-O, black O-O-O, en-passant file, state, pad].
The status row holds what the board alone doesn't show, so one frame has everything needed to
predict the next. The real rules come from python-chess. A move that isn't legal (or any move
after the game has ended) changes nothing: the model has to learn that too.
"""
import chess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS, COLS = 9, 8
N_TOKENS = ROWS * COLS
EMPTY = 0                                    # 1-6 white P N B R Q K, 7-12 black p n b r q k
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
WHITE_TO_MOVE, BLACK_TO_MOVE = 13, 14
CASTLE_YES, CASTLE_NO = 15, 16
EP_NONE = 17                                 # 18-25: en passant possible on file a..h
ONGOING, CHECK, CHECKMATE, STALEMATE = 26, 27, 28, 29
PAD = 30
N_CLASSES = 31
PROMOS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
CASTLE_ROOKS = [chess.H1, chess.A1, chess.H8, chess.A8]   # status cells 1-4
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def cell_index(square):
    """Token index (0-63) of a python-chess square."""
    return (7 - chess.square_rank(square)) * 8 + chess.square_file(square)


def square_of(index):
    return chess.square(index % 8, 7 - index // 8)


def piece_token(piece):
    return 1 + PIECES.index(piece.piece_type) + (0 if piece.color == chess.WHITE else 6)


def game_over(board):
    return board.is_checkmate() or board.is_stalemate()


def tokens(board):
    t = np.zeros(N_TOKENS, np.int64)
    for sq, piece in board.piece_map().items():
        t[cell_index(sq)] = piece_token(piece)
    rights = board.clean_castling_rights()
    t[64] = WHITE_TO_MOVE if board.turn == chess.WHITE else BLACK_TO_MOVE
    for i, rook in enumerate(CASTLE_ROOKS):
        t[65 + i] = CASTLE_YES if rights & chess.BB_SQUARES[rook] else CASTLE_NO
    t[69] = EP_NONE if board.ep_square is None else 18 + chess.square_file(board.ep_square)
    t[70] = (CHECKMATE if board.is_checkmate() else STALEMATE if board.is_stalemate()
             else CHECK if board.is_check() else ONGOING)
    t[71] = PAD
    return t


def board_from_tokens(t):
    """Rebuild a python-chess board from tokens (to let a player read a dreamed position)."""
    b = chess.Board(None)
    for i in range(64):
        v = int(t[i])
        if 1 <= v <= 12:
            b.set_piece_at(square_of(i), chess.Piece(PIECES[(v - 1) % 6], chess.WHITE if v <= 6 else chess.BLACK))
    b.turn = int(t[64]) != BLACK_TO_MOVE
    b.castling_rights = 0
    for i, rook in enumerate(CASTLE_ROOKS):
        if int(t[65 + i]) == CASTLE_YES:
            b.castling_rights |= chess.BB_SQUARES[rook]
    ep = int(t[69])
    if 18 <= ep <= 25:
        b.ep_square = chess.square(ep - 18, 5 if b.turn == chess.WHITE else 2)
    return b


def step(board, move):
    """The real rules: a legal move is played; anything else changes nothing."""
    if not game_over(board) and board.is_legal(move):
        board.push(move)


def encode_move(move):
    return cell_index(move.from_square), cell_index(move.to_square), PROMOS.index(move.promotion)


def move_kind(board, move):
    """Label a move attempt (before it is played) for per-kind accuracy reports."""
    if game_over(board):
        return "after game over"
    if not board.is_legal(move):
        return "illegal attempt"
    if board.is_castling(move):
        return "castling"
    if board.is_en_passant(move):
        return "en passant"
    if move.promotion:
        return "promotion"
    board.push(move)
    mate, check = board.is_checkmate(), board.is_check()
    board.pop()
    if mate:
        return "checkmate"
    if check:
        return "check"
    return "capture" if board.is_capture(move) else "quiet move"


def random_attempt(board, rng):
    """A random move attempt, usually illegal: one of your pieces (or any square) to any square."""
    own = [sq for sq, p in board.piece_map().items() if p.color == board.turn]
    frm = own[int(rng.integers(len(own)))] if own and rng.random() < 0.8 else int(rng.integers(64))
    return chess.Move(frm, int(rng.integers(64)))


def choose_move(board, rng, style, p_illegal):
    """Players: style 0 random legal moves; style 1 greedy (mate in one if it sees it, else
    grabs the most valuable piece it can). Both like special moves (castling, en passant,
    promotion) so they appear often, and sometimes try a random (usually illegal) move."""
    legal = list(board.legal_moves)
    if not legal or rng.random() < p_illegal:
        return random_attempt(board, rng)
    special = [m for m in legal if m.promotion or board.is_castling(m) or board.is_en_passant(m)]
    if special and rng.random() < 0.5:
        return special[int(rng.integers(len(special)))]
    if style == 1:
        if rng.random() < 0.5:
            for m in legal:
                if board.gives_check(m):
                    board.push(m)
                    mate = board.is_checkmate()
                    board.pop()
                    if mate:
                        return m
        captures = [m for m in legal if board.is_capture(m)]
        if captures and rng.random() < 0.8:
            def victim(m):
                p = board.piece_at(m.to_square)
                return VALUE[p.piece_type] if p else 1
            best = max(victim(m) for m in captures)
            captures = [m for m in captures if victim(m) == best]
            return captures[int(rng.integers(len(captures)))]
    return legal[int(rng.integers(len(legal)))]


def play_game(seed, max_plies=200):
    """One recorded game: token frames, move attempts and 'special' flags."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    styles, p_illegal = rng.integers(0, 2, 2), rng.uniform(0.05, 0.25)
    frames, acts, special, after_end = [tokens(board)], [], [], 0
    for _ in range(max_plies):
        move = choose_move(board, rng, styles[int(board.turn)], p_illegal)
        kind = move_kind(board, move)
        step(board, move)
        frames.append(tokens(board))
        acts.append(encode_move(move))
        special.append(kind not in ("quiet move", "illegal attempt"))
        if game_over(board):
            after_end += 1
            if after_end > 1:
                break
    return np.array(frames, np.uint8), np.array(acts, np.int64), np.array(special)


# ---------- drawing ----------

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SYMBOL = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜", chess.BISHOP: "♝",
          chess.KNIGHT: "♞", chess.PAWN: "♟"}


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def render(t, move=None, S=44):
    """Draw tokens as a board + status strip. move = (from_index, to_index) to highlight."""
    glyph, small = _font(int(S * 0.8)), _font(11)
    img = Image.new("RGB", (8 * S, 9 * S + 6), (12, 14, 18))
    d = ImageDraw.Draw(img)
    hl = set(move[:2]) if move is not None else set()
    for i in range(64):
        r, c = divmod(i, 8)
        x, y = c * S, r * S
        light = (r + c) % 2 == 0
        fill = ((205, 210, 106) if light else (170, 162, 58)) if i in hl else ((240, 217, 181) if light else (181, 136, 99))
        d.rectangle([x, y, x + S - 1, y + S - 1], fill=fill)
        v = int(t[i])
        if 1 <= v <= 12:
            white = v <= 6
            d.text((x + S / 2, y + S / 2 + 2), SYMBOL[PIECES[(v - 1) % 6]], font=glyph, anchor="mm",
                   fill=(250, 250, 250) if white else (25, 25, 25), stroke_width=1,
                   stroke_fill=(30, 30, 30) if white else (200, 200, 200))
        elif v:
            d.text((x + S / 2, y + S / 2), "?", font=glyph, anchor="mm", fill=(255, 60, 60))   # not a piece
    y = 8 * S + 6
    names = {WHITE_TO_MOVE: "White", BLACK_TO_MOVE: "Black", CASTLE_YES: "yes", CASTLE_NO: "no",
             EP_NONE: "-", ONGOING: "", CHECK: "check", CHECKMATE: "mate", STALEMATE: "stalemate", PAD: ""}
    labels = ["turn", "W O-O", "W O-O-O", "B O-O", "B O-O-O", "e.p.", "state", ""]
    for c in range(8):
        v = int(t[64 + c])
        text = names.get(v, "abcdefgh"[v - 18] if 18 <= v <= 25 else "?")
        x = c * S
        d.rectangle([x + 1, y, x + S - 2, y + S - 7], fill=(40, 45, 56))
        d.text((x + S / 2, y + 12), text, fill="white", font=_font(12), anchor="mm")
        d.text((x + S / 2, y + 28), labels[c], fill=(140, 146, 158), font=small, anchor="mm")
    return img
