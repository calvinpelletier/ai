import chess


def move_to_coords(move, player):
    x1, y1 = square_to_coords(move.from_square, player)
    x2, y2 = square_to_coords(move.to_square, player)
    return x1, y1, x2, y2

def coords_to_move(board, player, x1, y1, x2, y2, underpromo):
    square1 = coords_to_square(x1, y1, player)
    square2 = coords_to_square(x2, y2, player)
    return board.find_move(square1, square2, underpromo)


def square_to_coords(square, player):
    x = chess.square_file(square)
    y = perspective(chess.square_rank(square), player)
    return x, y

def coords_to_square(x, y, player):
    return chess.square(x, perspective(y, player))


def color_to_side(color, player):
    return player if color else -player

def side_to_color(side, player):
    return player == 1 if side == 1 else player == -1


# make y=0 the top of the board from perspective of player
def perspective(y, player):
    if player == 1:
        return 7 - y
    elif player == -1:
        return y
    else:
        raise Exception(player)


def eq_boards(a, b):
    if a.has_legal_en_passant() != b.has_legal_en_passant():
        return False

    if a.has_legal_en_passant() and a.ep_square != b.ep_square:
        return False

    return a.board_fen() == b.board_fen() and \
        a.castling_rights == b.castling_rights and \
        a.turn == b.turn


def board_to_key(board):
    fen = board.fen()
    board, turn, castling, ep, _, _ = fen.split(' ')
    return ' '.join([board, turn, castling, ep])
