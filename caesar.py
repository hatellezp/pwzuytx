from typing import Tuple, Optional


def index_no_error(element, li):
    try:
        return li.index(element)
    except ValueError:
        return -1


def letter_distance(start: str, end: str, alphabet: str) -> int:
    alpha_list = list(alphabet)

    start_index = index_no_error(start, alpha_list)
    end_index = index_no_error(end, alpha_list)

    if start_index == -1 or end_index == -1:
        return -1
    else:
        return end_index - start_index if (start_index <= end_index) else len(alpha_list) - (start_index - end_index)


def rotate_message(message: str, alphabet: str, rotation: int) -> str:
    """
    this function do not make any assumption on the alphabet, you should
    use higher level functions for that

    :param message: a string to be rotated with rotation 'rotation' and
                    alphabet 'alphabet'
    :param alphabet: a string which contains the alphabet
    :param rotation: the rotation integer
    :return: a string with the encrypted message
    """
    # transform the alphabet to a list
    alpha_list = list(alphabet)
    alpha_length = len(alpha_list)

    # return the message with rotation modulo the length of the alphabet
    # this can be done in a sole expression, but I want it to be readable
    rot_message = []
    for x in message:
        x_index = index_no_error(x, alpha_list)

        # perform rotation only in x_index is not equal to -1 (not found
        rot_x = x if x_index == -1 else alpha_list[(x_index + rotation) % alpha_length]

        # add to rot_message
        rot_message.append(rot_x)

    return ''.join(rot_message)


def rotate_message_az(message: str, rotation: int, respect_uppercase) -> str:
    # first normalize the message
    message = message if respect_uppercase else message.lower()

    alphabet_lowercase = "abcdefghijklmnopqrstuvwxyz"
    alphabet_uppercase = alphabet_lowercase.upper()
    
    rot_message = []
    for x in message:
        if x.islower():
            rot_message.append(rotate_message(x, alphabet_lowercase, rotation))
        else:
            rot_message.append(rotate_message(x, alphabet_uppercase, rotation))
            
    return ''.join(rot_message) 


def encode_message_az(message: str, rotation: int, respect_uppercase=False) -> str:
    return rotate_message_az(message, rotation, respect_uppercase)


def decode_message_az(message: str, rotation: int, respect_uppercase=False) -> str:
    return rotate_message_az(message, -rotation, respect_uppercase)


def enc_dec_message_az(message: str, rotation: Optional[int] = None, hint: Optional[Tuple[str, str]] = None, encode=True, respect_uppercase=False) -> str:
    # alphabet
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    if rotation is not None:
        rotation = rotation
    elif hint is not None:
        # compute the length of the rotation
        start, end = hint
        rotation = letter_distance(start, end, alphabet)
    else:
        print("you must provide a rotation or a hint to encode and decode")
        return message

    return encode_message_az(message, rotation, respect_uppercase) if encode else decode_message_az(message, rotation, respect_uppercase)
