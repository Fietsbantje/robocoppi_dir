import random

def fallback_message():
    response = ["I totally hear you, I just can't really wrap my whiskers around it yet",
                "You talkin' to me? ... You talkin' to me?? ... You talkin' to me!?",
                "Prrrrr...prrrrr...prrrrr...prrrrr...prrrrr..prrrprrprrrrrrprrrrr",
                "Prrrrrrrrrrrrrr...prrrrrrrrrrrrr...prrrrrrrrrrrr...prrrrrrrrrrrr",
                "PRRRRRRRRRRRRRRRRR...PRRRRRRRRRRRRRRRR...PRRRRRRRRRRRRRRRR...PRRRPRRRPRRR"
                "Do you want to talk some more or shall we play a game?"] [random.randrange(6)]
    return response