import random

def fallback_message():
    response = ["I totally hear you, I just can't really wrap my whiskers around it yet",
                "You talkin' to me? ... You talkin' to me?? ... You talkin' to me!?",
                "Prrrrr...prrrrr...prrrrr...prrrrr...prrrrr..prrrprrprrrrrrprrrrr"] [random.randrange(3)]
    return response