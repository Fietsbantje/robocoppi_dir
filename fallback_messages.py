import random

def fallback_message():
    response = ["I totally hear you, I just can't really wrap my whiskers around it yet",
                "You talkin' to me? ... You talkin' to me?? ... You talkin' to me!?"] [random.randrange(2)]
    return response     