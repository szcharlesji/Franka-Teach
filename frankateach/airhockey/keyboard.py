"""pygame keyboard -> per-arm movement intent.

Intent is expressed in BOX axes: +x is "up the table" (away from you), +y is
left. box.rotate_intent() turns that into a base-frame velocity, so the keys
feel the same regardless of how the arm is mounted relative to the table.
"""

import pygame

KEY_SETS = {
    "wasd": {"up": pygame.K_w, "down": pygame.K_s, "left": pygame.K_a, "right": pygame.K_d},
    "arrows": {
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
    },
}

# Calibration jog only: raise/lower the EE to find the play height.
JOG_UP = pygame.K_e
JOG_DOWN = pygame.K_q

KEY_HOME = pygame.K_h
KEY_FREEZE = pygame.K_SPACE
KEY_RELEASE = pygame.K_ESCAPE
KEY_RECORD = pygame.K_SPACE
KEY_REDO = pygame.K_r
KEY_DONE = pygame.K_RETURN


def intent_from_keys(pressed, keys):
    """(vx, vy) in {-1,0,1}^2 from a pygame key-state array."""
    vx = float(pressed[keys["up"]]) - float(pressed[keys["down"]])
    vy = float(pressed[keys["left"]]) - float(pressed[keys["right"]])
    return vx, vy


def resolve_keys(name):
    if name not in KEY_SETS:
        raise ValueError(f"Unknown key set {name!r}, expected one of {list(KEY_SETS)}")
    return KEY_SETS[name]
