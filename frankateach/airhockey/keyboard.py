"""Key sets for air hockey teleop.

Single source of truth for which physical keys drive which arm. Values are
browser KeyboardEvent.code names, because the UI is a web page -- there is no
pygame dependency anywhere in this project any more (the NUC has no monitor).

Intent is expressed in BOX axes: +x is "up the table" (away from you), +y is
left. PlayBox.rotate_intent() turns that into a base-frame velocity, so the keys
feel the same regardless of how the arm is mounted relative to the table.
"""

KEY_SETS = {
    "wasd": {"up": "KeyW", "down": "KeyS", "left": "KeyA", "right": "KeyD"},
    "arrows": {
        "up": "ArrowUp",
        "down": "ArrowDown",
        "left": "ArrowLeft",
        "right": "ArrowRight",
    },
}

# Calibration jog only: raise/lower the EE to find the play height.
JOG_UP = "KeyE"
JOG_DOWN = "KeyQ"

KEY_HOME = "KeyH"
KEY_FREEZE = "Space"
KEY_RELEASE = "Escape"


def resolve_keys(name):
    if name not in KEY_SETS:
        raise ValueError(f"Unknown key set {name!r}, expected one of {list(KEY_SETS)}")
    return KEY_SETS[name]


def intent_from_held(held, keys):
    """(vx, vy) in {-1,0,1}^2 from a set of held KeyboardEvent.code strings."""
    vx = float(keys["up"] in held) - float(keys["down"] in held)
    vy = float(keys["left"] in held) - float(keys["right"] in held)
    return vx, vy
