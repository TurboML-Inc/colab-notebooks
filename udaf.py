def create_state():
    return 0, 0


def accumulate(state, value, weight):
    if value is None or weight is None:
        return state
    (s, w) = state
    s += value * weight
    w += weight
    return s, w


def retract(state, value, weight):
    if value is None or weight is None:
        return state
    (s, w) = state
    s -= value * weight
    w -= weight
    return s, w


def merge_states(state_a, state_b):
    (s_a, w_a) = state_a
    (s_b, w_b) = state_b
    return s_a + s_b, w_a + w_b


def finish(state):
    (sum, weight) = state
    if weight == 0:
        return None
    else:
        return sum / weight
