import random

parenthesis = {
    "(": ")",
    "[": "]",
}

VOCAB = ["(", ")", "[", "]", "[PAD]", "[CLS]", "[SEP]"]


def generate_clean(parenthesis, min_len=4, max_len=40, max_depth=4):
    opens = list(parenthesis.keys())
    closes = parenthesis

    stack = []
    out = []

    while True:
        remaining = max_len - len(out)

        if len(out) >= min_len and (len(stack) == 0 or remaining <= len(stack)):
            break

        can_open = len(stack) < max_depth and remaining > len(stack) + 1
        can_close = len(stack) > 0

        if len(out) < min_len:
            p_open = 0.8
        else:
            p_open = min(0.7, max(0.3, remaining / max_len))

        if can_open and (not can_close or random.random() < p_open):
            op = random.choice(opens)
            out.append(op)
            stack.append(op)
        else:
            op = stack.pop()
            out.append(closes[op])

    while stack:
        out.append(closes[stack.pop()])

    return out


def E1_missing_closer(seq, parenthesis):
    closers = [i for i, x in enumerate(seq) if x in parenthesis.values()]
    if not closers:
        return seq, None
    i = random.choice(closers)

    corrupted = seq[:i] + seq[i+1:]
    # to repair: insert deleted token back before position i
    return corrupted, ("INSERT", i, seq[i])


def E2_spurious_opener(seq, parenthesis):
    i = random.randint(0, len(seq))
    tok = random.choice(list(parenthesis.keys()))

    corrupted = seq[:i] + [tok] + seq[i:]
    # to repair: delete inserted opener at i
    return corrupted, ("DELETE", i)


def E3_type_mismatch(seq, parenthesis):
    closers = list(parenthesis.values())
    idxs = [i for i, x in enumerate(seq) if x in closers]
    if not idxs:
        return seq, None

    i = random.choice(idxs)
    correct = seq[i]
    wrong = random.choice([c for c in closers if c != correct])

    corrupted = seq.copy()
    corrupted[i] = wrong
    # to repair: replace wrong token with correct one
    return corrupted, ("REPLACE", i, correct)


def E4_premature_close(seq, parenthesis):
    i = random.randint(0, len(seq))
    tok = random.choice(list(parenthesis.values()))

    corrupted = seq[:i] + [tok] + seq[i:]
    # to repair: delete inserted premature closer
    return corrupted, ("DELETE", i)


def make_edit_labels(corrupted, edit_info):
    """
    Returns token-level labels aligned with corrupted input.
    Labels in:
      OK, DELETE, INSERT(x), REPLACE(x)
    """
    labels = ["OK"] * len(corrupted)

    if edit_info is None:
        return labels

    op = edit_info[0]

    if op == "DELETE":
        i = edit_info[1]
        if 0 <= i < len(labels):
            labels[i] = "DELETE"

    elif op == "REPLACE":
        i = edit_info[1]
        gold_tok = edit_info[2]
        if 0 <= i < len(labels):
            labels[i] = f"REPLACE({gold_tok})"

    elif op == "INSERT":
        i = edit_info[1]
        gold_tok = edit_info[2]
        # insertion before position i in raw seq -> shift by +1 after [CLS]
        if i < len(labels):
            labels[i] = f"INSERT({gold_tok})"

    return labels


def generate_examples(nb_ex, min_len=4, max_len=40, max_depth=4, corrupt_prob=0.5, task="both"):
    """
    task:
      - "detection"  -> returns (tokens, binary_label)
      - "correction" -> returns (tokens, edit_labels)
      - "both"       -> returns (tokens, binary_label, edit_labels)
    """
    examples = []

    error_functions = [
        E1_missing_closer,
        E2_spurious_opener,
        E3_type_mismatch,
        E4_premature_close
    ]

    for _ in range(nb_ex):
        clean = generate_clean(parenthesis, min_len=min_len, max_len=max_len-2, max_depth=max_depth)
        corrupted = clean.copy()

        binary_label = 1
        edit_info = None

        if random.random() < corrupt_prob:
            binary_label = 0
            error = random.choice(error_functions)
            corrupted, edit_info = error(corrupted, parenthesis)

        # add specials
        corrupted = ["[CLS]"] + corrupted + ["[SEP]"]
        clean = ["[CLS]"] + clean + ["[SEP]"]

        # shift edit positions because of [CLS]
        if edit_info is not None:
            if edit_info[0] == "DELETE":
                edit_info = ("DELETE", edit_info[1] + 1)
            elif edit_info[0] == "REPLACE":
                edit_info = ("REPLACE", edit_info[1] + 1, edit_info[2])
            elif edit_info[0] == "INSERT":
                edit_info = ("INSERT", edit_info[1] + 1, edit_info[2])

        edit_labels = make_edit_labels(corrupted, edit_info)

        while len(corrupted) < 80:
            corrupted.append("[PAD]")
            edit_labels.append("OK")

        corrupted = corrupted[:80]
        edit_labels = edit_labels[:80]

        if task == "detection":
            examples.append((corrupted, binary_label))
        elif task == "correction":
            examples.append((corrupted, edit_labels))
        else:
            examples.append((corrupted, binary_label, edit_labels))

    return examples

# generate tracking error types
def generate_examples_with_error_type(nb_ex, min_len=4, max_len=40, max_depth=4, corrupt_prob=0.5, task="both"):
    examples = []
    error_functions = [
        ("E1", E1_missing_closer),
        ("E2", E2_spurious_opener),
        ("E3", E3_type_mismatch),
        ("E4", E4_premature_close),
    ]
    for _ in range(nb_ex):
        clean = generate_clean(parenthesis, min_len=min_len, max_len=max_len-2, max_depth=max_depth)
        corrupted = clean.copy()
        binary_label = 1
        error_type = "none"
        edit_info = None

        if random.random() < corrupt_prob:
            binary_label = 0
            name, fn = random.choice(error_functions)
            corrupted, edit_info = fn(corrupted, parenthesis)
            error_type = name

        corrupted = ["[CLS]"] + corrupted + ["[SEP]"]

        if edit_info is not None:
            if edit_info[0] == "DELETE":
                edit_info = ("DELETE", edit_info[1] + 1)
            elif edit_info[0] == "REPLACE":
                edit_info = ("REPLACE", edit_info[1] + 1, edit_info[2])
            elif edit_info[0] == "INSERT":
                edit_info = ("INSERT", edit_info[1] + 1, edit_info[2])

        edit_labels = make_edit_labels(corrupted, edit_info)

        while len(corrupted) < 80:
            corrupted.append("[PAD]")
            edit_labels.append("OK")

        corrupted = corrupted[:80]
        edit_labels = edit_labels[:80]

        if task == "both":
            examples.append((corrupted, binary_label, edit_labels, error_type))
        elif task == "detection":
            examples.append((corrupted, binary_label, error_type))
        elif task == "correction":
            examples.append((corrupted, edit_labels, error_type))

    return examples