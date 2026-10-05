import re


# Matches labels such as "Support1:", "**Attack 2**:", "- Supporting argument 3)" at the start of a line
ARG_LABEL_RE = re.compile(
    r"^[\s>*_#\-\d.]*(support|attack)\w*(?:\s+argument)?[\s_]*(\d+)[\s*_]*[:.)\-][\s*_]*",
    re.I | re.M,
)


def clean_argument(argument):
    """Normalises a single generated argument, returning "N/A" if it is empty or declined."""
    argument = re.sub(r"\s+", " ", argument.replace("*", "")).strip(" \"'`")
    if not argument or re.match(r"^n/?a\b", argument, re.I):
        return "N/A"
    # Drop a trailing incomplete sentence (e.g. cut off by the token limit)
    sentences = re.findall(r".*?[.?!](?=\s|$)", argument)
    if sentences:
        argument = "".join(sentences).strip()
    return argument


def extract_args(text, breadth=None):
    """
    Splits a joint generation of the form "Support1: ... Support2: ... Attack1: ..." into
    lists of supporting and attacking arguments. Declined ("N/A"), empty and duplicate
    arguments are dropped, and each list is truncated to at most `breadth` arguments.
    """
    labels = list(ARG_LABEL_RE.finditer(text))
    support_list, attack_list, seen = [], [], set()
    for i, match in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
        # Ignore anything after a blank line following the last argument (e.g. commentary)
        body = text[match.end() : end]
        if i + 1 == len(labels):
            body = body.split("\n\n")[0]
        argument = clean_argument(body)
        if argument == "N/A" or argument.lower() in seen:
            continue
        seen.add(argument.lower())
        if match.group(1).lower() == "support":
            support_list.append(argument)
        else:
            attack_list.append(argument)

    if breadth is not None:
        support_list, attack_list = support_list[:breadth], attack_list[:breadth]
    return support_list, attack_list


def construct_constraint_fun(
    tokenizer, prompt, force_prefix=None, force_options=None, end_after_options=False
):
    # Note that we disregard the BOS token when using input IDs
    if force_prefix is not None:
        force_prefix = tokenizer(force_prefix).input_ids[1:]
    if force_options is not None:
        force_options = [tokenizer(op).input_ids[1:] for op in force_options]
    all_tokens = list(tokenizer.get_vocab().values())

    def constraint_fun(batch_id, input_ids):
        prompt_len = len(tokenizer(prompt).input_ids)
        generated_tokens = input_ids[prompt_len:].tolist()
        num_generated = len(generated_tokens)
        prefix_len = 0 if force_prefix is None else len(force_prefix)

        if force_prefix is not None and num_generated < prefix_len:
            # Force prefix to be generated first if provided
            return [force_prefix[num_generated]]
        elif num_generated >= prefix_len and force_options is not None:
            # Determine what option tokens have been generated
            op_tokens = generated_tokens[prefix_len:]
            num_op = len(op_tokens)

            # Calculate valid option continuations
            possible_continuations = [
                c[num_op]
                for c in force_options
                if num_op < len(c) and c[:num_op] == op_tokens
            ]

            if not possible_continuations and end_after_options:
                # No further continuations — terminate generation as requested
                return [tokenizer.eos_token_id]
            elif not possible_continuations:
                # No further continuations, but can continue free generation
                return all_tokens
            else:
                # Allow generation to terminate if desirable
                if op_tokens in force_options:
                    possible_continuations.append(tokenizer.eos_token_id)
                # Force generation according to options
                return possible_continuations
        else:
            return all_tokens

    return constraint_fun
