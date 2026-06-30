from mars_core.evaluator import canonical_answer, is_valid_prediction


def test_option_letter_parser():
    assert canonical_answer("Final answer: (C)", "option_letter") == "(C)"
    assert is_valid_prediction("(C)", "option_letter")


def test_boolean_parser():
    assert canonical_answer("Final answer: false", "boolean") == "false"
    assert is_valid_prediction("true", "boolean")


def test_yes_no_parser():
    assert canonical_answer("Therefore, yes.", "yes_no") == "yes"
    assert is_valid_prediction("no", "yes_no")


def test_valid_invalid_parser_prefers_invalid():
    assert canonical_answer("This argument is invalid.", "valid_invalid") == "invalid"
    assert is_valid_prediction("valid", "valid_invalid")


def test_numeric_parser():
    assert canonical_answer("#### 1,234.5", "numeric") == "1234.5"
    assert is_valid_prediction("-3.14", "numeric")
