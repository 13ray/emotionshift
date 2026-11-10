"""B4 inputs: past k turns with speaker names, and the E4 controls."""

import pandas as pd
import pytest

from emotionshift.data.labels import EMOTIONS

pytest.importorskip("transformers")

from emotionshift.models.roberta import RobertaConfig
from emotionshift.models.roberta_context import RobertaContext, context_inputs, k_name, parse_k

TINY = "hf-internal-testing/tiny-random-RobertaModel"


def frame():
    rows = [(0, 0, "A", "a0"), (0, 1, "B", "b1"), (0, 2, "A", "a2"), (0, 3, "B", "b3"),
            (1, 0, "C", "c0"), (1, 1, "D", "d1"), (1, 2, "C", "c2")]
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn", "speaker", "text"])
    df["emotion"] = ["neutral", "joy", "neutral", "anger", "neutral", "sadness", "neutral"]
    return df


def test_window_sizes():
    df = frame()
    assert context_inputs(df, 0, sep="|") == ["A: a0", "B: b1", "A: a2", "B: b3", "C: c0", "D: d1", "C: c2"]
    assert context_inputs(df, 1, sep="|")[3] == "A: a2 | B: b3"
    assert context_inputs(df, 2, sep="|")[3] == "B: b1 | A: a2 | B: b3"
    assert context_inputs(df, "all", sep="|")[3] == "A: a0 | B: b1 | A: a2 | B: b3"
    assert context_inputs(df, 10, sep="|") == context_inputs(df, "all", sep="|")


def test_context_is_causal_and_stays_inside_the_dialogue():
    out = context_inputs(frame(), "all", sep="|")
    assert out[4] == "C: c0"                       # first turn of dialogue 1 sees nothing of dialogue 0
    assert "a2" not in out[1] and "b3" not in out[2]   # no future turns


def test_row_order_is_kept_for_shuffled_frames():
    df = frame()
    shuffled = df.sample(frac=1, random_state=0)
    expected = dict(zip(df.text, context_inputs(df, 2, sep="|"), strict=False))
    assert context_inputs(shuffled, 2, sep="|") == [expected[t] for t in shuffled.text]


def test_controls():
    df = frame()
    assert context_inputs(df, 3, sep="|", mode="none") == context_inputs(df, 0, sep="|")
    real = context_inputs(df, "all", sep="|")
    shuf = context_inputs(df, "all", sep="|", mode="shuffled", seed=1)
    for i, (r, s) in enumerate(zip(real, shuf, strict=False)):
        assert r.count("|") == s.count("|")                      # same number of context turns
        assert r.rsplit("| ", 1)[-1] == s.rsplit("| ", 1)[-1]    # current utterance untouched
        own = {"a", "b"} if df.dialogue_id[i] == 0 else {"c", "d"}
        assert all(part.strip()[0].lower() not in own for part in s.split("|")[:-1])   # from other dialogues
    spk = context_inputs(df, "all", sep="|", mode="speakers", seed=0)
    assert [p.split(": ")[1] for p in spk[3].split(" | ")] == ["a0", "b1", "a2", "b3"]   # text in place
    assert sorted(p.split(": ")[0] for p in spk[3].split(" | ")) == ["A", "A", "B", "B"]


def test_k_names():
    assert parse_k("all") is None and parse_k("3") == 3 and k_name(None) == "all" and k_name(5) == "5"
    with pytest.raises(ValueError):
        parse_k(-1)


def test_fit_predict_save_load(tmp_path):
    df = frame()
    m = RobertaContext(RobertaConfig(model_name=TINY, epochs=1, batch_size=4, max_len=32), device="cpu", k=2)
    m.fit_frames(df, df)
    assert m.tokenizer.truncation_side == "left"
    pred = m.predict_frame(df)
    assert len(pred) == len(df) and set(pred) <= set(EMOTIONS)
    assert len(m.predict_frame(df, mode="shuffled")) == len(df)
    assert m.predict_dialogue(["A", "B"], ["hi", "hello"])[0] in EMOTIONS
    m.save(tmp_path / "b4")
    again = RobertaContext.load(tmp_path / "b4", device="cpu")
    assert again.k == "2" and again.predict_frame(df) == pred
