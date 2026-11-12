import numpy as np
import torch

from catadye.model import GPT, GPTConfig
from catadye.testtime import freeze_all_but_mlp, run_ttt, stream_batch

CFG = GPTConfig(vocab_size=100, n_layer=2, n_head=2, d_model=32, ctx=16, n_sub=3)


def _setup():
    torch.manual_seed(0)
    m = GPT(CFG)
    stream = np.random.default_rng(0).integers(0, 100, 16 * 4 * 40 + 2).astype(np.uint16)
    ev = (torch.randint(0, 100, (4, 16)), torch.randint(0, 100, (4, 16)))
    return m, stream, ev


def _snap(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def test_frozen_arm_changes_nothing():
    m, stream, ev = _setup()
    before = _snap(m)
    run_ttt(m, stream, ev, ev, "frozen", 0.0, "cpu", n_steps=8, bs_windows=4, ctx=16, eval_every=4)
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters())


def test_only_mlp_sublocks_update():
    m, stream, ev = _setup()
    before = _snap(m)
    run_ttt(m, stream, ev, ev, "ttt_single", 1e-2, "cpu", n_steps=4, bs_windows=4, ctx=16, eval_every=4)
    for n, p in m.named_parameters():
        is_mlp = ".mlp.subs." in n
        assert (not torch.equal(before[n], p)) == is_mlp, n


def test_cms_slow_block_updates_only_on_period():
    m, stream, ev = _setup()
    before = _snap(m)
    run_ttt(m, stream, ev, ev, "ttt_cms", 1e-2, "cpu", periods=(1, 2, 4), n_steps=3, bs_windows=4, ctx=16,
            eval_every=3)
    slow = [n for n in before if ".mlp.subs.2." in n]
    assert all(torch.equal(before[n], dict(m.named_parameters())[n]) for n in slow)  # period 4 not reached in 3 steps


def test_stream_windows_are_consecutive_and_shifted():
    stream = np.arange(16 * 4 * 3 + 2).astype(np.uint16)
    x, y = stream_batch(stream, 2, 4, 16, "cpu")
    assert x[0, 0].item() == 64 and (y[:, :] == x + 1).all()


def test_score_before_update():
    m, stream, ev = _setup()
    r = run_ttt(m, stream, ev, ev, "ttt_single", 1e-2, "cpu", n_steps=1, bs_windows=4, ctx=16, eval_every=1)
    m2, _, _ = _setup()
    x, y = stream_batch(stream, 1, 4, 16, "cpu")
    with torch.no_grad():
        _, l0 = m2(x, y)
    assert abs(r["online"][0] - l0.item()) < 1e-5  # first online loss is from the *untouched* weights
