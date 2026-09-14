import pytest

from embodiedforge.viewers.timing import RateClock


def test_independent_render_and_physics_deadlines_do_not_round_to_physics_ticks():
    physics, frames = RateClock(), RateClock()
    now, steps, images = 0.0, 0, 0
    while now < 0.999:
        steps += physics.due(now, 50)
        images += frames.due(now, 30)
        now = min(physics.next, frames.next)
    assert steps == 50 and images == 30


def test_overrun_and_rate_change_do_not_queue_catch_up_steps():
    clock = RateClock()
    assert clock.due(0, 50)
    assert clock.due(10, 50)  # Slow renderer or model initialization.
    assert not clock.due(10, 50)
    assert not clock.due(10.019, 50)
    assert clock.due(10.019, 25)  # Speed setting takes effect immediately.
    assert not clock.due(10.039, 25)


def test_short_render_delay_does_not_shift_every_following_physics_tick():
    clock = RateClock()
    assert clock.due(0, 50)
    assert clock.due(0.032, 50)
    assert clock.next == pytest.approx(0.040)
    assert not clock.due(0.032, 50)
    assert clock.due(0.040, 50)
