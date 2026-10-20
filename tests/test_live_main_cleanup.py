"""The live entry point closes both owners and preserves the first failure."""

from unittest.mock import MagicMock

import pytest

from embodiedforge import go1_live


@pytest.mark.parametrize(
    "primary_type", [None, RuntimeError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("viewer_type", [None, OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "policy_type", [None, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_main_retains_primary_and_closes_both_owners(
    tmp_path, monkeypatch, primary_type, viewer_type, policy_type
):
    primary = primary_type("initial update failed") if primary_type else None
    viewer_error = viewer_type("viewer close failed") if viewer_type else None
    policy_error = policy_type("policy close failed") if policy_type else None
    order = []
    viewer, policy = MagicMock(), MagicMock()
    viewer.is_running.return_value = False
    policy.metadata = {"iteration": 0, "command_limits": [1, 1, 1]}
    policy.update.side_effect = primary

    def close(name, error):
        order.append(name)
        if error is not None:
            raise error

    viewer.close.side_effect = lambda: close("viewer", viewer_error)
    policy.close.side_effect = lambda: close("policy", policy_error)
    monkeypatch.setattr(go1_live, "LivePolicyProcess", lambda *a, **kw: policy)
    monkeypatch.setattr(go1_live, "WebViewer", lambda *a, **kw: viewer)
    monkeypatch.setattr(
        go1_live, "frame_dependencies", lambda *a: {"metadata_ok": True}
    )
    monkeypatch.setattr(go1_live, "setup_logging", lambda *a: None)
    monkeypatch.setattr(go1_live, "get_logger", lambda *a: MagicMock())
    # Ctrl+C remains a normal stop at this entry point; later cleanup failures
    # must still be reported. Other primary errors remain observable.
    expected = (
        primary
        if primary is not None and primary_type is not KeyboardInterrupt
        else viewer_error or policy_error
    )
    if expected is None:
        go1_live.main(["--run", str(tmp_path)])
    else:
        with pytest.raises(BaseException) as caught:
            go1_live.main(["--run", str(tmp_path)])
        assert caught.value is expected
    assert order == ["viewer", "policy"]
    viewer.close.assert_called_once()
    policy.close.assert_called_once()
