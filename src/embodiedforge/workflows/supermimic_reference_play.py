"""Bridge reference-policy history and phase resets to the standard mjlab viewer."""


class MotionViewerEnv:
    def __init__(self, backend):
        if backend.N != 1:
            raise ValueError("Reference-policy play currently requires one environment")
        self.backend = backend
        self.history = None
        self.last_step = None
        self.last_action = None

    def __getattr__(self, name):
        return getattr(self.backend.raw, name)

    @property
    def unwrapped(self):
        return self

    def reset(self, *, env_ids=None, seed=None, options=None):
        import torch

        del options
        if seed is not None:
            torch.manual_seed(seed)
        ids = (
            torch.zeros(1, dtype=torch.long, device=self.backend.device)
            if env_ids is None
            else env_ids
        )
        self.backend.reset(ids, frames=torch.zeros_like(ids))
        self.backend.done_eval[ids] = False
        self.backend.failed_eval[ids] = False
        self.backend.raw.episode_length_buf[ids] = 0
        self.history = self.last_step = self.last_action = None
        return {}, {}

    def step(self, actions):
        transition = self.backend.step(actions)
        done = transition["done"]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            self.reset(env_ids=ids)
        return {}, transition["r"], transition["term"], done & ~transition["term"], {}

    def close(self):
        self.backend.raw.close()


def build_viewer_policy(backend, policy, args):
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import ViewerConfig

    backend.raw.cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="pelvis",
        distance=2.8,
        fovy=55.0,
        elevation=-5.0,
        azimuth=120.0,
    )
    proxy = MotionViewerEnv(backend)
    wrapped = RslRlVecEnvWrapper(proxy)

    @torch.no_grad()
    def act(observation):
        del observation
        current = int(backend.ep_step[0])
        if current != proxy.last_step:
            latest = backend.mdp.obs(
                backend.state(), backend.a_prev, backend.xi, backend.f
            )
            proxy.history = (
                latest[:, None].repeat(1, args["hist"], 1)
                if proxy.history is None
                else torch.cat((proxy.history[:, 1:], latest[:, None]), 1)
            )
            proxy.last_action = policy(
                proxy.history, backend.motion.goal(backend.f, args["goal_frames"])
            )
            proxy.last_step = current
        return proxy.last_action

    return wrapped, act
