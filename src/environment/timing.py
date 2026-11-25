"""One explicit clock shared by training, validation and all evaluation policies."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeTiming:
    begin_s: int
    warmup_s: int
    horizon_s: int
    decision_s: int

    @classmethod
    def from_config(cls, config):
        s = config['sumo']
        value = cls(s.get('begin_time', 0), s.get('warmup_seconds', 5),
                    s['num_seconds'], s['delta_time'])
        if any(not isinstance(x, int) for x in vars(value).values()):
            raise ValueError('Timing values must be integer seconds (SUMO step-length=1)')
        if value.begin_s < 0 or value.warmup_s < 0 or min(value.horizon_s, value.decision_s) < 1:
            raise ValueError('Invalid episode timing')
        return value

    @property
    def window_begin_s(self):
        return self.begin_s + self.warmup_s

    @property
    def end_s(self):
        return self.window_begin_s + self.horizon_s
