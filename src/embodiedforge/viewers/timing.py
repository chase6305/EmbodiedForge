"""Wall-clock deadlines without catch-up bursts or render-time double counting."""


class RateClock:
    def __init__(self):
        self.next = 0.0
        self.rate = None

    def due(self, now, rate):
        if rate != self.rate:
            self.next, self.rate = now, rate
        if now < self.next:
            return False
        period = 1 / rate
        # Preserve phase after a short render delay. Reset only when an entire
        # additional tick was missed; never accumulate catch-up work.
        self.next = self.next + period if now - self.next < period else now + period
        return True
