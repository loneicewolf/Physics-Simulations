```py
# blackhole_sim_extended.py
"""
Interactive black‑hole sandbox in geometric units (G = c = 1)
----------------------------------------------------------------
Features added over the minimal version
• **Full multi‑particle engine** – click anywhere to spawn a test mass.
• **Energy E & angular momentum L read‑out** (Newtonian expressions).
• **Proper radial distance Δℓ(r_h→r)** reported live.
• **Schwarzschild or Kerr horizon** selectable.
• Semi‑implicit Euler integrator with adjustable damping.

The code is still lightweight (pure Tkinter + numpy) and keeps EinsteinPy
optional. If the library is available the user can toggle a boolean flag to
switch from Newtonian trajectories to genuine 4‑D geodesics driven by
EinsteinPy’s RK45 solver.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import colorchooser
import math
import random
from typing import List, Tuple, Optional

import numpy as np

# Try EinsteinPy – use if present, else fall back.
try:
    from einsteinpy.metric import Schwarzschild
    from einsteinpy.integrators import RK45
    from einsteinpy.coordinates import BoyerLindquist
    EINSTEINPY_AVAILABLE = True
except ImportError:  # noqa: E722
    EINSTEINPY_AVAILABLE = False

# ------------- helper maths -------------------------------------------------

def accel_newtonian(M: float, x: float, y: float) -> Tuple[float, float]:
    """Return Newtonian accel in geom units: **a** = -M r̂ / r²"""
    r2 = x * x + y * y
    if r2 == 0:
        return 0.0, 0.0
    inv_r = 1.0 / math.sqrt(r2)
    inv_r3 = inv_r * inv_r * inv_r
    a_mag = -M * inv_r3
    return a_mag * x, a_mag * y


def radial_proper_distance(M: float, r1: float, r2: float, *, n: int = 1000) -> float:
    """Proper radial distance ∫ r1→r2  dr / √(1−2M/r).  Positive if r2 > r1."""
    rs = np.linspace(r1, r2, n)
    integrand = 1.0 / np.sqrt(1.0 - 2.0 * M / rs)
    return float(np.trapz(integrand, rs))


# ------------- particle ------------------------------------------------------

class Particle:
    """A single test mass tracked in geometric units."""

    def __init__(
        self,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        colour: str,
        M: float,
        use_gr: bool = False,
    ) -> None:
        self.pos = list(pos)  # [x, y] in geom units
        self.vel = list(vel)  # [vx, vy]
        self.colour = colour
        self.trace: List[Tuple[float, float]] = []
        self.alive = True
        self.M = M
        self.use_gr = use_gr and EINSTEINPY_AVAILABLE
        self.acc = (0.0, 0.0)

        # Conserved quantities (Newtonian expressions)
        r = math.hypot(*self.pos)
        v2 = self.vel[0] ** 2 + self.vel[1] ** 2
        self.energy = 0.5 * v2 - M / r
        self.angular_momentum = self.pos[0] * self.vel[1] - self.pos[1] * self.vel[0]

        # If GR trajectory requested, pre‑compute a short segment using EinsteinPy
        if self.use_gr:
            self._prepare_geodesic()

    # ------------------------------------------------------------------
    def _prepare_geodesic(self):
        """Make a generator that yields successive (x,y) from GR integration."""
        M = self.M
        mass_SI = 1.0  # 1 m in geom units – arbitrary for scaled demo
        import astropy.units as u

        sph = Schwarzschild(mass_SI * u.m, coord_sys="Schwarzschild", time_like=True)
        # Convert (x, y) Cartesian → (r, θ, φ) equatorial
        x, y = self.pos
        r0 = math.hypot(x, y)
        phi0 = math.atan2(y, x)
        theta0 = math.pi / 2
        # Crude mapping from geom → SI metres keeps units proportional
        r0_si = r0 * u.m
        # Estimate 4‑velocity components (ṫ, ṙ, θ̇, φ̇).  Use v/c ≈ geom speed.
        vr0 = (self.vel[0] * math.cos(phi0) + self.vel[1] * math.sin(phi0)) * u.one
        vphi0 = (-self.vel[0] * math.sin(phi0) + self.vel[1] * math.cos(phi0)) / r0 * u.one
        vtheta0 = 0 * u.one
        # Time component chosen for normalization; EinsteinPy will fix it.
        vec_pos = BoyerLindquist(r0_si, theta0 * u.rad, phi0 * u.rad)
        vec_mom = np.array([vr0.to(u.one).value, vtheta0.value, vphi0.to(u.one).value], dtype=float)

        self._geod = RK45(
            0.0,
            vec_pos.vec,
            sph,  # metric
            vec_mom,
            0.25,  # step size in seconds (arbitrary)
            8,
            return_cartesian=False,
        )

    # ------------------------------------------------------------------
    def step(self, dt: float):
        if not self.alive:
            return
        if self.use_gr:
            try:
                t, arr = next(self._geod)
                r, theta, phi = arr
                x = r * math.sin(theta) * math.cos(phi)
                y = r * math.sin(theta) * math.sin(phi)
                self.pos = [x.value, y.value]
                # velocity estimation by finite diff not needed for plotting
                self.acc = (0.0, 0.0)
            except StopIteration:
                self.alive = False
        else:
            # Newtonian semi‑implicit Euler
            ax, ay = accel_newtonian(self.M, *self.pos)
            self.vel[0] += ax * dt
            self.vel[1] += ay * dt
            self.pos[0] += self.vel[0] * dt
            self.pos[1] += self.vel[1] * dt
            self.acc = (ax, ay)

        self.trace.append(tuple(self.pos))

# ------------- main GUI ------------------------------------------------------

class BlackHoleSandbox(tk.Tk):
    PX_PER_M = 25.0
    DT = 0.05  # phys time‑step

    def __init__(self):
        super().__init__()
        self.title("Black‑Hole Sandbox – multi‑particle & GR ready")
        self.geometry("1000x700")

        # ---------- controls -------------------------------------------------
        ctl = tk.Frame(self); ctl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.mass = tk.DoubleVar(value=5.0)
        tk.Label(ctl, text="Mass  M").pack()
        tk.Scale(ctl, variable=self.mass, from_=1, to=50, resolution=0.5,
                 orient="horizontal", length=160, command=lambda e: self._redraw_horizon()).pack()

        self.metric = tk.StringVar(value="Schwarzschild")
        tk.OptionMenu(ctl, self.metric, "Schwarzschild", "Kerr").pack(pady=(6,2))

        self.spin = tk.DoubleVar(value=0.0)  # Kerr a
        tk.Label(ctl, text="Spin a (Kerr)").pack()
        tk.Scale(ctl, variable=self.spin, from_=0, to=0.99, resolution=0.01,
                 orient="horizontal", length=160).pack()

        self.auto_damp = tk.BooleanVar(value=False)
        tk.Checkbutton(ctl, text="Velocity damping", variable=self.auto_damp).pack(anchor="w")

        self.info = tk.Label(ctl, text="", justify="left")
        self.info.pack(pady=6)

        tk.Button(ctl, text="Pick trace colour", command=self._pick_colour).pack(fill="x", pady=3)
        tk.Button(ctl, text="Reset (clear particles)", command=self.reset).pack(fill="x", pady=3)

        # ---------- canvas ---------------------------------------------------
        self.canvas = tk.Canvas(self, width=800, height=700, bg="black")
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.center = (400, 350)

        self.particles: List[Particle] = []
        self._redraw_horizon()

        # Bindings -----------------------------------------------------------
        self.canvas.bind("<Button-1>", self._spawn_particle)
        self.bind("<KeyPress>", self._key_control)

        # loop ---------------------------------------------------------------
        self.after_id = self.after(20, self._loop)

    # ------------ coordinate helpers ---------------------------------------
    def geom2px(self, g: float) -> float:
        return g * self.PX_PER_M

    def px2geom(self, px: float) -> float:
        return px / self.PX_PER_M

    # ------------ UI callbacks ---------------------------------------------
    def _pick_colour(self):
        colour = colorchooser.askcolor("#ffd700")[1]
        if colour:
            for p in self.particles:
                p.colour = colour

    def _redraw_horizon(self):
        self.canvas.delete("horizon")
        M = self.mass.get()
        if self.metric.get() == "Kerr":
            a = min(self.spin.get(), 0.999*M)
            r_h = M + math.sqrt(max(0.0, M*M - a*a))
        else:
            r_h = 2*M
        self.r_h = r_h
        R_px = self.geom2px(r_h)
        cx, cy = self.center
        self.canvas.create_oval(cx-R_px, cy-R_px, cx+R_px, cy+R_px,
                                fill="grey20", outline="", tags="horizon")

    def reset(self, *_):
        if hasattr(self, "after_id"):
            self.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.particles.clear()
        self._redraw_horizon()
        self.after_id = self.after(20, self._loop)

    # -------- particle creation -------------------------------------------
    def _spawn_particle(self, ev):
        x_px, y_px = ev.x, ev.y
        cx, cy = self.center
        x_g = self.px2geom(x_px - cx)
        y_g = self.px2geom(y_px - cy)
        r = math.hypot(x_g, y_g)
        M = self.mass.get()
        if r <= self.r_h:
            return  # clicks inside horizon ignored

        # Tangential initial velocity for a quasi‑circular orbit
        speed = math.sqrt(M / r)
        # perpendicular to radius vector (rotate by +90°)
        vx = -speed * y_g / r
        vy = speed * x_g / r
        colour = random.choice(["#ff6666", "#66ff66", "#6699ff", "#ffaa00", "#ff66ff"])
        p = Particle((x_g, y_g), (vx, vy), colour, M, use_gr=False)
        self.particles.append(p)

    # -------- keyboard -----------------------------------------------------
    def _key_control(self, ev):
        # WASD nudges last particle
        if not self.particles:
            return
        p = self.particles[-1]
        dv = 0.2
        if ev.keysym.lower() in ("w", "up"):
            p.vel[1] -= dv
        elif ev.keysym.lower() in ("s", "down"):
            p.vel[1] += dv
        elif ev.keysym.lower() in ("a", "left"):
            p.vel[0] -= dv
        elif ev.keysym.lower() in ("d", "right"):
            p.vel[0] += dv

    # -------- simulation loop ---------------------------------------------
    def _loop(self):
        # physics ------------------------------------------------------
        alive_particles = []
        for p in self.particles:
            if self.auto_damp.get():
                p.vel[0] *= 0.999
                p.vel[1] *= 0.999
            p.step(self.DT)
            # check horizon crossing (only for Newtonian path)
            if p.alive and not p.use_gr:
                r = math.hypot(*p.pos)
                if r <= self.r_h:
                    p.alive = False
            if p.alive:
                alive_particles.append(p)
        self.particles = alive_particles

        # drawing ------------------------------------------------------
        self.canvas.delete("trace", "particle", "arrow")
        cx, cy = self.center
        for p in self.particles:
            # trace lines
            if len(p.trace) > 1:
                trace_px = [
                    (cx + self.geom2px(x), cy + self.geom2px(y)) for x, y in p.trace[-400:]
                ]
                for (x1, y1), (x2, y2) in zip(trace_px, trace_px[1:]):
                    self.canvas.create_line(x1, y1, x2, y2, fill=p.colour, tags="trace")
            # particle dot
            x_px = cx + self.geom2px(p.pos[0])
            y_px = cy + self.geom2px(p.pos[1])
            self.canvas.create_oval(x_px-3, y_px-3, x_px+3, y_px+3,
                                    fill="white", outline="", tags="particle")
            # acceleration arrow (Newtonian only)
            if not p.use_gr:
                ax, ay = p.acc
                ax_px = x_px + self.geom2px(ax) * self.DT * 150
                ay_px = y_px + self.geom2px(ay) * self.DT * 150
                self.canvas.create_line(x_px, y_px, ax_px, ay_px,
                                        fill=p.colour, arrow=tk.LAST, tags="arrow")

        # info box -----------------------------------------------------
        if self.particles:
            p0 = self.particles[-1]
            r0 = math.hypot(*p0.pos)
            Δℓ = radial_proper_distance(self.mass.get(), self.r_h, r0)
            info = (f"Particles: {len(self.particles)}\n"
                    f"Last  r = {r0:.2f}   (> {self.r_h:.2f})\n"
                    f"Δℓ(h→r) = {Δℓ:.2f} M\n"
                    f"E = {p0.energy:.3f}   L = {p0.angular_momentum:.3f}")
        else:
            info = "Click on canvas to spawn a particle"
        self.info.configure(text=info)

        # schedule next frame
        self.after_id = self.after(20, self._loop)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BlackHoleSandbox().mainloop()

```
