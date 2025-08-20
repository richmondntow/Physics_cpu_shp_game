#!/usr/bin/env python3
"""
main.py - CPU-only 2D SPH fluid simulation with PyGame frontend
Uses NumPy + Numba for physics, PyGame for rendering and basic interactivity.
"""

import numpy as np
import pygame
from numba import njit
import time

# ----------------------
# Simulation parameters
# ----------------------
N = 400                # initial number of particles
DT = 0.003             # time step
H = 0.04               # smoothing radius
MASS = 0.02
REST_DENS = 1000.0
GAS_CONST = 2000.0
VISC = 250.0
G = -9.8               # gravity acceleration (m/s^2)
SPACE = 1.0            # simulation domain [0,1] in both axes

# Rendering parameters
SCREEN_W, SCREEN_H = 800, 800
PARTICLE_RADIUS = 3    # pixels
BG_COLOR = (10, 10, 30)
PARTICLE_COLOR = (120, 180, 255)

# Derived kernel constants
POLY6 = 315.0/(64.0*np.pi*H**9)
SPIKY_GRAD_COEF = -45.0/(np.pi*H**6)
VISC_LAP_COEF = 45.0/(np.pi*H**6)

# ----------------------
# Initialize particle arrays
# ----------------------
np.random.seed(0)
pos = (np.random.rand(N, 2).astype(np.float32) * 0.5) + np.array([0.25, 0.25], dtype=np.float32)
vel = np.zeros((N, 2), dtype=np.float32)
rho = np.zeros(N, dtype=np.float32)
p = np.zeros(N, dtype=np.float32)

# Numba-accelerated physics functions
@njit
def compute_density_pressure(pos, rho, p, N, H, MASS, POLY6, REST_DENS, GAS_CONST):
    for i in range(N):
        density = 0.0
        xi = pos[i,0]; yi = pos[i,1]
        for j in range(N):
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            r2 = dx*dx + dy*dy
            if r2 < H*H:
                density += MASS * POLY6 * (H*H - r2)**3
        if density < 1e-8:
            density = REST_DENS
        rho[i] = density
        p[i] = GAS_CONST * (density - REST_DENS)

@njit
def compute_forces(pos, vel, rho, p, acc, N, H, MASS, VISC, SPIKY_GRAD_COEF, VISC_LAP_COEF, G):
    for i in range(N):
        fpx = 0.0; fpy = 0.0
        fvx = 0.0; fvy = 0.0
        xi = pos[i,0]; yi = pos[i,1]
        vix = vel[i,0]; viy = vel[i,1]
        for j in range(N):
            if j == i: continue
            dx = xi - pos[j,0]
            dy = yi - pos[j,1]
            r2 = dx*dx + dy*dy
            if r2 < H*H and r2 > 1e-12:
                r = np.sqrt(r2)
                # Pressure term (spiky gradient)
                grad = SPIKY_GRAD_COEF * (H - r)**2
                pj_term = -MASS * (p[i] + p[j])/(2.0 * rho[j])
                fpx += pj_term * grad * (dx / r)
                fpy += pj_term * grad * (dy / r)
                # Viscosity
                lap = VISC_LAP_COEF * (H - r)
                vjx = vel[j,0]; vjy = vel[j,1]
                fvx += VISC * MASS * (vjx - vix) / rho[j] * lap
                fvy += VISC * MASS * (vjy - viy) / rho[j] * lap
        # gravity
        fgy = G * rho[i]
        ax = (fpx + fvx) / rho[i]
        ay = (fpy + fvy + fgy) / rho[i]
        acc[i,0] = ax; acc[i,1] = ay

@njit
def integrate(pos, vel, acc, N, DT):
    for i in range(N):
        vel[i,0] += DT * acc[i,0]
        vel[i,1] += DT * acc[i,1]
        pos[i,0] += DT * vel[i,0]
        pos[i,1] += DT * vel[i,1]
        # boundary conditions (simple bounce)
        if pos[i,0] < 0.0:
            pos[i,0] = 0.0; vel[i,0] *= -0.5
        if pos[i,0] > 1.0:
            pos[i,0] = 1.0; vel[i,0] *= -0.5
        if pos[i,1] < 0.0:
            pos[i,1] = 0.0; vel[i,1] *= -0.5
        if pos[i,1] > 1.0:
            pos[i,1] = 1.0; vel[i,1] *= -0.5

def step(pos, vel, rho, p, N):
    acc = np.zeros_like(pos)
    compute_density_pressure(pos, rho, p, N, H, MASS, POLY6, REST_DENS, GAS_CONST)
    compute_forces(pos, vel, rho, p, acc, N, H, MASS, VISC, SPIKY_GRAD_COEF, VISC_LAP_COEF, G)
    integrate(pos, vel, acc, N, DT)
    return pos

# Utility to convert simulation space [0,1] to screen coords
def sim_to_screen(pt):
    x = int(pt[0] * SCREEN_W)
    y = int((1.0 - pt[1]) * SCREEN_H)  # flip y for screen
    return x, y

def add_particle(world_pos):
    global pos, vel, rho, p, N
    if N >= 3000:
        return
    # append new particle at normalized position
    nx = world_pos[0] / SCREEN_W
    ny = 1.0 - (world_pos[1] / SCREEN_H)
    pos = np.vstack((pos, np.array([nx, ny], dtype=np.float32)))
    vel = np.vstack((vel, np.zeros(2, dtype=np.float32)))
    rho = np.concatenate((rho, np.array([REST_DENS], dtype=np.float32)))
    p = np.concatenate((p, np.array([0.0], dtype=np.float32)))
    N = pos.shape[0]

def reset_sim(n_particles=400):
    global pos, vel, rho, p, N
    N = n_particles
    np.random.seed(0)
    pos = (np.random.rand(N, 2).astype(np.float32) * 0.5) + np.array([0.25, 0.25], dtype=np.float32)
    vel = np.zeros((N, 2), dtype=np.float32)
    rho = np.zeros(N, dtype=np.float32)
    p = np.zeros(N, dtype=np.float32)

def main():
    global pos, vel, rho, p, N
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("2D SPH - CPU (NumPy + Numba) - PyGame Frontend")
    clock = pygame.time.Clock()

    # Warm-up Numba compilation (first calls)
    compute_density_pressure(pos, rho, p, N, H, MASS, POLY6, REST_DENS, GAS_CONST)
    compute_forces(pos, vel, rho, p, np.zeros_like(pos), N, H, MASS, VISC, SPIKY_GRAD_COEF, VISC_LAP_COEF, G)
    integrate(pos, vel, np.zeros_like(pos), N, DT)

    paused = False
    running = True
    frame = 0

    while running:
        t0 = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_sim(400)
                elif event.key == pygame.K_i:
                    reset_sim(min(2000, N + 100))
                elif event.key == pygame.K_k:
                    reset_sim(max(50, N - 100))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click -> add particle at mouse
                    add_particle(event.pos)

        if not paused:
            pos = step(pos, vel, rho, p, N)

        # render
        screen.fill(BG_COLOR)
        for i in range(N):
            x, y = sim_to_screen(pos[i])
            pygame.draw.circle(screen, PARTICLE_COLOR, (x, y), PARTICLE_RADIUS)
        # HUD text
        font = pygame.font.SysFont("Arial", 16)
        info = f"Particles: {N}  |  Frame: {frame}  |  FPS: {int(clock.get_fps())}"
        text_surf = font.render(info, True, (220,220,220))
        screen.blit(text_surf, (10, 10))

        pygame.display.flip()
        frame += 1
        clock.tick(60)  # cap to 60 FPS
        # print frame time occasionally
        if frame % 30 == 0:
            t1 = time.time()
            # print(f"Step time avg: {(t1-t0):.4f}s, Particles: {N}")

    pygame.quit()

if __name__ == "__main__":
    main()
