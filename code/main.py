import pygame
from pygame.locals import *

# Inizializzazione Pygame
pygame.init()

# Impostazioni finestra di gioco
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Super Mario Bros")

# Colori
white = (255, 255, 255)
blue = (0, 0, 255)

# Personaggio Mario
mario_width, mario_height = 50, 50
mario_x, mario_y = 50, height - mario_height - 10
mario_vel = 5

# Piattaforme del livello
platforms = [
    (0, height - 20, width, 20),  # Piattaforma di base
    (300, height - 100, 200, 10),  # Piattaforma sopra
]

# Ciclo di gioco
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Movimento di Mario
    keys = pygame.key.get_pressed()
    if keys[K_LEFT] and mario_x > 0:
        mario_x -= mario_vel
    if keys[K_RIGHT] and mario_x < width - mario_width:
        mario_x += mario_vel

    # Aggiornamento schermo
    screen.fill(white)

    # Disegna piattaforme
    for platform in platforms:
        pygame.draw.rect(screen, blue, platform)

    # Disegna Mario
    pygame.draw.rect(screen, blue, (mario_x, mario_y, mario_width, mario_height))

    # Gestione collisioni con le piattaforme
    for platform in platforms:
        if (
            mario_x < platform[0] + platform[2]
            and mario_x + mario_width > platform[0]
            and mario_y < platform[1] + platform[3]
            and mario_y + mario_height > platform[1]
        ):
            mario_y = platform[1] - mario_height

    pygame.display.flip()

    # Imposta il frame rate
    clock.tick(30)

# Uscita dal gioco
pygame.quit()
