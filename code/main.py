import pygame
import sys

# Inizializzazione di Pygame
pygame.init()

# Costanti
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)

# Classe per rappresentare Mario
class Mario(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("mario.png").convert_alpha()
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50)
        self.velocity_y = 0
        self.jump_power = -15  # Potenza del salto
        self.shrink_factor = 0.5  # Fattore di rimpicciolimento
        self.shrunk = False  # Indica se Mario è rimpicciolito

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT]:
            self.rect.x += 5

        # Simulazione della gravità
        self.velocity_y += 1
        self.rect.y += self.velocity_y

        # Assicura che Mario rimanga all'interno dello schermo
        if self.rect.y > SCREEN_HEIGHT - self.rect.height:
            self.rect.y = SCREEN_HEIGHT - self.rect.height
            self.velocity_y = 0

        # Controllo del salto
        if keys[pygame.K_SPACE] and (self.rect.y == SCREEN_HEIGHT - self.rect.height or pygame.sprite.spritecollide(self, platforms, False)):
            self.velocity_y = self.jump_power

        # Controllo rimpicciolimento
        if keys[pygame.K_s] and not self.shrunk:
            new_width = int(self.rect.width * self.shrink_factor)
            new_height = int(self.rect.height * self.shrink_factor)
            self.image = pygame.transform.scale(self.original_image, (new_width, new_height))
            self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.bottom))
            self.shrunk = True
        elif not keys[pygame.K_s] and self.shrunk:
            # Ripristina le dimensioni originali
            self.image = self.original_image
            self.rect = self.image.get_rect(center=(self.rect.centerx, self.rect.bottom))
            self.shrunk = False

# Classe per rappresentare una piattaforma
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((0, 0, 255))  # Colore blu
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# Inizializzazione dello schermo
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super Mario Bros Simplified")

# Gruppo per gli sprite
all_sprites = pygame.sprite.Group()
platforms = pygame.sprite.Group()

# Creazione di Mario
mario = Mario()
all_sprites.add(mario)

# Creazione delle piattaforme
platform1 = Platform(100, SCREEN_HEIGHT - 100, 200, 20)
platform2 = Platform(400, SCREEN_HEIGHT - 200, 200, 20)
platform3 = Platform(200, SCREEN_HEIGHT - 350, 200, 20)

platforms.add(platform1, platform2, platform3)
all_sprites.add(platform1, platform2, platform3)

# Ciclo di gioco
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Aggiornamento degli sprite
    all_sprites.update()

    # Controllo delle collisioni con le piattaforme
    hits = pygame.sprite.spritecollide(mario, platforms, False)
    for platform in hits:
        # Se Mario sta cadendo, fermo la caduta
        if mario.velocity_y > 0:
            mario.rect.y = platform.rect.y - mario.rect.height
            mario.velocity_y = 0

    # Disegno degli sprite
    screen.fill(WHITE)
    all_sprites.draw(screen)

    # Aggiornamento dello schermo
    pygame.display.flip()

    # Limitazione dei fotogrammi al secondo
    clock.tick(30)

# Chiusura del gioco
pygame.quit()
sys.exit()
