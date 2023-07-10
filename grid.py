# Create the grid
import numpy as np
import pygame

grid_size = 20

# The abstract representation of the grid.

# A nxn grid
grid = np.zeros((grid_size, grid_size))

pygame.init()

screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()


class ClickableTile(pygame.sprite.Sprite):

    def __init__(self, pos, size, state, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((size, size))
        self.state = state
        self.position = position
        if self.state == 0:
            self.image.fill('darkgrey')
        else:
            self.image.fill('white')
        self.rect = self.image.get_rect(topleft=pos)

    def on_click(self):
        if self.state == 0:
            self.image.fill('white')
            self.state = 1
        elif self.state == 1:
            self.image.fill('darkgrey')
            self.state = 0


class GridGenerator:

    def __init__(self):
        self.grid = np.zeros((grid_size, grid_size))
        self.grid[-10:, :] = 1
        self.setup_grid()

    def setup_grid(self):
        self.palette_group = pygame.sprite.Group()
        p_tile = screen_width // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                state = self.grid[i][j]
                tile = ClickableTile(((j * p_tile), (i * p_tile)), p_tile - 1, state, position=(i, j))
                self.palette_group.add(tile)

    def update_grid(self):
        for sprite in self.palette_group.sprites():
            self.grid[sprite.position[0]][sprite.position[1]] = sprite.state

    def print_grid(self):
        print(self.grid)

    def save_grid(self):
        np.save('grid.npy', self.grid)


gridgenerator = GridGenerator()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                for sprite in gridgenerator.palette_group.sprites():
                    if sprite.rect.collidepoint(pos):
                        sprite.on_click()
                        gridgenerator.update_grid()
                        # print(gridgenerator.print_grid())

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            print("saved")
            gridgenerator.save_grid()

    gridgenerator.palette_group.draw(screen)
    # Update the display
    pygame.display.update()
    clock.tick(30)
    pygame.display.flip()


# Quit the game
pygame.quit()




