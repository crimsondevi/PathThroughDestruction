import pygame
import numpy as np
import pod

# Set up the display
screen_width, screen_height = 1000, 600
sub_screen = 600

pygame.init()
pygame.display.set_caption("2D Platformer")
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


class Tile(pygame.sprite.Sprite):

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

    def update_state(self):
        if self.state == 0:
            self.image.fill('darkgrey')
        else:
            self.image.fill('white')


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


class Player(pygame.sprite.Sprite):

    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)
        self.start_pos = pos
        self.image = pygame.Surface((16, 16))
        self.image.fill('red')
        self.rect = self.image.get_rect(topleft=pos)
        self.direction = pygame.math.Vector2(0, 0)
        self.speed = 8
        self.gravity = 1
        self.jump_speed = -10
        self.grounded = False
        self.NO_MOVEMENT = False

    def get_input(self):
        if self.NO_MOVEMENT:
            return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            self.direction.x = 1
        elif keys[pygame.K_LEFT]:
            self.direction.x = -1
        else:
            self.direction.x = 0

        if keys[pygame.K_SPACE] and self.grounded:
            self.jump()

    def apply_gravity(self):
        self.direction.y += self.gravity
        self.rect.y += self.direction.y

    def jump(self):
        self.direction.y = self.jump_speed

    def update(self):
        self.get_input()

    def turn_green(self):
        self.image.fill('green')

    def reset(self):
        self.rect.x = self.start_pos[0]
        self.rect.y = self.start_pos[1]
        self.image.fill('red')


# Define grid properties
grid_size = 10  # N
cell_size = screen_width // grid_size

class Level:
    def __init__(self, surface):
        self.level_number = 1;
        self.display_surface = surface
        self.grid = self.get_level_grid()
        self.tile_size = sub_screen / self.grid.shape[0]
        self.player = pygame.sprite.GroupSingle()
        self.tiles = pygame.sprite.Group()
        self.setup_level()
        self.player_max_x = sub_screen - self.player.sprite.rect.width
        self.level_complete = False

    def get_level_grid(self):
        level_file = 'level{}.npy'.format(self.level_number)
        return np.load(level_file)

    def setup_level(self):
        for y, column in enumerate(self.grid):
            for x, item in enumerate(column):
                state = self.grid[y][x]
                position = (y, x)
                tile = Tile((x * self.tile_size, y * self.tile_size), self.tile_size - 1, state, position)
                self.tiles.add(tile)

        self.player.add(Player((0, self.tile_size * 10)))

    def update_level(self):
        for tile_sprite in self.tiles.sprites():
            tile_sprite.state = self.grid[tile_sprite.position[0]][tile_sprite.position[1]]
            tile_sprite.update_state()

    def horizontal_movement_collision(self):
        player = self.player.sprite
        x_inc = player.direction.x * player.speed
        max_x = self.player_max_x
        player.rect.x = max(0, min(max_x, player.rect.x + x_inc))

        for sprite in self.tiles.sprites():
            if sprite.state == 0:
                continue
            if sprite.rect.colliderect(player.rect):
                if player.direction.x < 0:
                    player.rect.left = sprite.rect.right
                elif player.direction.x > 0:
                    player.rect.right = sprite.rect.left

    def vertical_movement_collision(self):
        player = self.player.sprite
        player.apply_gravity()
        player.grounded = False

        for sprite in self.tiles.sprites():
            if sprite.state == 0:
                continue
            if sprite.rect.colliderect(player.rect):
                if player.direction.y > 0:
                    player.rect.bottom = sprite.rect.top
                    player.direction.y = 0
                    player.grounded = True
                elif player.direction.y < 0:
                    player.rect.top = sprite.rect.bottom
                    player.direction.y = 0

    def check_win(self):
        player = self.player.sprite
        if player.rect.x == self.player_max_x:
            return True

    def get_grid(self):
        return self.grid

    def set_grid(self, grid):
        self.grid = grid

    def reset_grid(self):
        self.grid = self.get_level_grid()
        player = self.player.sprite
        player.reset()
        self.update_level()

    def load_next_level(self):
        self.level_number += 1
        self.level_complete = False
        self.reset_grid()

    def run(self):
        self.tiles.draw(self.display_surface)

        self.player.update()
        self.horizontal_movement_collision()
        self.vertical_movement_collision()
        self.player.draw(self.display_surface)

        # TODO: Create proper win condition and screen
        if not self.level_complete and self.check_win():
            print("win")
            player = self.player.sprite
            player.turn_green()
            self.level_complete = True


class TilePalette:
    def __init__(self):
        self.grid = np.zeros((5, 5))
        self.palette_size = 200
        self.setup_palette()

    def setup_palette(self):
        p_tile = self.palette_size // 5
        x_start = 650
        y_start = 30
        self.palette_group = pygame.sprite.Group()

        for i in range(5):
            for j in range(5):
                state = self.grid[i][j]
                tile = ClickableTile((x_start + (j * p_tile), y_start + (i * p_tile)), p_tile - 1, state, position=(i, j))
                self.palette_group.add(tile)

    def update_palette_grid(self):
        for sprite in self.palette_group.sprites():
            self.grid[sprite.position[0]][sprite.position[1]] = sprite.state

    def get_palette(self):
        return self.grid


class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        font = pygame.font.Font(None, 30)
        text = font.render(self.text, True, (255, 255, 255))
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)


ResetButton = Button(700, 310, 200, 50, "Reset", (0, 0, 255))
NextButton = Button(700, 380, 200, 50, "Next Level", (0, 255, 255))
ApplyInference = Button(700, 450, 200, 50, "Apply", (0, 255, 0))
TrainButton = Button(700, 520, 200, 50, "Train", (255, 0, 0))


palette = TilePalette()
level = Level(screen)

model = pod.PoD()


# Game loop
current_level = None
count = 0
COUNT = 30
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                if ResetButton.rect.collidepoint(pos):
                    level.reset_grid()
                
                if level.level_complete and NextButton.rect.collidepoint(pos):
                    level.load_next_level()

                elif ApplyInference.rect.collidepoint(pos):
                    level.reset_grid()
                    current_level = level.grid
                    count = COUNT

                elif TrainButton.rect.collidepoint(pos):
                    TrainButton.color = (100, 100, 100)
                    TrainButton.draw(screen)
                    pygame.display.update()
                    sample = palette.grid
                    model.add_goal(sample)
                    model.train()
                    TrainButton.color = (255, 0, 0)
                    TrainButton.draw(screen)
                    pygame.display.update()

                else:
                    for sprite in palette.palette_group.sprites():
                        if sprite.rect.collidepoint(pos):
                            sprite.on_click()
                            palette.update_palette_grid()

    # Clear the screen
    screen.fill(BLACK)

    # General Logic
    level.run()

    level.player.sprite.NO_MOVEMENT = False

    if count > 0:
        count -= 1
        new_level = model.infer(current_level, rounds=150)
        current_level = new_level
        level.grid = new_level
        level.update_level()
        level.player.sprite.NO_MOVEMENT = True

    # Render Buttons and Palette
    ResetButton.draw(screen)
    if level.level_complete: NextButton.draw(screen)
    ApplyInference.draw(screen)
    TrainButton.draw(screen)
    palette.palette_group.draw(screen)

    # Update the display
    pygame.display.update()
    clock.tick(30)
    pygame.display.flip()


# Quit the game
pygame.quit()
