import pygame
import random

#Creating list of words
words = []
f = open('words.txt', 'r')
for line in f:
    words.append(line.strip())

def main():
    pygame.init()

    pygame.display.set_caption("hangman")

    screen = pygame.display.set_mode((1350, 700))

    white = (255, 255, 255)

    # Initializing hangman place
    screen.fill(white)
    base = pygame.image.load("hangman.png")
    base = pygame.transform.scale(base, (750, 340))
    screen.blit(base, (100, 50))

    # Initializing word and blanks
    def new_word():
        return random.choice(words)

    word = new_word()
    size = len(word)
    line = pygame.image.load("line.png")
    line = pygame.transform.scale(line, (115, 80))
    spacing = 25
    for i in range(size):
        screen.blit(line, (i*115 + spacing, 500))

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()

if __name__ == "__main__":
    main()