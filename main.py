import pygame
import random
import time

#Creating list of words
words = []
f = open('words.txt', 'r')
for line in f:
    words.append(line.strip())

pygame.font.init()
small_font = pygame.font.SysFont('Comic Sans MS', 30)
big_font = pygame.font.SysFont('Comic Sans MS', 60)
white, black, red = (255, 255, 255), (0, 0, 0), (255, 0, 0)

clock = pygame.time.Clock()

def new_word():
    return random.choice(words)

def lost(screen, word):
    screen.fill(white)
    msg = 'You Lost! The Correct Word was ' + word + '.'
    display = big_font.render(msg, True, black)
    text_rect = display.get_rect(center = (675, 325))
    screen.blit(display, text_rect)
    pygame.display.update()
    time.sleep(3)
    main()

def win(screen, word, error):
    screen.fill(white)
    msg1 = 'You Won! The word was ' + word + '.'
    msg2 = 'You won with ' + str(error) + ' mistakes to spare!'
    display1 = big_font.render(msg1, False, black)
    display2 = big_font.render(msg2, False, black)
    text_rect1 = display1.get_rect(center = (675, 300))
    text_rect2 = display2.get_rect(center = (675, 400))
    screen.blit(display1, text_rect1)
    screen.blit(display2, text_rect2)
    pygame.display.update()
    time.sleep(3)
    main()


def main():
    pygame.init()

    pygame.display.set_caption("hangman")

    screen = pygame.display.set_mode((1350, 700))

    # Initializing hangman place
    screen.fill(white)
    base = pygame.image.load("hangman.png")
    base = pygame.transform.scale(base, (800, 400))
    screen.blit(base, (100, 0))

    # Initializing word and blanks
    word = new_word()
    size = len(word)
    spacing = 75
    for i in range(size):
        pygame.draw.line(screen, black, (i*115 + spacing, 550), (i*115 + 170, 550), 5)
    wrong = 0
    correct = 0
    seen = []

    # Initializing "wrong" letter bank
    bank_title = small_font.render('Wrong Letters', False, black)
    screen.blit(bank_title, (815, 30))
    pygame.draw.rect(screen, black, (675, 85, 500, 250), 3)

    # Initializing body parts
    body_parts = ['dummy circle placeholder', pygame.Rect(440, 195, 0, 145), pygame.Rect(360, 225, 80, 40),
                  pygame.Rect(440, 265, 80, -40), pygame.Rect(440, 340, -60, 80), pygame.Rect(440, 340, 60, 80)]

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # Checking if key pressed is a-z by looking at ASCII value
                if 97 <= event.key <= 122:
                    letter = event.unicode
                    # Going through different scenarios of letter press
                    if letter in seen:
                        msg = "This Letter Has Already Been Guessed!"
                        display = big_font.render(msg, False, red)
                        screen.blit(display, (100, 600))
                        pygame.display.update()
                        time.sleep(1)
                        display2 = big_font.render(msg, False, white)
                        screen.blit(display2, (100, 600))
                    elif letter in word:
                        indices = [i for i, x in enumerate(word) if x == letter]
                        correct += len(indices)
                        display = big_font.render(letter, False, black)
                        for i in indices:
                            screen.blit(display, (i*115 + 110, 475))
                        seen.append(letter)
                    else:
                        wrong += 1
                        if wrong == 1:
                            pygame.draw.circle(screen, black, (440, 150), 45, 5)
                        else:
                            pygame.draw.line(screen, black, body_parts[wrong-1].topleft, body_parts[wrong-1].bottomright, 5)
                        display = big_font.render(letter, False, black)
                        text_rect = display.get_rect(center = (wrong*50 + 660, 115))
                        screen.blit(display, text_rect)
                        seen.append(letter)

            # Ending
            if wrong == 6:
                lost(screen, word)
            elif correct == len(word):
                win(screen, word, 5 - wrong)

        pygame.display.update()

main()