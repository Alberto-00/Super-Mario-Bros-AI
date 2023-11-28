import pygame

CUSTOM_REWARDS = {
    "time": -0.1,  # per second that passes by
    "death": -100.,  # mario dies
    "extra_life": 100.,  # mario gets an extra life, which includes getting 100th coin
    "mushroom": 20.,  # mario eats a mushroom to become big
    "flower": 25.,  # mario eats a flower
    "mushroom_hit": -10.,  # mario gets hit while big
    "flower_hit": -15.,  # mario gets hit while fire mario
    "coin": 15.,  # mario gets a coin
    "score": 15.,  # mario hit enemies
    "victory": 1000  # mario win
}


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((240, 256))
    pygame.display.set_caption("Super Mario Bros")
    return screen


def custom_rewards(name, tmp_info):
    reward = 0

    # detect score
    if tmp_info['score'] != name['score']:
        reward += CUSTOM_REWARDS['score']

    # detect x_pos
    if tmp_info['x_pos'] != name['x_pos']:
        reward += name['x_pos'] - tmp_info['x_pos']

    # detect time
    if tmp_info['time'] != name['time']:
        reward += CUSTOM_REWARDS['time']

    # detect if finished
    if name['x_pos'] > 3159 or (tmp_info['flag_get'] != name['flag_get'] and name['flag_get']):
        print('Vittoria\n')
        reward += CUSTOM_REWARDS['victory']

    # detect deaths
    if 'TimeLimit.truncated' in name and name['x_pos'] < 3159:
        reward += CUSTOM_REWARDS["death"]

    # detect extra lives
    if tmp_info['life'] != name['life'] and name["life"] > 2:
        reward += CUSTOM_REWARDS['extra_life']

    # detect getting a coin
    if tmp_info['coins'] != name['coins']:
        reward += CUSTOM_REWARDS['coin']

        if name["coins"] > 6:
            reward += 500

    # detect if mario ate a mushroom, ate a flower, or got hit without dying
    if tmp_info['status'] != name['status']:
        # 2 - fire mario. only achieved if eating flower while super mario
        # 1 - super mario. only achieved if eating mushroom while small mario
        # 0 - small mario. only achieved if hit while super mario or fire mario. if hit while small mario, death.

        # if small value was sent, you got hit when you were big
        if tmp_info['status'] == 'tall' and name['status'] == 'small':
            reward += CUSTOM_REWARDS['mushroom_hit']

        # or worse, you got hit when you were a flower
        elif tmp_info['status'] == 'fireball' and name['status'] == 'tall':
            reward += CUSTOM_REWARDS['flower_hit']

        # ate a flower (assuming was still super mario. if eating flower while small mario, mario only becomes super
        # mario so this value would be a value of 1, and be caught in the value == 1 checks)
        elif name['status'] == 'fireball':
            reward += CUSTOM_REWARDS['flower']

        # if currently super mario, only need to check if this is from eating mushroom. if hit while fire mario,
        # goes back to small mario
        elif name['status'] == 'tall':
            reward += CUSTOM_REWARDS['mushroom']

    return reward, name


def show_state(enviroment, ep=0, info=""):
    screen = pygame.display.get_surface()
    image = enviroment.render(mode='rgb_array')
    image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    screen.blit(image, (0, 0))
    pygame.display.flip()
    pygame.display.set_caption(f"Episode: {ep} {info}")
    pygame.time.delay(50)  # Aggiungi un ritardo per rallentare la visualizzazione
