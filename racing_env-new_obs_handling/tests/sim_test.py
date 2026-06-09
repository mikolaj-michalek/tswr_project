import torch
import racing_env
from racing_env.envs.simulators.simulator import Simulator
from racing_env.utils.obs_config import ObservationConfig
import pygame
import datetime



if __name__ == "__main__":
    iter = 10000
    num_envs = 2
    observation_config = ObservationConfig()
    observation_config.load_from_file("obs_config", save_to_wandb=False)
    sim = Simulator(
        vehicle_config="xray",
        tire_config="pacejka",
        dt=0.01,
        integration_method="rk4",
        num_envs=num_envs,
        rand_config=None,
        obs_config=observation_config,
        enable_render=True,
        #enable_render=False,
        delay=False,
        tracks=["icra_2023"],
    )
    iteret = 0

    # print how many points sim.track has
    print("Track points: ", len(sim.track_x_l))

    state_history = []
    # gt_data = import_gt_data("./data/test_opti/iros_xray_30_7.csv")

    sim.screen_renderer.start_render()
    for i in range(iter):
        start_time = datetime.datetime.now()

        ax = 0.0
        ay = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.close()
                sim.screen_renderer.close()

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            ay = 1.0
        if pressed[pygame.K_DOWN]:
            ay = -1.0
        if pressed[pygame.K_LEFT]:
            ax = -1.0
        if pressed[pygame.K_RIGHT]:
            ax = 1.0

        sim.screen_renderer.render([0], sim.closest_idx, sim.state, True)

        u1 = torch.tensor([ax, ay])

        # create u wich osh 100x2
        u = torch.stack([u1 for _ in range(num_envs)], dim=0)

        state = sim.step(u)
        # state_history.append(state)
        obs = sim.calculate_observation()
        reward = sim.calculate_reward()
        # print("Reward shape: ", reward.shape, " obs shape: ", obs.shape)

        end_time = datetime.datetime.now()
        iteret += 1
        # wait 0.1 s
        # while (datetime.datetime.now() - start_time).microseconds / 1000 < 50:
        #     pass
        if iteret % 1024 == 0:
            sim.reset()
        # print("Time taken ms: ", (end_time - start_time).microseconds / 1000)

    # state_hist = torch.stack(state_history)

    sim.close()
