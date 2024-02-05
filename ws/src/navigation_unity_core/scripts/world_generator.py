import yaml
import numpy as np
import time


class WorldGenerator:
    def __init__(self, service, world_path, scenes=["forest", "mall"]):
        self.last_world = None
        self.reset_world = service
        self.world_path = world_path
        self.scenes = scenes

    def randomly_change_world(self):
        # for this to work it is expected you have scenes and maps all done in one world
        idx = np.random.choice(np.arange(len(self.scenes)))
        wrld = self.scenes[idx]
        if wrld == self.last_world:
            return
        self.last_world = wrld
        self.change_world(wrld + ".yaml")

        return idx

    def time_based_change_world(self, day_time=0, scene_index=0, teleport=None, fog_density=0.0, day_progress_speed=0.0,
                                lights=False):
        today = day_time % 1
        hours = float(today) * 24.0
        min = (hours % 1) * 60.0
        day_hours = int(hours)
        day_minutes = int(min)
        day_progress_speed = day_progress_speed  # 3500.0
        fog_density = fog_density
        scene = self.scenes[scene_index] + ".yaml"
        world_name = "generated.yaml"
        new_data = {}
        new_data["time_hour"] = day_hours
        new_data["time_minutes"] = day_minutes
        new_data["time_progress_speed"] = day_progress_speed
        new_data["fog_density"] = fog_density
        if fog_density > 0:
            new_data["fog_enabled"] = True
        else:
            new_data["fog_enabled"] = False
        if scene_index == 0:
            spawn_point = 0
        else:
            spawn_point = 0
        if teleport is None:
            new_data["spawn_point_number"] = spawn_point

        self.update_yaml(self.world_path + scene, self.world_path + world_name, new_data)
        self.change_world(world_name)
        return day_minutes, day_hours, day_progress_speed, fog_density, spawn_point

    def update_yaml(self, file_in, file_out, new_yaml_data_dict):
        with open(file_in, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml.update(new_yaml_data_dict)
        with open(file_out, 'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile)

    def change_world(self, world_name):
        wrld_path = self.world_path + world_name
        with open(wrld_path, 'r') as file:
            file_content = file.read()
        self.reset_world(file_content)
        time.sleep(2)
        return
