import torch
# from mlagents_envs.registry import default_registry
# env = default_registry["Walker"].make()
# env.reset()

# print(env.behavior_specs)
# behavior_name=list(env.behavior_specs)[0]
# print(env.behavior_specs[behavior_name].observation_specs[0].shape)
# print(type(env.behavior_specs[behavior_name].observation_specs[0].shape))
# print(env.behavior_specs[behavior_name].action_spec.continuous_size)

# env.close()
t1 = torch.tensor([
        [
            [11, 12, 13], 
            [21, 22, 23], 
            [31, 32, 33]
        ]
    ])
t2 = t1.argmax(dim=2, keepdim=False)
t3 = t1.argmax(dim=2, keepdim=True)
print(t2)
print(t3)