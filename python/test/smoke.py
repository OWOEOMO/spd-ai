from env.spd_env import SPDEnv

JAR_PATH = "game/shattered-pixel-dungeon/desktop/build/libs/desktop-3.2.0.jar"
CAPTURE_REGION = (1558, 422, 640, 360)  # 替換成實際量測值

env = SPDEnv(jar_path=JAR_PATH, capture_region=CAPTURE_REGION)
obs, _ = env.reset()
print("Initial obs shape:", obs.shape)

# 隨機點擊 10 次
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print("Step:", reward, done)

env.close()
