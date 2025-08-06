from env.spd_env import SPDEnv

JAR_PATH = "../game/shattered-pixel-dungeon/desktop/build/libs/desktop-3.2.0.jar"
CAPTURE_REGION = (1558, 422, 720, 1220)  # 替換成實際量測值

env = SPDEnv(jar_path=JAR_PATH, capture_region=CAPTURE_REGION)
obs, _ = env.reset()
print("Initial obs shape:", obs.shape)

# 隨機點擊 100 次
for i in range(100):
    action = env.action_space.sample()
    obs, r, done, trunc, info = env.step(action)

    if i % 10 == 0:          # 只每 10 步看一次畫面
        env.render()

env.close()
