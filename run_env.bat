cd E:\GithubClone\spd-ai
E:
conda activate E:\GithubClone\spd-ai\my_env




gradlew.bat :desktop:clean :desktop:distZip
java --add-opens java.base/java.lang=ALL-UNNAMED -jar E:/GithubClone/spd-ai/game/shattered-pixel-dungeon/desktop/build/libs/desktop-3.2.0.jar --ai-mode