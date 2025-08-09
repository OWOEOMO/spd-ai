/*
 * Pixel Dungeon
 * Copyright (C) 2012-2015 Oleg Dolya
 *
 * Shattered Pixel Dungeon
 * Copyright (C) 2014-2025 Evan Debenham
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

package com.shatteredpixel.shatteredpixeldungeon.desktop;

import com.badlogic.gdx.Files;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3FileHandle;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Preferences;
import com.badlogic.gdx.utils.SharedLibraryLoader;
import com.shatteredpixel.shatteredpixeldungeon.SPDSettings;
import com.shatteredpixel.shatteredpixeldungeon.ShatteredPixelDungeon;
import com.shatteredpixel.shatteredpixeldungeon.services.news.News;
import com.shatteredpixel.shatteredpixeldungeon.services.news.NewsImpl;
import com.shatteredpixel.shatteredpixeldungeon.services.updates.UpdateImpl;
import com.shatteredpixel.shatteredpixeldungeon.services.updates.Updates;
import com.watabou.noosa.Game;
import com.watabou.utils.FileUtils;
import com.watabou.utils.Point;

import org.lwjgl.util.tinyfd.TinyFileDialogs;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.Locale;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

import com.badlogic.gdx.InputProcessor;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.utils.JsonReader;
import com.badlogic.gdx.utils.JsonValue;
import com.badlogic.gdx.utils.Timer;

public class DesktopLauncher {
	private static final com.badlogic.gdx.utils.Json gdxJson = new com.badlogic.gdx.utils.Json();
	static {                           // ← 建議放在類載入時做一次
		gdxJson.setOutputType(com.badlogic.gdx.utils.JsonWriter.OutputType.json);
		gdxJson.setUsePrototypes(false);
	}
	public static void main (String[] args) {

		if (!DesktopLaunchValidator.verifyValidJVMState(args)){
			return;
		}

		//detection for FreeBSD (which is equivalent to linux for us)
		//TODO might want to merge request this to libGDX
		if (System.getProperty("os.name").contains("FreeBSD")) {
			SharedLibraryLoader.isLinux = true;
			//this overrides incorrect values set in SharedLibraryLoader's static initializer
			SharedLibraryLoader.isIos = false;
			SharedLibraryLoader.is64Bit = System.getProperty("os.arch").contains("64") || System.getProperty("os.arch").startsWith("armv8");
		}
		
		final String title;
		if (DesktopLauncher.class.getPackage().getSpecificationTitle() == null){
			title = System.getProperty("Specification-Title");
		} else {
			title = DesktopLauncher.class.getPackage().getSpecificationTitle();
		}
		
		Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
			@Override
			public void uncaughtException(Thread thread, Throwable throwable) {
				Game.reportException(throwable);
				StringWriter sw = new StringWriter();
				PrintWriter pw = new PrintWriter(sw);
				throwable.printStackTrace(pw);
				pw.flush();
				String exceptionMsg = sw.toString();

				//shorten/simplify exception message to make it easier to fit into a message box
				exceptionMsg = exceptionMsg.replaceAll("\\(.*:([0-9]*)\\)", "($1)");
				exceptionMsg = exceptionMsg.replace("com.shatteredpixel.shatteredpixeldungeon.", "");
				exceptionMsg = exceptionMsg.replace("com.watabou.", "");
				exceptionMsg = exceptionMsg.replace("com.badlogic.gdx.", "");
				exceptionMsg = exceptionMsg.replace("\t", "  "); //shortens length of tabs

				//replace ' and " with similar equivalents as tinyfd hates them for some reason
				exceptionMsg = exceptionMsg.replace('\'', '’');
				exceptionMsg = exceptionMsg.replace('"', '”');

				if (exceptionMsg.length() > 1000){
					exceptionMsg = exceptionMsg.substring(0, 1000) + "...";
				}

				if (exceptionMsg.contains("Couldn’t create window")){
					TinyFileDialogs.tinyfd_messageBox(title + " Has Crashed!",
							title + " was not able to initialize its graphics display, sorry about that!\n\n" +
									"This usually happens when your graphics card has misconfigured drivers or does not support openGL 2.0+.\n\n" +
									"If you are certain the game should work on your computer, please message the developer (Evan@ShatteredPixel.com)\n\n" +
									"version: " + Game.version + "\n" +
									exceptionMsg,
							"ok", "error", false);
				} else {
					TinyFileDialogs.tinyfd_messageBox(title + " Has Crashed!",
							title + " has run into an error it cannot recover from and has crashed, sorry about that!\n\n" +
									"If you could, please email this error message to the developer (Evan@ShatteredPixel.com):\n\n" +
									"version: " + Game.version + "\n" +
									exceptionMsg,
							"ok", "error", false);
				}
				System.exit(1);
			}
		});
		
		Game.version = DesktopLauncher.class.getPackage().getSpecificationVersion();
		if (Game.version == null) {
			Game.version = System.getProperty("Specification-Version");
		}
		
		try {
			Game.versionCode = Integer.parseInt(DesktopLauncher.class.getPackage().getImplementationVersion());
		} catch (NumberFormatException e) {
			Game.versionCode = Integer.parseInt(System.getProperty("Implementation-Version"));
		}

		if (UpdateImpl.supportsUpdates()){
			Updates.service = UpdateImpl.getUpdateService();
		}
		if (NewsImpl.supportsNews()){
			News.service = NewsImpl.getNewsService();
		}
		
		Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
		
		config.setTitle( title );
		
		// 關閉 vSync，讓渲染迴圈不受螢幕更新率限制。
		config.useVsync(false);

		// foregroundFPS = 0 取消 LibGDX 的前景限幀。
		config.setForegroundFPS(0);

		// idleFPS = 0 取消背景限幀（視窗不在前景時）。
		config.setIdleFPS(0);

		// 檢查是否傳入 --ai-mode 旗標，日後可改成 headless 啟動。
        boolean aiMode = Arrays.asList(args).contains("--ai-mode");
		System.out.println("[Launcher] aiMode = " + aiMode);

		// 解析 --pos=x,y & --size=W x H
		int winX = -1, winY = -1;
		int winW = 640, winH = 360; // 預設 ai-mode 小窗尺寸
		for (String arg : args) {
			if (arg.startsWith("--pos=")) {
				try {
					String[] xy = arg.substring(6).split(",");
					winX = Integer.parseInt(xy[0].trim());
					winY = Integer.parseInt(xy[1].trim());
				} catch (Exception ignored) {}
			} else if (arg.startsWith("--size=")) {
				try {
					String[] wh = arg.substring(7).toLowerCase(Locale.ROOT).split("x");
					winW = Integer.parseInt(wh[0].trim());
					winH = Integer.parseInt(wh[1].trim());
				} catch (Exception ignored) {}
			}
		}


 		// ------------------------------------------------------------
 		// AI mode：啟動無邊框小視窗（或可改成完全隱藏）以最高速度執行
		if (aiMode) {
			config.setWindowedMode(winW, winH);
			config.setDecorated(false);
			config.setResizable(false);
			if (winX >= 0 && winY >= 0) {
				config.setWindowPosition(winX, winY);
			}
		}

		//if I were implementing this from scratch I would use the full implementation title for saves
		// (e.g. /.shatteredpixel/shatteredpixeldungeon), but we have too much existing save
		// date to worry about transferring at this point.
		String vendor = DesktopLauncher.class.getPackage().getImplementationTitle();
		if (vendor == null) {
			vendor = System.getProperty("Implementation-Title");
		}
		vendor = vendor.split("\\.")[1];

		String basePath = "";
		Files.FileType baseFileType = null;
		if (SharedLibraryLoader.isWindows) {
			if (System.getProperties().getProperty("os.name").equals("Windows XP")) {
				basePath = "Application Data/." + vendor + "/" + title + "/";
			} else {
				basePath = "AppData/Roaming/." + vendor + "/" + title + "/";
			}
			baseFileType = Files.FileType.External;
		} else if (SharedLibraryLoader.isMac) {
			basePath = "Library/Application Support/" + title + "/";
			baseFileType = Files.FileType.External;
		} else if (SharedLibraryLoader.isLinux) {
			String XDGHome = System.getenv("XDG_DATA_HOME");
			if (XDGHome == null) XDGHome = System.getProperty("user.home") + "/.local/share";

			String titleLinux = title.toLowerCase(Locale.ROOT).replace(" ", "-");
			basePath = XDGHome + "/." + vendor + "/" + titleLinux + "/";

			baseFileType = Files.FileType.Absolute;
		}

		config.setPreferencesConfig( basePath, baseFileType );
		SPDSettings.set( new Lwjgl3Preferences( new Lwjgl3FileHandle(basePath + SPDSettings.DEFAULT_PREFS_FILE, baseFileType) ));
		FileUtils.setDefaultFileProperties( baseFileType, basePath );
		
		config.setWindowSizeLimits( 720, 400, -1, -1 );
		Point p = SPDSettings.windowResolution();
		config.setWindowedMode( p.x, p.y );

		config.setMaximized(SPDSettings.windowMaximized());

		//going fullscreen on launch is a bit buggy
		// so game always starts windowed and then switches in DesktopPlatformSupport.updateSystemUI
		//config.setFullscreenMode(Lwjgl3ApplicationConfiguration.getDisplayMode());

		//records whether window is maximized or not for settings
		DesktopWindowListener listener = new DesktopWindowListener();
		config.setWindowListener( listener );
		
		config.setWindowIcon("icons/icon_16.png", "icons/icon_32.png", "icons/icon_48.png",
				"icons/icon_64.png", "icons/icon_128.png", "icons/icon_256.png");

 		/* ============================================================
 		 *  Thread：讀取 STDIN，每行一個 JSON，觸發滑鼠點擊
 		 *  例：{"x":320,"y":240,"button":0}
 		 * ========================================================== */
 		if (aiMode) {
 			new Thread(() -> {
 				com.badlogic.gdx.utils.JsonReader reader = new com.badlogic.gdx.utils.JsonReader();
				
				try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.InputStreamReader(System.in))) {
 					String line;
 					while ((line = br.readLine()) != null) {

						line = line.trim();
						if (line.isEmpty()) continue;          // 空行直接跳過

						System.out.println("[AI] raw input: " + line);  // ← 印收到的字串
						JsonValue j;
						try {
							j = reader.parse(line);
						} catch (Exception ex) {
							System.err.println("[AI] Bad JSON, skip");
							continue;
						}
						if (!j.isObject()) continue;

						if (j.has("cmd")) {
							String cmd = j.getString("cmd", "");
							if ("reset".equals(cmd)) { //{"cmd":"reset"}
								// 在 LibGDX 渲染執行緒執行遊戲邏輯
								Gdx.app.postRunnable(() -> {
									try {
										// 1) 選擇職業：這裡預設 WARRIOR，可改成 MAGE、ROGUE、HUNTRESS
										Class<?> heroEnumCls = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.actors.hero.HeroClass");
										Object warrior = Enum.valueOf(
											(Class<Enum>) heroEnumCls.asSubclass(Enum.class), "WARRIOR");
										Class<?> gipCls = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.GamesInProgress");
										gipCls.getField("selectedClass").set(null, warrior);

										// 2) 清除當前英雄及 Daily 標誌，並重設隨機種子
										Class<?> dungeonCls = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.Dungeon");
										dungeonCls.getField("hero").set(null, null);
										dungeonCls.getField("daily").setBoolean(null, false);
										dungeonCls.getField("dailyReplay").setBoolean(null, false);
										dungeonCls.getMethod("initSeed").invoke(null);

										// 3) 清除任何操作指示
										Class<?> actionIndicatorCls = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.ui.ActionIndicator");
										actionIndicatorCls.getMethod("clearAction").invoke(null);

										// 4) 將 InterlevelScene.mode 設為 DESCEND
										Class<?> interlevelSceneCls = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.scenes.InterlevelScene");
										Class<?> modeEnum = Class.forName(
											"com.shatteredpixel.shatteredpixeldungeon.scenes.InterlevelScene$Mode");
										Object descend = Enum.valueOf(
											(Class<Enum>) modeEnum.asSubclass(Enum.class), "DESCEND");
										interlevelSceneCls.getField("mode").set(null, descend);

										// 5) 切換場景進入地牢第一層
										Class<?> gameCls = Class.forName("com.watabou.noosa.Game");
										gameCls.getMethod("switchScene", Class.class).invoke(null, interlevelSceneCls);

										System.out.println("[AI] reset -> started new game (" + warrior + ")");
									} catch (Exception e) {
										e.printStackTrace();
									}
								});
								continue; // 已處理 reset，跳過後面的滑鼠指令
							}
							if ("get_state".equals(cmd)) { //{"cmd":"get_state"}
								StateSnapshot snap = capture();
								System.out.println("##STATE##" + gdxJson.toJson(snap));
								continue;
							}

						}


 						final int mx = j.getInt("x");
 						// SDL座標左上→LibGDX左下，需反轉 y
 						final int my = Gdx.graphics.getHeight() - j.getInt("y");
 						final int btn = j.getInt("button", 0); // 0=左鍵
 
 						// 必須在渲染執行緒呼叫 input processor
 						com.badlogic.gdx.Gdx.app.postRunnable(() -> {
							System.out.println("[AI] click at (" + mx + ", " + my + "), btn=" + btn);
 							com.badlogic.gdx.InputProcessor ip = com.badlogic.gdx.Gdx.input.getInputProcessor();
 							if (ip != null) {
 								ip.touchDown(mx, my, 0, btn);
 								ip.touchUp(mx, my, 0, btn);
 							}
 						});
 					}
 				} catch (Exception e) {
 					e.printStackTrace();
 				}
 			}, "AI-stdin").start();
 		}

		ShatteredPixelDungeon game = new ShatteredPixelDungeon(new DesktopPlatformSupport()){
			@Override public void create() {
				super.create();
				if (aiMode) watchScene();          // ← 用輪詢的方式監看場景
			}
		};

		new Lwjgl3Application(game, config);   // 只有這裡保留在 main()

	}

	private static void watchScene() {
		Timer.schedule(new Timer.Task() {
			@Override public void run () {
				try {
					Class<?> gameCls = Class.forName("com.watabou.noosa.Game");
					Object   scn     = gameCls.getMethod("scene").invoke(null);
					if (scn == null) return;                        // 尚未初始化

					String simple = scn.getClass().getSimpleName();
					// 只有是 GameScene 時才需要套用
					if ("GameScene".equals(simple)) applyFast(gameCls, scn);
				} catch (Exception e){ e.printStackTrace(); }
			}
		}, 0f, 0.2f);   // 每 0.2 秒檢查一次
	}

	private static void applyFast(Class<?> gameCls, Object gs) throws Exception {
		// 若已經是快轉狀態就略過（避免重複設）
		float ts = gameCls.getField("timeScale").getFloat(null);
		if (ts >= 9.9f) return;

		/* ① 倍速 */
		gameCls.getField("timeScale").setFloat(null, 10f);

		/* ② 關閉節流 */
		java.lang.reflect.Field nd = gs.getClass().getDeclaredField("notifyDelay");
		nd.setAccessible(true);
		nd.setFloat(gs, 0f);

		System.out.println("[FastMode] 重新套用：timeScale=10, notifyDelay=0");
	}

	// ──★ 2.1 乾淨的靜態資料類 ───────────────────────────
	public static class StateSnapshot {
		public int depth, gold, explored;
		public int hp, ht, exp, lvl;
		public boolean alive;
		// public int keys, food, mobs, visibleEnemies;
	}

	// ──★ 2.2 獨立的靜態方法 ────────────────────────────
	private static StateSnapshot capture() throws Exception {
		StateSnapshot s = new StateSnapshot();

		Class<?> dungeon = Class.forName("com.shatteredpixel.shatteredpixeldungeon.Dungeon");
		Object   hero    = dungeon.getField("hero").get(null);
		Class<?> heroCls = hero.getClass();

		s.depth = dungeon.getField("depth").getInt(null);
		s.gold  = dungeon.getField("gold").getInt(null);

		s.hp    = heroCls.getField("HP").getInt(hero);
		s.ht    = heroCls.getField("HT").getInt(hero);

		try {
			s.alive = (boolean) heroCls.getMethod("isAlive").invoke(hero);
		} catch (Exception ex) {
			s.alive = false;          // 保底
			System.err.println("[capture] " + ex);
		}

		s.lvl   = heroCls.getField("lvl").getInt(hero);
		try {
			s.exp = heroCls.getField("exp").getInt(hero);
		} catch (Exception ex) {
			s.exp = 0;          // 保底
			System.err.println("[capture] " + ex);
		}
        Object levelObj = dungeon.getField("level").get(null);
        s.explored = countExplored(levelObj);
		return s;
	}

	// 放在 DesktopLauncher 類內任意位置（例如 main 最前面或最下方）：
	private static int countExplored(Object level) {
		try {
			java.lang.reflect.Field f = level.getClass().getField("visited"); // boolean[]
			boolean[] v = (boolean[]) f.get(level);
			int c = 0; for (boolean b : v) if (b) c++;
			return c;
		} catch (Exception ignored) {}
		try {
			java.lang.reflect.Field f = level.getClass().getField("mapped"); // boolean[]
			boolean[] v = (boolean[]) f.get(level);
			int c = 0; for (boolean b : v) if (b) c++;
			return c;
		} catch (Exception ignored) {}
		return -1; // 取不到就返回 -1
	}


}

