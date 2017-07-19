package tamaki5_wiki;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class FilePro5 {

	static void fileWrite4(String txt, String filename) {

		try {
			File file = new File(filename);

			if (checkBeforeWritefile(file)) {
				FileWriter filewriter = new FileWriter(file, true);
				//ファイルに追加書き込み，true

				filewriter.write(txt);

				filewriter.close();
			} else {
				System.out.println("ファイル:" + filename + "に書き込めません");
			}

		} catch (IOException e) {
			System.out.println(e);
		}
	}

	static void fileWriteCsv(String txt, String filename) {

		try {
			File fileCsv = new File(filename);

			if (checkBeforeWritefile(fileCsv)) {
				FileOutputStream fos = new FileOutputStream(fileCsv);
				OutputStreamWriter osw = new OutputStreamWriter(fos, "Shift_JIS");

				osw.write(txt);

				osw.close();
			} else {
				System.out.println("ファイル:" + filename + "に書き込めません");
			}

		} catch (IOException e) {
			System.out.println(e);
		}
	}

	static void fileReadandDo(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				while ((line = bufr.readLine()) != null) {
					i++;
					System.out.println(i + "行目を読み込み");
					if (Determine3.IsEngSent(line)) {
						Correct.Process(line);
					} else {
						System.out.println(i + "行目は文でないため飛ばします");
					}

				}

				bufr.close();
			} else {
				Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (IOException e) {
			System.out.println(e);
		}
	}

	//wikipediaコーパス確認，整形用
	static void fileReadandDoForWiki(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int j;
				while ((line = bufr.readLine()) != null) {
					i++;
					System.out.println(i + "行目を読み込み");
					if (Determine3.IsIncludingEndMark(line)) {
						fileWrite4(line + "\n", "WikiSentAll_test.txt"); //ピリオド等含むやつ全て出力
						String St[] = new String[10000];
						St = line.split("(?<=([\\.\\?!]\\s))", 0); //ピリオドとかで分割
						for (j = 0; j < St.length; j++) {
							if (Determine3.IsEngSent(St[j])) { //if文で単語が複数あるのとないの分ける
								fileWrite4(St[j] + "\n", "WikiSentTure1_test.txt"); //あるやつ出力
							} else {
								fileWrite4(St[j] + "\n", "WikiSentFalse1_test.txt"); //ないやつ出力
							}
						}

					} else {//ピリオド等含まないとき
						System.out.println(i + "行目は文ではありません");
						fileWrite4(line + "\n", "WikiNonSent_test.txt");
					}

				}

				bufr.close();
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//wikipediaコーパス例外になる表現発見用
	static void fileReadandDoForWiki2(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int j;
				while ((line = bufr.readLine()) != null) {
					i++;
					System.out.println(i + "行目を読み込み");
					line = line.replaceAll("'", "");
					line = line.replaceAll("\"", "");
					line = line.replaceAll(",", "");
					line = line.replaceAll("-", "");
					line = line.replaceAll("–", "");
					line = line.replaceAll("—", "");
					line = line.replaceAll("\\(", "");
					line = line.replaceAll("\\)", "");
					line = line.replaceAll(";", "");
					line = line.replaceAll(":", "");
					line = line.replaceAll("&", "");
					line = line.replaceAll("  ", " ");

					String St[] = new String[10000];
					St = line.split("(?<=([\\.\\?!]\\s))", 0);
					for (j = 0; j < St.length; j++) {
						if (Determine3.IsNotIncludingMark(St[j])) {

							if (Determine3.IsEngSentTest(St[j])) {
								fileWrite4(St[j] + "\n", "C:\\ruby_bundle\\WikiSentTure_test4.txt");
							} else {
								fileWrite4(St[j] + "\n", "C:\\ruby_bundle\\WikiSentFalse_test4.txt");
								//3単語とスペース2個ないやつ
							}
						} else {
							fileWrite4(St[j] + "\n", "C:\\ruby_bundle\\WikiSentWithMark_test4.txt");
							//余計なマークあるやつ削除
						}
					}

				}

				bufr.close();
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//ピリオド等文末表現をもつものがどれだけあるか
	static void CountSentFromFile(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int piriod = 0;
				int readline = 0;
				int j;
				int flagP;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 100000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					if (line.indexOf(". ") != -1 || line.indexOf("! ") != -1 || line.indexOf("? ") != -1) {
						readline++;
						flagP = 0;

						char[] ch = line.toCharArray();
						for (j = 0; j < ch.length; j++) {
							if (ch[j] == '.' || ch[j] == '!' || ch[j] == '?') {
								flagP = 1;
							} else if (j > 0 && flagP == 1) {
								if (ch[j - 1] == '.' || ch[j - 1] == '!' || ch[j - 1] == '?') {
									if (ch[j] == ' ') {
										piriod++;
									} else {
										flagP = 0;
									}

								} else {
									flagP = 0;

								}

							}

						}

					}

				}

				bufr.close();
				System.out.println("ファイル行数:" + i);
				System.out.println("ピリオド等の数:" + piriod);
				System.out.println("文のある行数:" + readline);
			} else {
				Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//ピリオド等文末表現をもつものがどれだけあるかカウント
	static void CountSentFromFile2(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int piriod = 0;
				int readline = 0;
				int before;
				int diff = 0;
				int difftmp = 0;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 1000000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					//indexOfは正規表現使えない
					if (line.indexOf(". ") != -1 || line.indexOf("! ") != -1 || line.indexOf("? ") != -1) {
						readline++;
						before = line.length();
						//replaceallは正規表現
						line = line.replaceAll("\\. ", "");
						line = line.replaceAll("! ", "");
						line = line.replaceAll("\\? ", "");
						difftmp = before - line.length();
					}
					diff += difftmp;
				}
				piriod = diff / 2;
				bufr.close();
				System.out.println("ファイル行数    :" + i);
				System.out.println("ピリオド等の数*2:" + diff);
				System.out.println("ピリオド等の数  :" + piriod);
				System.out.println("文のある行数    :" + readline);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//ピリオド等文末表現をもつものがどれだけあるかカウント
	static void CountSentFromFile3(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int piriod = 0;
				int readline = 0;

				int difftmp = 0;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 1000000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					//indexOfは正規表現使えない
					if (line.indexOf(". ") != -1 || line.indexOf("! ") != -1 || line.indexOf("? ") != -1) {
						readline++;
						String St[] = new String[10000];
						St = line.split("(?<=([\\.\\?!]\\s))", 0);
						difftmp = St.length;
					}
					piriod += difftmp;
				}

				bufr.close();
				System.out.println("ファイル行数    :" + i);

				System.out.println("ピリオド等の数  :" + piriod);
				System.out.println("文のある行数    :" + readline);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//余計な記号含まないものがどれだけあるかカウント
	static void ExcludeSentWithMark(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int readline = 0;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 100000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					if (Determine3.IsEngSentTest2(line)) {
						readline++;
						fileWrite4(line + "\n", "C:\\ruby_bundle\\WikiSentWithoutMark1.txt");

					}

				}

				bufr.close();
				System.out.println("ファイル行数         :" + i);
				System.out.println("余計な記号の無い行数 :" + readline);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//余計な記号含まないものがどれだけあるかカウント
	static void CheckIncludingWords(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int readline = 0;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 100000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					if (Determine3.IsEngSentTest1(line)) {
						readline++;
						fileWrite4(line + "\n", "C:\\ruby_bundle\\WikiSentWithWords1.txt");

					}

				}

				bufr.close();
				System.out.println("ファイル行数   :" + i);
				System.out.println("単語を含む行数 :" + readline);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	static void CheckIncludingEndMark(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int readline = 0;

				while ((line = bufr.readLine()) != null) {
					i++;
					if (i % 100000 == 0) {
						System.out.println(i + "行目を読み込み");
					}
					if (Determine3.IsIncludingEndMark2(line)) {
						readline++;
						fileWrite4(line + "\n", "C:\\ruby_bundle\\WikiSentWithEndMark1.txt");

					}

				}

				bufr.close();
				System.out.println("ファイル行数         :" + i);
				System.out.println("ピリオド等を含む行数 :" + readline);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	static void makeMiniWiki(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				while ((line = bufr.readLine()) != null) {
					i++;
					if(i<20){
						System.out.println("i:" + i);
					}
					if (i % 60 == 0) {
						fileWrite4(line + "\n", "C:\\cygwin64\\home\\cspc13user\\miniWiki_tmp2.txt");
					}

				}

				bufr.close();
				System.out.println("ファイル行数         :" + i);
			} else {
				//Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	//パターンテスト用
	static void fileReadandPatternTest(File file) {

		try {
			if (checkBeforeReadfile(file)) {
				BufferedReader bufr = new BufferedReader(new FileReader(file));
				String line;
				int i = 0;
				int j;

				String pat1 = "[\\s\\w,;:'\"/\\(\\)][\\s\\w,;:'\"/\\(\\)]+[\\s\\w.,;:'\"/\\(\\)\\.\\?!]";
				//↑これはダメ，-や'や"に対応できてない
				//コンマや()や:はOK，|や[]は正しくはじいている
				String pat2 = "(\\s)*(\\w)+\\s(\\w)+.*";
				//↑これはダメ
				String pat3 = "[\\s\\w.,'\"/\\(\\)]+[\\.\\?!;:][\\s\\w.,;:'\"/\\(\\)\\.\\?!]*";
				//↑これはダメだけど，まぁまぁよさそう？
				//あとは-に対応すればよさそう？
				String pat4 = ".*(\\w)+\\s(\\w)+.*";
				//↑これはダメ
				String pat5 = "[^{}|\\[\\]]+";
				//これはOK？

				fileWrite4(pat1 + "\n\n", "C:\\ruby_bundle\\TestPat1.txt");
				fileWrite4(pat2 + "\n\n", "C:\\ruby_bundle\\TestPat2.txt");
				fileWrite4(pat3 + "\n\n", "C:\\ruby_bundle\\TestPat3.txt");
				fileWrite4(pat4 + "\n\n", "C:\\ruby_bundle\\TestPat4.txt");
				fileWrite4(pat5 + "\n\n", "C:\\ruby_bundle\\TestPat5.txt");

				while ((line = bufr.readLine()) != null) {
					i++;
					System.out.println(i + "行目を読み込み");
					String St[] = new String[10000];
					St = line.split("(?<=([\\.\\?!]\\s))", 0); //ピリオドとかで分割
					for (j = 0; j < St.length; j++) {
						line = Determine3.MatchPattern(St[j], pat1);
						fileWrite4(Determine3.MatchPattern(St[j], pat1), "C:\\ruby_bundle\\TestPat1.txt");
						fileWrite4(Determine3.MatchPattern(St[j], pat2), "C:\\ruby_bundle\\TestPat2.txt");
						fileWrite4(Determine3.MatchPattern(St[j], pat3), "C:\\ruby_bundle\\TestPat3.txt");
						fileWrite4(Determine3.MatchPattern(St[j], pat4), "C:\\ruby_bundle\\TestPat4.txt");
						fileWrite4(Determine3.MatchPattern(St[j], pat5), "C:\\ruby_bundle\\TestPat5.txt");
					}

				}

				bufr.close();
			} else {
				Pref2.LsetText("「" + file.getName() + "」から読み込めません");
				System.out.println("ファイル:" + file.getName() + "にから読み込めません");
			}

		} catch (

		IOException e) {
			System.out.println(e);
		}
	}

	private static boolean checkBeforeWritefile(File file) {
		if (file.exists()) {
			if (file.isFile() && file.canWrite()) {
				return true;
			}
		} else { //ファイル無い時は新規作成，
			try {
				if (file.createNewFile()) {
					System.out.println("ファイル:" + file.getName() + "の作成に成功しました");
					if (file.exists() && file.isFile() && file.canWrite())
						return true;
				} else {
					System.out.println("ファイル:" + file.getName() + "の作成に失敗しました");
				}
			} catch (IOException e) {
				System.out.println(e);
			}

		}

		return false;
	}

	static boolean checkBeforeReadfile(File file) {
		if (file.exists()) {
			if (file.isFile() && file.canRead()) {
				return true;
			}
		}

		return false;
	}

}
